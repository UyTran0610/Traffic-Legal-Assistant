import sqlite3
import os
import re
import shutil
import uuid
import threading
import subprocess
import time
import atexit
import ctypes
import ctypes.wintypes
import signal
import urllib.request
import traceback
import numpy as np
import concurrent.futures
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from transformers import AutoTokenizer
import onnxruntime as ort

from config import (
    DATA_PATH, DB_PATH, DB_FILE,
    EMBEDDING_MODEL_NAME, EMBEDDING_ONNX_MODEL_NAME, EMBEDDING_TOKENIZER_PATH,
    RERANKER_MODEL_NAME, RERANKER_ONNX_MODEL_NAME,
    LLM_MODEL_PATH, LLAMA_SERVER_PATH,
    LLAMA_SERVER_PORT, CHUNK_SIZE, CHUNK_OVERLAP, RERANK_THRESHOLD,
    SYSTEM_PROMPT, QUERY_REWRITE_PROMPT, LEGAL_QUERY_PROMPT,
)

AMBIGUOUS_PATTERNS = ["còn ", "thế còn ", "vậy còn ", "thì sao", "thế nào", "như vậy thì"]

# ==========================================
# STATE TOÀN CỤC CỦA AI & SERVER NGẦM
# ==========================================
class AIState:
    vector_store = None
    reranker_model = None
    llm_chain = None
    llm_instance = None
    server_process = None
    is_ready = False
    init_error = None
    rewrite_chain = None   # dùng trong rewrite_query_with_history()
    legal_chain   = None   # dùng trong generate_legal_queries()

ai_state = AIState()


# ==========================================
# ONNX EMBEDDING
# ==========================================
class ONNXEmbedding(Embeddings):
    """
    LangChain-compatible embedding wrapper chạy thuần ONNX Runtime trên CPU.
    Tương thích với AITeamVN/Vietnamese_Embedding (kiến trúc encoder / feature-extraction).
    """

    def __init__(self, model_path: str, tokenizer_path: str = None, max_length: int = 256, batch_size: int = 16):
        print(f"   Đang load ONNX Embedding từ: {model_path}")
        tok_path = tokenizer_path or model_path
        
        # --- ÁP DỤNG FAST TOKENIZER ---
        self.tokenizer  = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
        
        # --- THÊM TỐI ƯU HÓA ONNX GRAPH & THREAD TUNING ---
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 0  # 0 để ONNX Runtime tự quyết định số luồng tối ưu theo CPU
        
        self.session    = ort.InferenceSession(
            os.path.join(model_path, "model.onnx"),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.max_length  = max_length
        self.batch_size  = batch_size
        print(f"   ✅ ONNX Embedding sẵn sàng (CPU) | max_len={max_length} | batch={batch_size}")

    # ── nội bộ ──────────────────────────────────────────────
    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        feed = {k: v.astype(np.int64) for k, v in enc.items() if k in self.input_names}
        outputs = self.session.run(None, feed)

        last_hidden = outputs[0]                        # (B, L, H)
        attention_mask = enc["attention_mask"]          # (B, L)

        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)  # (B, L, 1)
        sum_hidden    = (last_hidden * mask_expanded).sum(axis=1)             # (B, H)
        count         = mask_expanded.sum(axis=1).clip(min=1e-9)              # (B, 1)
        embeddings    = sum_hidden / count                                    # (B, H)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-9)
        return (embeddings / norms).astype(np.float32)

    def _encode_all(self, texts: List[str]) -> List[List[float]]:
        all_vecs: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            all_vecs.append(self._encode_batch(batch))
        result = np.vstack(all_vecs) if all_vecs else np.empty((0,), dtype=np.float32)
        return result.tolist()

    # ── LangChain Embeddings interface ──────────────────────
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode_all(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._encode_all([text])[0]


# ==========================================
# ONNX RERANKER
# ==========================================
class ONNXReranker:
    """
    Thay thế HuggingFaceCrossEncoder, chạy thuần CPU qua ONNX Runtime (không dùng optimum).
    Tương thích với file model.onnx được export thủ công bằng torch.onnx.export.
    """

    def __init__(self, model_path: str):
        print(f"   Đang load ONNX Reranker từ: {model_path}")
        
        # --- ÁP DỤNG FAST TOKENIZER ---
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        
        # --- THÊM TỐI ƯU HÓA ONNX GRAPH & THREAD TUNING ---
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 0  # 0 để ONNX Runtime tự quyết định số luồng tối ưu theo CPU
        
        self.session   = ort.InferenceSession(
            os.path.join(model_path, "model.onnx"),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {i.name for i in self.session.get_inputs()}
        print(f"   ✅ ONNX Reranker sẵn sàng (CPU) | inputs: {self.input_names}")

    def score(self, pairs: list) -> list:
        if not pairs:
            return []
        
        all_scores =[]
        BATCH_SIZE = 8
        
        for i in range(0, len(pairs), BATCH_SIZE):
            batch_pairs = pairs[i:i+BATCH_SIZE]
            queries  = [p[0] for p in batch_pairs]
            passages = [p[1] for p in batch_pairs]

            enc = self.tokenizer(
                queries, passages,
                padding=True, truncation=True, max_length=512,
                return_tensors="np",
            )
            feed = {k: v.astype(np.int64) for k, v in enc.items() if k in self.input_names}
            if "token_type_ids" in self.input_names and "token_type_ids" not in feed:
                feed["token_type_ids"] = np.zeros_like(feed["input_ids"], dtype=np.int64)

            outputs = self.session.run(None, feed)
            logits  = outputs[0].reshape(-1).astype(np.float32)
            scores = (1.0 / (1.0 + np.exp(-logits))).tolist()
            all_scores.extend(scores)
            
        return all_scores


# ==========================================
# QUẢN LÝ TIẾN TRÌNH LLAMA-SERVER
# ==========================================

_job_handle = None 

def _create_job_object():
    global _job_handle
    if os.name != 'nt':
        return None

    kernel32 = ctypes.windll.kernel32
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        print("   ⚠️  Không thể tạo Job Object, fallback sang atexit")
        return None

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit",     ctypes.c_int64),
            ("LimitFlags",             ctypes.c_uint32),
            ("MinimumWorkingSetSize",  ctypes.c_size_t),
            ("MaximumWorkingSetSize",  ctypes.c_size_t),
            ("ActiveProcessLimit",     ctypes.c_uint32),
            ("Affinity",               ctypes.c_size_t),
            ("PriorityClass",          ctypes.c_uint32),
            ("SchedulingClass",        ctypes.c_uint32),
        ]

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [("x", ctypes.c_uint64 * 6)]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo",                IO_COUNTERS),
            ("ProcessMemoryLimit",    ctypes.c_size_t),
            ("JobMemoryLimit",        ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed",     ctypes.c_size_t),
        ]

    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

    JobObjectExtendedLimitInformation = 9
    ok = kernel32.SetInformationJobObject(
        job,
        JobObjectExtendedLimitInformation,
        ctypes.byref(info),
        ctypes.sizeof(info),
    )
    if not ok:
        print("   ⚠️  SetInformationJobObject thất bại, fallback sang atexit")
        kernel32.CloseHandle(job)
        return None

    _job_handle = job 
    print("   ✅ Windows Job Object đã tạo — llama-server sẽ tự tắt khi app đóng")
    return job

def _assign_process_to_job(job_handle, proc: subprocess.Popen):
    if os.name != 'nt' or job_handle is None:
        return False
    kernel32 = ctypes.windll.kernel32
    ok = kernel32.AssignProcessToJobObject(job_handle, int(proc._handle))
    if ok:
        print("   ✅ llama-server đã được gắn vào Job Object")
    else:
        print("   ⚠️  AssignProcessToJobObject thất bại (có thể server đã trong Job khác)")
    return bool(ok)

_watchdog_stop   = threading.Event()
# Lock bảo vệ đoạn khởi động lại tiến trình — ngăn Watchdog spawn tiến trình mới
# đúng lúc stop_llama_server() đang dọn dẹp (tránh orphaned process).
_restart_lock    = threading.Lock()

def _watchdog_loop(job_handle, cmd, creationflags):
    from config import LLAMA_SERVER_PORT
    PING_URL = f"http://127.0.0.1:{LLAMA_SERVER_PORT}/v1/models"
    PING_INTERVAL   = 5    
    MAX_FAIL_BEFORE_RESTART = 3  

    consecutive_fail = 0
    print("🔍 Watchdog đã khởi động — theo dõi llama-server...")

    while not _watchdog_stop.is_set():
        _watchdog_stop.wait(PING_INTERVAL)
        if _watchdog_stop.is_set():
            break

        proc = ai_state.server_process
        if proc is None:
            continue

        process_alive = (proc.poll() is None)

        http_ok = False
        try:
            req = urllib.request.Request(PING_URL)
            req.add_header("Connection", "close")
            with urllib.request.urlopen(req, timeout=3) as r:
                http_ok = (r.status == 200)
        except Exception:
            pass

        if http_ok:
            consecutive_fail = 0
            continue

        consecutive_fail += 1
        print(f"   ⚠️  [Watchdog] Ping thất bại lần {consecutive_fail}/{MAX_FAIL_BEFORE_RESTART} "
              f"(process_alive={process_alive})")

        if consecutive_fail < MAX_FAIL_BEFORE_RESTART:
            continue

        print("   🔄 [Watchdog] Server không phản hồi — đang khởi động lại...")
        consecutive_fail = 0

        if process_alive:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try: proc.kill()
                except Exception: pass

        # ── Kiểm tra cờ dừng TRƯỚC khi spawn tiến trình mới.
        # Dùng _restart_lock để tránh race với stop_llama_server() đang chạy
        # song song: nếu stop đã set cờ và đang chờ lock thì ta không spawn nữa.
        if _watchdog_stop.is_set():
            break

        try:
            with _restart_lock:
                # Kiểm tra lần hai sau khi đã vào lock — stop_llama_server có thể
                # đã set cờ ngay trong khoảng ta đang đợi acquire lock.
                if _watchdog_stop.is_set():
                    break

                new_proc = subprocess.Popen(
                    cmd,
                    creationflags=creationflags,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                ai_state.server_process = new_proc

            if job_handle:
                _assign_process_to_job(job_handle, new_proc)

            deadline = time.time() + 60
            while time.time() < deadline and not _watchdog_stop.is_set():
                try:
                    with urllib.request.urlopen(
                        urllib.request.Request(PING_URL), timeout=2
                    ) as r:
                        if r.status == 200:
                            print("   ✅ [Watchdog] Server đã restart thành công!")
                            break
                except Exception:
                    pass
                time.sleep(2)
        except Exception as ex:
            print(f"   ❌ [Watchdog] Không thể restart server: {ex}")

    print("🔍 Watchdog đã dừng.")

def start_llama_server():
    from config import LLAMA_SERVER_PORT, LLAMA_SERVER_PATH, LLM_MODEL_PATH 

    if not os.path.exists(LLAMA_SERVER_PATH):
        raise FileNotFoundError(f"Không tìm thấy file {LLAMA_SERVER_PATH}!")
    if not os.path.exists(LLM_MODEL_PATH):
        raise FileNotFoundError(f"Không tìm thấy model tại {LLM_MODEL_PATH}!")

    try:
        if os.name == 'nt':
            subprocess.run(
                ["taskkill", "/f", "/im", os.path.basename(LLAMA_SERVER_PATH)],
                capture_output=True
            )
            time.sleep(0.5)  
    except Exception:
        pass

    print(f"🚀 Khởi động Engine AI ngầm (Cổng: {LLAMA_SERVER_PORT})...")

    cmd = [
        LLAMA_SERVER_PATH,
        "-m", LLM_MODEL_PATH,
        "-c", "6144",
        "-b", "256",
        "--ubatch-size", "256",
        "-t", "4",
        "--port", str(LLAMA_SERVER_PORT),
    ]
    creationflags = 0x08000000 if os.name == 'nt' else 0

    job_handle = _create_job_object()

    ai_state.server_process = subprocess.Popen(
        cmd,
        creationflags=creationflags,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if job_handle:
        _assign_process_to_job(job_handle, ai_state.server_process)

    print("⏳ Đang tải mô hình ngôn ngữ vào bộ nhớ (có thể mất 10-30 giây)...")
    PING_URL = f"http://127.0.0.1:{LLAMA_SERVER_PORT}/v1/models"
    start_time = time.time()
    server_ready = False

    while time.time() - start_time < 120:
        if ai_state.server_process.poll() is not None:
            raise RuntimeError(
                "llama-server.exe đã thoát ngay sau khi khởi động. "
                "Kiểm tra lại đường dẫn model hoặc VRAM."
            )
        try:
            req = urllib.request.Request(PING_URL)
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    server_ready = True
                    print("✅ Llama Server đã tải model thành công!")
                    break
        except Exception:
            time.sleep(1)

    if not server_ready:
        raise RuntimeError(
            "Quá thời gian chờ Llama Server khởi động. "
            "Vui lòng kiểm tra lại cấu hình RAM/VRAM."
        )

    _watchdog_stop.clear()
    t = threading.Thread(
        target=_watchdog_loop,
        args=(job_handle, cmd, creationflags),
        daemon=True,
        name="llama-watchdog",
    )
    t.start()


def stop_llama_server():
    # Set cờ dừng TRƯỚC — Watchdog sẽ không bắt đầu restart mới sau bước này.
    _watchdog_stop.set()

    # Acquire lock để đảm bảo không có tiến trình mới nào đang được spawn
    # trong _watchdog_loop ngay lúc ta dọn dẹp.
    with _restart_lock:
        proc = ai_state.server_process
        if proc and proc.poll() is None:
            print("🛑 Đang tắt Engine AI...")
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        ai_state.server_process = None

atexit.register(stop_llama_server)

def _sigint_handler(sig, frame):
    stop_llama_server()
    raise SystemExit(0)

try:
    signal.signal(signal.SIGINT,  _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)
except (OSError, ValueError):
    pass  


# ==========================================
# KHỞI TẠO EMBEDDING (ONNX)
# ==========================================
def _build_embedding_function():
    onnx_path = EMBEDDING_ONNX_MODEL_NAME
    tok_path  = EMBEDDING_TOKENIZER_PATH

    if os.path.exists(os.path.join(onnx_path, "model.onnx")):
        print(f"   [Embedding] ✅ Dùng ONNX built-in: {onnx_path}")
        print(f"   [Embedding]    Tokenizer từ      : {tok_path}")
        return ONNXEmbedding(onnx_path, tokenizer_path=tok_path, max_length=1024, batch_size=8)

    raise FileNotFoundError(f"Không tìm thấy ONNX Embedding tại: {os.path.join(onnx_path, 'model.onnx')}. Bạn đang tắt chế độ dùng PyTorch nên bắt buộc phải có ONNX.")


# ==========================================
# XỬ LÝ TÀI LIỆU & VECTOR DB
# ==========================================
def load_and_process_documents():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        return []

    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyMuPDFLoader)
    documents = loader.load()
    if not documents: return []

    for doc in documents:
        cleaned_text = re.sub(r'(?i)\d*\s*CÔNG BÁO/Số.*?\d{4}\s*\d*', '', doc.page_content)
        doc.page_content = cleaned_text.strip()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\nĐiều ", "\nChương ", "\nPhần ", "\n\n", "\n", ". ", " "],
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

def init_vector_db():
    print(f"🧠 Khởi tạo Embedding Model...")
    embedding_func = _build_embedding_function()

    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        try:
            print(f"   [VectorDB] Tải ChromaDB từ cache: {DB_PATH}")
            return Chroma(
                persist_directory=DB_PATH,
                embedding_function=embedding_func,
                collection_metadata={"hnsw:space": "ip"}
            )
        except Exception as e:
            print(f"   [VectorDB] Cache lỗi ({e}), xây dựng lại...")
            shutil.rmtree(DB_PATH)

    print(f"   [VectorDB] Đang xử lý tài liệu PDF từ: {DATA_PATH}")
    chunks = load_and_process_documents()
    if not chunks:
        print(f"   [VectorDB] ⚠️ Không tìm thấy tài liệu PDF nào trong {DATA_PATH}")
        return None

    print(f"   [VectorDB] Tổng số chunks: {len(chunks)} — Đang nạp vào ChromaDB...")
    db = Chroma(
        embedding_function=embedding_func,
        persist_directory=DB_PATH,
        collection_metadata={"hnsw:space": "ip"}
    )
    total = len(chunks)
    for i in range(0, total, 50):
        try:
            db.add_documents(chunks[i : i + 50])
            print(f"   [VectorDB] Đã nạp {min(i + 50, total)}/{total} chunks...")
        except Exception as e:
            print(f"   [VectorDB] ⚠️ Lỗi batch {i}–{min(i+50, total)}: {e} — bỏ qua batch này")
            continue
    print(f"   [VectorDB] ✅ Hoàn tất nạp {total} chunks vào ChromaDB.")
    return db


# ==========================================
# RAG PIPELINE
# ==========================================
MAX_HISTORY_TURNS = 3

def format_chat_history(messages: list, for_rewrite: bool = False) -> str:
    if not messages or len(messages) <= 1: return "(Chưa có lịch sử trò chuyện)"
    history = messages[:-1]
    if for_rewrite:
        history = history[-2:]
    else:
        max_msgs = MAX_HISTORY_TURNS * 2
        if len(history) > max_msgs: history = history[-max_msgs:]

    lines = [f"{'Người dùng' if m['role'] == 'user' else 'Trợ lý'}: {m['text']}" for m in history]
    return "\n".join(lines)

def rewrite_query_with_history(original_query: str, messages: list) -> str:
    query_lower = original_query.lower().strip()
    is_ambiguous = (len(original_query.split()) <= 8 and any(p in query_lower for p in AMBIGUOUS_PATTERNS))

    if not is_ambiguous:
        print(f"   [Rewrite] Câu hỏi rõ ràng, giữ nguyên: \"{original_query}\"")
        return original_query

    rewrite_history = format_chat_history(messages, for_rewrite=True)
    if rewrite_history == "(Chưa có lịch sử trò chuyện)":
        print(f"   [Rewrite] Không có lịch sử, giữ nguyên: \"{original_query}\"")
        return original_query

    print(f"   [Rewrite] Câu hỏi mơ hồ — đang viết lại với lịch sử 1 lượt trước...")
    try:
        result = ai_state.rewrite_chain.invoke({
            "chat_history": rewrite_history,
            "question": original_query
        }).strip()
        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()
        rewritten = result.replace('"', '').split("\n")[0].strip() or original_query
        print(f"   [Rewrite] ✅ Đã viết lại: \"{rewritten}\"")
        return rewritten
    except Exception as e:
        print(f"   [Rewrite] ⚠️ Lỗi rewrite: {e} — giữ nguyên câu gốc")
        return original_query

def generate_legal_queries(original_query: str) -> str:
    print(f"   [LegalTerm] Đang chuyển đổi sang thuật ngữ pháp lý...")
    try:
        result = ai_state.legal_chain.invoke({"question": original_query}).strip()
        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()
        legal_term = result.replace('"', '').replace("Output:", "").strip()
        print(f"   [LegalTerm] ✅ Thuật ngữ pháp lý: \"{legal_term}\"")
        return legal_term
    except Exception as e:
        print(f"   [LegalTerm] ⚠️ Lỗi generate legal term: {e} — dùng câu gốc")
        return original_query

def retrieve_context(query: str, top_k: int = 3, messages: list = None):
    messages = messages or []
    print(f"\n{'='*60}")
    print(f"📥 Câu hỏi gốc: \"{query}\"")
    print(f"{'='*60}")

    full_history = format_chat_history(messages, for_rewrite=False)

    query_lower = query.lower().strip()
    is_ambiguous = (len(query.split()) <= 8 and any(p in query_lower for p in AMBIGUOUS_PATTERNS))

    if is_ambiguous:
        print(f"   [Query] Câu hỏi mơ hồ — rewrite trước, legal term sau...")
        rewritten_query = rewrite_query_with_history(query, messages)
        legal_term = generate_legal_queries(rewritten_query)
    else:
        # Câu rõ ràng: bỏ qua rewrite, chỉ gọi 1 lần legal term.
        print(f"   [Query] Câu hỏi rõ ràng — bỏ qua rewrite, chỉ gọi legal term...")
        rewritten_query = query
        legal_term = generate_legal_queries(query)

    print(f"\n🔍 Tìm kiếm vector (k=10)...")
    print(f"   Query A (rewritten): \"{rewritten_query}\"")
    docs_original = ai_state.vector_store.similarity_search(rewritten_query, k=10)
    print(f"   → Tìm thấy {len(docs_original)} docs")

    if legal_term != rewritten_query:
        print(f"   Query B (legal term): \"{legal_term}\"")
        docs_legal = ai_state.vector_store.similarity_search(legal_term, k=10)
        print(f"   → Tìm thấy {len(docs_legal)} docs")
    else:
        docs_legal = []
        print(f"   Query B bỏ qua (giống Query A)")

    seen = set()
    all_docs = [d for d in docs_original + docs_legal if not (d.page_content in seen or seen.add(d.page_content))]
    print(f"   → Tổng sau dedup: {len(all_docs)} docs")

    if not all_docs:
        print("   ⚠️ Không tìm thấy tài liệu nào liên quan.")
        return None, legal_term, full_history

    if ai_state.reranker_model:
        print(f"\n⚖️  Reranking {len(all_docs)} docs (ngưỡng lọc: {RERANK_THRESHOLD})...")

        use_two_queries = legal_term and legal_term != rewritten_query

        if use_two_queries:
            n = len(all_docs)
            batch_pairs = (
                [[rewritten_query, d.page_content] for d in all_docs] +
                [[legal_term,      d.page_content] for d in all_docs]
            )
            all_scores = ai_state.reranker_model.score(batch_pairs)
            scores_A = all_scores[:n]
            scores_B = all_scores[n:]
            final_scores = [max(s1, s2) for s1, s2 in zip(scores_A, scores_B)]
            print(f"   Batch rerank 2 queries × {n} docs = {2*n} pairs (1 lần gọi model)")
        else:
            scores_A = ai_state.reranker_model.score([[rewritten_query, d.page_content] for d in all_docs])
            final_scores = list(scores_A)
            print(f"   Rerank 1 query × {len(all_docs)} docs")

        doc_score_pairs = sorted(zip(all_docs, final_scores), key=lambda x: x[1], reverse=True)
        filtered = [p for p in doc_score_pairs if p[1] > RERANK_THRESHOLD]
        final_docs = [p[0] for p in filtered[:top_k]] if filtered else [p[0] for p in doc_score_pairs[:3]]

        print(f"   → Sau lọc ngưỡng {RERANK_THRESHOLD}: {len(filtered)} docs")
        print(f"   → Sử dụng top {len(final_docs)} docs:")
        for i, (doc, score) in enumerate(doc_score_pairs[:len(final_docs)]):
            src = doc.metadata.get('source', 'N/A')
            page = doc.metadata.get('page', 'N/A')
            print(f"      [{i+1}] Score={score:.4f} | {os.path.basename(str(src))} - Tr.{page}")
    else:
        print(f"\n⚠️  Reranker không khả dụng, lấy top {top_k} docs từ vector search")
        final_docs = all_docs[:top_k]

    context_str = "\n\n".join(
        f"[Nguồn: {d.metadata.get('source')} - Tr.{d.metadata.get('page')}]\n{d.page_content}"
        for d in final_docs
    )
    print(f"\n✅ Đã chuẩn bị context ({len(final_docs)} đoạn văn bản)")
    print(f"{'='*60}\n")
    return context_str, legal_term, full_history


# ==========================================
# KHỞI TẠO TOÀN BỘ HỆ THỐNG AI
# ==========================================
def initialize_ai_system():
    try:
        ai_state.vector_store = init_vector_db()

        print(f"⚖️  Khởi tạo Reranker (ONNX)...")
        onnx_path = RERANKER_ONNX_MODEL_NAME
        
        if os.path.exists(onnx_path):
            ai_state.reranker_model = ONNXReranker(onnx_path)
        else:
            raise FileNotFoundError(f"Chưa tìm thấy model ONNX tại '{onnx_path}'. Bạn đang tắt PyTorch nên bắt buộc phải có model ONNX để chạy Reranker.")

        start_llama_server()

        print(f"🔗 Đang kết nối LangChain vào Llama Server...")

        _no_think_params = {
            "chat_template_kwargs": {"enable_thinking": False},
            "top_k": 20,
            "min_p": 0.0,
        }

        ai_state.llm_instance = ChatOpenAI(
            base_url=f"http://127.0.0.1:{LLAMA_SERVER_PORT}/v1",
            api_key="sk-no-key-required",
            model="qwen",
            temperature=0.2,
            top_p=0.8,
            streaming=False,
            timeout=60,
            model_kwargs={"extra_body": _no_think_params},
        )

        print(f"🔗 Build cached chains (rewrite + legal)...")
        ai_state.rewrite_chain = (
            ChatPromptTemplate.from_template(QUERY_REWRITE_PROMPT)
            | ai_state.llm_instance
            | StrOutputParser()
        )
        ai_state.legal_chain = (
            ChatPromptTemplate.from_template(LEGAL_QUERY_PROMPT)
            | ai_state.llm_instance
            | StrOutputParser()
        )
        print(f"   ✅ Cached chains đã sẵn sàng.")

        llm_for_chain = ChatOpenAI(
            base_url=f"http://127.0.0.1:{LLAMA_SERVER_PORT}/v1",
            api_key="sk-no-key-required",
            model="qwen",
            temperature=0.2,
            top_p=0.8,
            max_tokens=1024,
            streaming=True,
            timeout=120,
            model_kwargs={"extra_body": _no_think_params},
        )

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}")
        ])
        ai_state.llm_chain = prompt_template | llm_for_chain | StrOutputParser()

        ai_state.is_ready = True
        print("✅ HỆ THỐNG AI ĐÃ SẴN SÀNG TOÀN BỘ!")
    except Exception as e:
        ai_state.init_error = str(e)
        print(f"❌ AI INIT FAIL: {e}")


# ==========================================
# QUẢN LÝ PHIÊN CHAT (SQLite)
# ==========================================
class ChatSessionManager:
    def __init__(self):
        self.current_session_id = None
        self._local = threading.local()   
        self._lock  = threading.Lock()    
        self.init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            conn = sqlite3.connect(DB_FILE, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return self._local.conn

    def init_db(self):
        conn = self._get_conn()
        conn.execute('''CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES sessions(id)
        )''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)')
        conn.commit()

    def create_session(self):
        new_id = str(uuid.uuid4())
        conn = self._get_conn()
        with self._lock:
            conn.execute(
                "INSERT INTO sessions (id, title) VALUES (?, ?)",
                (new_id, "Cuộc trò chuyện mới")
            )
            conn.commit()
        self.current_session_id = new_id
        return new_id

    def get_all_sessions(self):
        conn = self._get_conn()
        cursor = conn.execute("SELECT id, title FROM sessions ORDER BY created_at DESC")
        return cursor.fetchall()

    def get_messages(self, session_id=None):
        sid = session_id or self.current_session_id
        if not sid: return []
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
            (sid,)
        )
        return [{"role": row[0], "text": row[1]} for row in cursor.fetchall()]

    def add_message(self, role, text, session_id=None):
        # Ưu tiên session_id được truyền vào tường minh (tránh race condition khi
        # người dùng chuyển session trong lúc AI đang stream câu trả lời).
        sid = session_id or self.current_session_id
        if not sid:
            sid = self.create_session()
        conn = self._get_conn()
        with self._lock:
            conn.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (sid, role, text)
            )
            if role == "user":
                cursor = conn.execute(
                    "SELECT count(*) FROM messages WHERE session_id = ? AND role = 'user'",
                    (sid,)
                )
                if cursor.fetchone()[0] == 1:
                    title = text[:40] + "..." if len(text) > 40 else text
                    conn.execute(
                        "UPDATE sessions SET title = ? WHERE id = ?",
                        (title, sid)
                    )
            conn.commit()

    def switch_session(self, session_id):
        with self._lock:
            self.current_session_id = session_id

    def rename_session(self, session_id, new_title):
        conn = self._get_conn()
        with self._lock:
            conn.execute(
                "UPDATE sessions SET title = ? WHERE id = ?",
                (new_title, session_id)
            )
            conn.commit()

    def delete_session(self, session_id):
        conn = self._get_conn()
        with self._lock:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
        if self.current_session_id == session_id:
            self.current_session_id = None

session_manager = ChatSessionManager()