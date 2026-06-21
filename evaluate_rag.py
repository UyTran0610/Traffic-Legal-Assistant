import re
import pandas as pd
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from backend import retrieve_context, ai_state, initialize_ai_system


# ============================================================
# TIỆN ÍCH: Làm sạch chuỗi trước khi ghi vào Excel
# ============================================================
def clean_for_excel(value):
    """Xóa ký tự điều khiển bất hợp lệ trong Excel (openpyxl IllegalCharacterError).
    Giữ lại: tab (\\x09), newline (\\x0A), carriage return (\\x0D).
    """
    if not isinstance(value, str):
        return value
    return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', value)


# ============================================================
# HÀM GỌI HỆ THỐNG RAG
# ============================================================
def ask_your_rag(question):
    context, legal_hint, _ = retrieve_context(question, top_k=3, messages=[])
    if not context:
        return "Xin lỗi, không tìm thấy tài liệu.", ""
    inputs = {
        "context": context,
        "question": question,
        "legal_hint": legal_hint,
        "chat_history": "(Không có)"
    }
    answer = ai_state.llm_chain.invoke(inputs)
    return answer, context


# ============================================================
# CẤU HÌNH LLM CHẤM ĐIỂM (Groq)
# ============================================================
GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

judge_llm = ChatGroq(
    temperature=0.0,
    model_name="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY
)

EVALUATION_PROMPT = """Bạn là một chuyên gia đánh giá hệ thống AI Tư vấn Pháp luật.
Nhiệm vụ của bạn là so sánh Câu trả lời của Hệ thống (System Answer) với Câu trả lời Chuẩn (Ground Truth) dựa trên Ngữ cảnh (Context) được cung cấp.

Hãy chấm điểm từ 1 đến 5 cho 2 tiêu chí sau:
1. Correctness (Tính chính xác): Hệ thống trả lời có đúng mức phạt, đúng luật so với Câu trả lời chuẩn không? (1: Hoàn toàn sai/Không trả lời được, 5: Hoàn toàn chính xác và đầy đủ).
2. Faithfulness (Tính trung thực): Hệ thống có dựa 100% vào Ngữ cảnh để trả lời không, hay tự bịa đặt thêm? (1: Bịa đặt hoàn toàn, 5: Trung thực 100% với ngữ cảnh).

DỮ LIỆU ĐÁNH GIÁ:
- Câu hỏi: {question}
- Ngữ cảnh hệ thống truy xuất được: {context}
- Câu trả lời của hệ thống (RAG): {rag_answer}
- Câu trả lời CHUẨN (Ground Truth): {ground_truth}

YÊU CẦU ĐẦU RA (Trả về ĐÚNG định dạng JSON sau, không kèm giải thích bên ngoài):
{{
    "correctness_score": <int 1-5>,
    "faithfulness_score": <int 1-5>,
    "reason": "<Giải thích ngắn gọn lý do chấm điểm>"
}}
"""

eval_prompt = ChatPromptTemplate.from_template(EVALUATION_PROMPT)
eval_chain = eval_prompt | judge_llm | JsonOutputParser()


# ============================================================
# HÀM CHẠY ĐÁNH GIÁ CHÍNH
# ============================================================
def run_evaluation(csv_path, max_samples=50):
    print("🚀 Đang khởi động hệ thống RAG Local...")
    initialize_ai_system()
    time.sleep(5)

    df = pd.read_csv(csv_path)
    if len(df) > max_samples:
        df = df.head(max_samples)

    output_file = f"Danh_gia_He_Thong_RAG_{len(df)}_cau.xlsx"
    results = []

    for index, row in df.iterrows():
        question    = str(row['question'])
        ground_truth = str(row['answer'])
        print(f"\n[{len(results)+1}/{len(df)}] Đang xử lý: {question}")

        # --- Gọi RAG (có bắt lỗi) ---
        try:
            rag_answer, context = ask_your_rag(question)
        except Exception as e:
            print(f"   ❌ Lỗi hệ thống RAG: {e}")
            rag_answer = "Lỗi hệ thống RAG"
            context    = ""

        # --- Gọi LLM chấm điểm (có bắt lỗi) ---
        try:
            eval_result = eval_chain.invoke({
                "question":     question,
                "context":      context,
                "rag_answer":   rag_answer,
                "ground_truth": ground_truth
            })
        except Exception as e:
            print(f"   ❌ Lỗi API chấm điểm: {e}")
            eval_result = {
                "correctness_score":  0,
                "faithfulness_score": 0,
                "reason": f"Lỗi API / Timeout: {e}"
            }

        correctness  = eval_result.get("correctness_score",  0)
        faithfulness = eval_result.get("faithfulness_score", 0)
        reason       = eval_result.get("reason", "")

        print(f"   => Điểm Chính xác : {correctness}/5")
        print(f"   => Điểm Trung thực: {faithfulness}/5")

        # --- Làm sạch toàn bộ chuỗi trước khi lưu ---
        results.append({
            "Câu hỏi":                  clean_for_excel(question),
            "Câu trả lời chuẩn (CSV)":  clean_for_excel(ground_truth),
            "Ngữ cảnh truy xuất (RAG)": clean_for_excel(context),
            "Câu trả lời của Bot (RAG)":clean_for_excel(rag_answer),
            "Điểm Chính xác (1-5)":     correctness,
            "Điểm Trung thực (1-5)":    faithfulness,
            "Nhận xét của Giám khảo":   clean_for_excel(reason),
        })

        # --- Lưu sau MỖI câu — không mất dữ liệu dù crash ---
        try:
            pd.DataFrame(results).to_excel(output_file, index=False)
            print(f"   💾 Đã lưu tạm ({len(results)}/{len(df)} câu) → {output_file}")
        except Exception as e:
            print(f"   ⚠️  Không thể lưu file tạm: {e}")

        # API Groq miễn phí: 30 request/phút → ngủ 2.5s là an toàn
        time.sleep(2.5)

    # --- TỔNG KẾT ---
    results_df    = pd.DataFrame(results)
    valid_results = results_df[results_df["Điểm Chính xác (1-5)"] > 0]

    avg_correctness  = valid_results["Điểm Chính xác (1-5)"].mean()  if len(valid_results) else 0
    avg_faithfulness = valid_results["Điểm Trung thực (1-5)"].mean() if len(valid_results) else 0

    print("\n" + "=" * 50)
    print("📊 TỔNG KẾT ĐÁNH GIÁ HỆ THỐNG")
    print(f"   Tổng số câu đã test       : {len(valid_results)}/{len(df)}")
    print(f"   Điểm Chính xác trung bình : {avg_correctness:.2f}/5.0")
    print(f"   Điểm Trung thực trung bình: {avg_faithfulness:.2f}/5.0")
    print("=" * 50)
    print(f"✅ Báo cáo đã lưu tại: {output_file}")


if __name__ == "__main__":
    run_evaluation("./data/Dataset_6000.csv", max_samples=50)