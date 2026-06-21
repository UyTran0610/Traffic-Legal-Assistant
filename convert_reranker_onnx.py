# Chạy 1 lần trên máy dev để tạo thư mục ./models/gte-multilingual-reranker-base-ONNX
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SRC  = "./models/gte-multilingual-reranker-base"
DEST = "./models/gte-multilingual-reranker-base_ONNX"
ONNX_PATH = os.path.join(DEST, "model.onnx")

os.makedirs(DEST, exist_ok=True)

print("📦 Đang load model PyTorch...")
tokenizer = AutoTokenizer.from_pretrained(SRC)
model = AutoModelForSequenceClassification.from_pretrained(
    SRC,
    trust_remote_code=True,
    torch_dtype=torch.float32,
)
model.eval()

# Tạo dummy input để trace
print("🔧 Tạo dummy input...")
dummy_pair = [["vượt đèn đỏ bị phạt bao nhiêu?", "Điều 6 Nghị định 100/2019 quy định mức phạt."]]
enc = tokenizer(
    [p[0] for p in dummy_pair],
    [p[1] for p in dummy_pair],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)

input_ids      = enc["input_ids"]
attention_mask = enc["attention_mask"]
token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids))

print("🚀 Đang export sang ONNX (có thể mất 1-2 phút)...")
with torch.no_grad():
    torch.onnx.export(
        model,
        (input_ids, attention_mask, token_type_ids),
        ONNX_PATH,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids":      {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "token_type_ids": {0: "batch_size", 1: "sequence_length"},
            "logits":         {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

# Lưu tokenizer sang DEST
print("💾 Lưu tokenizer...")
tokenizer.save_pretrained(DEST)

print(f"✅ Xong! Model ONNX đã lưu tại: {DEST}")
print(f"   File: {ONNX_PATH} ({os.path.getsize(ONNX_PATH) / 1024 / 1024:.1f} MB)")