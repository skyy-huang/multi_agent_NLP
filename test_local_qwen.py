from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_DIR = r"D:\Projects\NLP\models\Qwen1.5-1.8B-Chat"

def main():
    print("尝试从本地 Qwen2 1.8B 目录加载模型:", MODEL_DIR)

    # Qwen2 是 transformers 内置模型，不需要 trust_remote_code
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    print("Tokenizer 加载成功。")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("使用设备:", device)

    # 根据 config，默认 torch_dtype 是 bfloat16；我们按设备选择合适的 dtype
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (
        torch.float16 if device == "cuda" else torch.float32
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    ).eval()
    print("Model 加载成功。")

    prompt = "请用两三句话解释一下什么是大型语言模型。"
    inputs = tokenizer(prompt, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
        )

    print("\n=== 模型输出 ===")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
