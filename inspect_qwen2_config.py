from transformers import AutoConfig
from pathlib import Path

MODEL_DIR = Path(r"D:\Projects\NLP\models\Qwen1.5-1.8B-Chat")

def main():
    print("检查本地 Qwen2 config:", MODEL_DIR)
    cfg = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=False)
    print("Config 类型:", type(cfg))
    print("model_type:", getattr(cfg, "model_type", None))
    print("architectures:", getattr(cfg, "architectures", None))

if __name__ == "__main__":
    main()
