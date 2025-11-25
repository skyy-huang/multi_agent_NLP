import os
from typing import Dict, Union, Optional

# 尝试导入 torch 与 transformers，失败则标记不可用并使用占位实现
try:
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    AutoTokenizer = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    _TRANSFORMERS_AVAILABLE = False

try:
    from peft import PeftModel  # type: ignore
    _PEFT_AVAILABLE = True
except Exception:
    PeftModel = None  # type: ignore
    _PEFT_AVAILABLE = False

_FORCE_STUB = os.getenv("FORCE_STUDENT_STUB") == "1"
_NEED_STUB = _FORCE_STUB or (not _TORCH_AVAILABLE) or (not _TRANSFORMERS_AVAILABLE)

if _NEED_STUB:
    # ========== 轻量 Stub 模式 ==========
    # 在 CI / 无网络 / 单元测试场景下，或缺失核心依赖时，使用占位模型避免导入错误
    class HFChatLLM:  # type: ignore
        def __init__(self, base_model: str, lora_dir: Optional[str] = None, max_new_tokens: int = 512, **_):
            self.base_model = base_model
            self.lora_dir = lora_dir or ""
            self.max_new_tokens = max_new_tokens
            self.model_name = "student-stub"
        def _format_prompt(self, obj: Union[Dict, str]) -> str:
            if isinstance(obj, dict):
                return "\n".join(f"{k}: {v}" for k, v in obj.items())
            return str(obj)
        def invoke(self, prompt: Union[Dict, str]) -> str:
            text = self._format_prompt(prompt)
            # 简化输出，仅截断 + 标记
            head = text[: min(200, len(text))]
            return f"[STUB_GENERATION]\n{text[:120]}..."
        def __call__(self, prompt: Union[Dict, str]):  # 使其可调用
            return self.invoke(prompt)
        def __or__(self, other):  # 简单链式兼容：直接返回下游对象
            return other
else:
    class HFChatLLM:
        """HF + 可选 LoRA 聊天式学生模型封装 (.invoke)
        支持本地 Qwen/Qwen1.5-1.8B-Chat 及 LoRA 适配器加载。
        依赖: torch + transformers (+ peft 可选)
        """

        def __init__(
            self,
            base_model: str,
            lora_dir: Optional[str] = None,
            max_new_tokens: int = 512,
            device: Optional[str] = None,
            torch_dtype: Optional[str] = None,
            device_map: Optional[str] = None,
        ) -> None:
            self.base_model = base_model
            self.lora_dir = lora_dir or ""
            self.max_new_tokens = max_new_tokens
            self.model_name = base_model

            if device is None:
                device = "cuda" if (torch and getattr(torch, 'cuda', None) and torch.cuda.is_available()) else "cpu"
            self.device = torch.device(device) if torch else device

            # dtype 选择
            dtype = None
            if torch and torch_dtype:
                try:
                    dtype = getattr(torch, torch_dtype)
                except Exception:
                    dtype = None

            self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            load_kwargs = {"trust_remote_code": True}
            if dtype is not None:
                load_kwargs["torch_dtype"] = dtype
            if device_map:
                load_kwargs["device_map"] = device_map

            base = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)

            if self.lora_dir and os.path.exists(self.lora_dir) and _PEFT_AVAILABLE:
                try:
                    self.model = PeftModel.from_pretrained(base, self.lora_dir)
                    print(f"✅ Loaded LoRA adapters from {self.lora_dir}")
                except Exception as e:
                    print(f"⚠️ Failed to load LoRA adapters from {self.lora_dir}: {e}. Using base model.")
                    self.model = base
            else:
                if self.lora_dir and not os.path.exists(self.lora_dir):
                    print(f"⚠️ LoRA dir not found: {self.lora_dir}. Using base model.")
                if self.lora_dir and not _PEFT_AVAILABLE:
                    print("⚠️ peft not available; cannot load LoRA. Using base model.")
                self.model = base

            try:
                self.model.to(self.device)  # 若使用 device_map=auto 可能已分配，忽略异常
            except Exception:
                pass
            self.model.eval()

        def _format_prompt(self, obj: Union[Dict, str]) -> str:
            if isinstance(obj, dict):
                # flatten dict into readable text; avoid complex chat formatting here
                return "\n".join(f"{k}: {v}" for k, v in obj.items())
            return str(obj)

        def invoke(self, prompt: Union[Dict, str]) -> str:
            text = self._format_prompt(prompt)
            inputs = self.tokenizer(text, return_tensors="pt")
            if torch:
                try:
                    inputs = inputs.to(self.device)
                except Exception:
                    pass
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
                return self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            else:
                return "[Torch unavailable: stubbed generation]"

        def __call__(self, prompt: Union[Dict, str]):  # 供 LangChain Runnable 检测
            return self.invoke(prompt)
        def __or__(self, other):  # 保持与 PromptTemplate | LLM | Parser 链式兼容
            return other
