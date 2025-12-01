import os, json, argparse
from pathlib import Path
try:
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model
except ImportError as e:
    raise SystemExit(
        f"Missing training deps: {e}. Please install with:\n"
        "  pip install transformers peft datasets accelerate\n"
        "(Optional for QLoRA: pip install bitsandbytes)"
    )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True, help='JSONL distillation pairs path')
    ap.add_argument('--model', type=str, default='qwen/Qwen1.5-0.5B', help='Base HF model repo id or local path')
    ap.add_argument('--output', type=str, default='runs/lora-out', help='Output directory')
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--qlora', action='store_true')
    ap.add_argument('--max-length', type=int, default=1024)
    ap.add_argument('--gradient-accum', type=int, default=None)
    ap.add_argument('--warmup-ratio', type=float, default=0.03)
    ap.add_argument('--seed', type=int, default=42)
    # LoRA-specific overrides
    ap.add_argument('--lora-r', type=int, default=8)
    ap.add_argument('--lora-alpha', type=int, default=16)
    ap.add_argument('--lora-dropout', type=float, default=0.05)
    ap.add_argument(
        '--lora-target-modules',
        type=str,
        default=None,
        help='Comma-separated list of module name substrings for LoRA injection; if not set, use a Qwen-friendly default such as q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
    )
    return ap.parse_args()


def load_pairs(path: Path):
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for ln in f:
            if not ln.strip():
                continue
            obj = json.loads(ln)
            instr = obj.get('instruction', '')
            outp = obj.get('output', '')
            text = f"指令:\n{instr}\n\n优质答案:\n{outp}".strip()
            rows.append({'text': text})
    if not rows:
        raise SystemExit(f'No usable rows found in {path}')
    return rows


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f'Distill data not found: {data_path}')

    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

    import random, torch
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_pairs(data_path)
    ds = Dataset.from_list(rows)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tok(batch):
        return tokenizer(batch['text'], truncation=True, max_length=args.max_length)

    tokenized = ds.map(tok, batched=True, remove_columns=['text'])

    quant_kwargs = {}
    if args.qlora:
        try:
            import bitsandbytes as bnb  # noqa: F401
            quant_kwargs = dict(load_in_4bit=True, device_map='auto')
            print('✅ QLoRA 4-bit quantization enabled')
        except ImportError:
            print('⚠️ bitsandbytes not installed; continuing without 4-bit quantization')
    else:
        print('ℹ️ Standard LoRA (no 4-bit)')

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        **quant_kwargs,
    )

    # Prepare target_modules for Qwen/Qwen2-style architectures if user did not override
    if args.lora_target_modules:
        target_modules = [m.strip() for m in args.lora_target_modules.split(',') if m.strip()]
    else:
        # Default set that works well for Qwen/Qwen2 and many decoder-only LMs
        target_modules = [
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj',
        ]
        print(f'ℹ️ Using default LoRA target_modules for Qwen: {target_modules}')

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    grad_acc = args.gradient_accum or max(1, 8 // args.batch)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=grad_acc,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=args.fp16,
        logging_steps=10,
        save_strategy='epoch',
        report_to=[],
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    trainer.train()

    save_dir = str(out_dir)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    (out_dir / 'RUN_INFO.txt').write_text(
        f"Model: {args.model}\n"
        f"LoRA r={lora_cfg.r}, alpha={lora_cfg.lora_alpha}, dropout={lora_cfg.lora_dropout}\n"
        f"Target modules={target_modules}\n"
        f"QLoRA={args.qlora}\nRows={len(rows)}\nEpochs={args.epochs}\nBatch={args.batch}\n"
        f"GradAccum={grad_acc}\nMaxLen={args.max_length}\nSeed={args.seed}\n",
        encoding='utf-8',
    )
    print(f'✅ LoRA training complete -> {out_dir}')


if __name__ == '__main__':
    main()
