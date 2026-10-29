"""Supervised fine-tuning with LoRA on a CUDA server (TRL SFTTrainer), loss on the assistant turn only.

    python3 -m gpu.sft_trl --model Qwen/Qwen3.8-27B --data data/sft_train_v2.jsonl --out ckpt/q38_sft --epochs 2 --no-think
Input: the jsonl written by cuas_gym.sft_data ({"iid", "messages": [system, user, assistant]}).
Checkpoints are saved every --save-every steps; gpu/run_gpu.sh evaluates each one on the 20 held-out instances and links the best (generated-answer QR) as ckpt/<tag>_sft/best.
"""
import argparse
import json
import os
import random

import torch
from datasets import Dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=float, default=2.0); p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-len", type=int, default=4096); p.add_argument("--batch", type=int, default=2); p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lora-rank", type=int, default=16); p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--no-think", action="store_true", help="Qwen3 family: render the template with enable_thinking=false")
    p.add_argument("--load-4bit", action="store_true"); p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--valid", type=int, default=20, help="held-out rows for checkpoint selection (same shuffle as mlx_sft_data)"); p.add_argument("--seed", type=int, default=0); p.add_argument("--fp32", action="store_true", help="CPU/debug")
    a = p.parse_args()
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(a.model)
    src = [json.loads(l) for l in open(a.data)]
    random.Random(0).shuffle(src)                     # same split as mlx_sft_data.py / select_sft.py
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "valid_ids.json"), "w") as f:
        json.dump([r["iid"] for r in src[:a.valid]], f)
    rows = []
    for r in src[a.valid:]:
        m = r["messages"]
        # prompt-completion format: the loss covers the completion only (completion_only_loss defaults to True for it)
        rows.append({"prompt": m[:-1], "completion": [m[-1]]})
    ds = Dataset.from_list(rows)
    kw = {"dtype": torch.float32 if a.fp32 else torch.bfloat16}
    if a.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
    peft_cfg = LoraConfig(r=a.lora_rank, lora_alpha=a.lora_alpha, lora_dropout=0.0, task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    cfg = SFTConfig(output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=10,
                    per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.accum, max_length=a.max_len,
                    bf16=not a.fp32, use_cpu=a.fp32 and not torch.cuda.is_available(), gradient_checkpointing=True, logging_steps=5, save_steps=a.save_every, save_total_limit=20,
                    report_to=[], seed=a.seed, max_grad_norm=1.0)
    if a.no_think:
        # SFTTrainer has no chat_template_kwargs; render with the tokenizer's template and pass text instead
        end = "<|im_end|>\n" if "<|im_end|>" in tok.get_vocab() else tok.eos_token

        def render(ex):
            pr = tok.apply_chat_template(ex["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            return {"prompt": pr, "completion": ex["completion"][0]["content"] + end}
        ds = ds.map(render, remove_columns=ds.column_names)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(os.path.join(a.out, "final"))


if __name__ == "__main__":
    main()
