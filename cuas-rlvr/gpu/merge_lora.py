"""Merges a LoRA adapter into the base weights so vLLM can serve the trained model (or use vllm --enable-lora instead).

    python3 -m gpu.merge_lora --model Qwen/Qwen3.8-27B --adapter ckpt/q38_grpo_s0/final --out models/q38_grpo_s0
"""
import argparse

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--adapter", required=True); p.add_argument("--out", required=True)
    a = p.parse_args()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
    m = PeftModel.from_pretrained(m, a.adapter).merge_and_unload()
    m.save_pretrained(a.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(a.model).save_pretrained(a.out)


if __name__ == "__main__":
    main()
