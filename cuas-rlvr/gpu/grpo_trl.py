"""GRPO with LoRA on a CUDA server (PyTorch + TRL + PEFT), with the CUAS-Gym verifier as the reward.

Same recipe as cuas_rlvr.grpo_mlx (G answers per instance at temperature 1, group-normalized advantages, one
on-policy update per group, no KL term, token-level loss) expressed with trl.GRPOTrainer.

    cd python && pip install -r gpu/requirements.txt
    python3 -m gpu.grpo_trl --model Qwen/Qwen3.8-27B --adapter ckpt/q38_sft --data data/gym2 --out ckpt/q38_grpo_s0 \
        --reward outcome --steps 300 --group 8 --no-think
    # faster sampling with a vLLM worker on another GPU:
    #   CUDA_VISIBLE_DEVICES=1 trl vllm-serve --model Qwen/Qwen3.8-27B --max_model_len 8192
    #   CUDA_VISIBLE_DEVICES=0 python3 -m gpu.grpo_trl ... --vllm server
Set CUAS_ANON=1 in the environment for the anonymized inventory (prompt and verifier read it).
"""
import argparse
import json
import os
import random

import torch
from datasets import Dataset

from cuas_gym.cli import ids, load
from cuas_gym.prompt import SYSTEM, build_prompt, parse_answer
from cuas_gym.rewards import hybrid_reward

_CACHE = {}


def _load(data, iid):
    if iid not in _CACHE:
        _CACHE[iid] = load(data, iid)
    return _CACHE[iid]


def completion_text(c):
    if isinstance(c, list):  # conversational: list of messages
        return "".join(m.get("content") or "" for m in c)
    return c or ""


def make_reward(data, mode, lam, log_path):
    def reward(prompts, completions, iid, **kw):
        out, recs = [], []
        for i, c in zip(iid, completions):
            inst, v = _load(data, i)
            text = completion_text(c)
            r = hybrid_reward(v, parse_answer(text), lam)
            val = 0.0 if not r["valid"] else {"hybrid": r["r"], "outcome": r["r_out"], "process": r["r_proc"]}[mode]
            out.append(float(val)); recs.append({"iid": i, "valid": r["valid"], "qr": r["r_out"], "r": float(val), "n_chars": len(text)})
        if log_path:
            with open(log_path, "a") as f:
                f.write(json.dumps({"r_mean": sum(out) / len(out), "r_max": max(out), "valid": sum(x["valid"] for x in recs) / len(recs),
                                    "qr": sum(x["qr"] for x in recs) / len(recs), "iid": iid[0]}) + "\n")
        return out
    reward.__name__ = f"cuas_{mode}"
    return reward


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--adapter", help="start from this LoRA adapter (SFT)")
    p.add_argument("--data", default="data/gym2"); p.add_argument("--out", required=True)
    p.add_argument("--reward", choices=["hybrid", "outcome", "process"], default="outcome"); p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=300); p.add_argument("--group", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5); p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--lora-rank", type=int, default=16); p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--micro", type=int, default=2, help="sequences per forward/backward pass")
    p.add_argument("--no-think", action="store_true", help="Qwen3 family: chat_template_kwargs.enable_thinking=false")
    p.add_argument("--penalize-truncated", action="store_true", help="keep completions cut at --max-tokens in the loss (reward 0, invalid) instead of masking them")
    p.add_argument("--vllm", choices=["off", "colocate", "server"], default="off")
    p.add_argument("--vllm-url", default=None, help="server mode: http://host:8000")
    p.add_argument("--vllm-mem", type=float, default=0.3, help="colocate mode: fraction of GPU memory for vLLM")
    p.add_argument("--fp32", action="store_true", help="CPU/debug: float32 instead of bf16"); p.add_argument("--load-4bit", action="store_true", help="QLoRA")
    p.add_argument("--save-every", type=int, default=25); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-split", default="train")
    a = p.parse_args()
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    random.seed(a.seed); torch.manual_seed(a.seed)
    train_ids = ids(a.data, a.train_split)
    rng = random.Random(a.seed)
    order = [train_ids[rng.randrange(len(train_ids))] for _ in range(a.steps)]  # one instance per step, as in grpo_mlx
    rows = []
    for iid in order:
        inst, _ = _load(a.data, iid)
        rows.append({"prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(inst)}], "iid": iid})
    ds = Dataset.from_list(rows)

    tok = AutoTokenizer.from_pretrained(a.model)
    kw = {"dtype": torch.float32 if a.fp32 else torch.bfloat16}
    if a.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
    peft_cfg = None
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter, is_trainable=True)
    else:
        peft_cfg = LoraConfig(r=a.lora_rank, lora_alpha=a.lora_alpha, lora_dropout=0.0, task_type="CAUSAL_LM",
                              target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    os.makedirs(a.out, exist_ok=True)
    log_path = os.path.join(a.out, "log.jsonl")
    cfg = GRPOConfig(
        output_dir=a.out, max_steps=a.steps, learning_rate=a.lr, lr_scheduler_type="constant", warmup_steps=0,
        num_generations=a.group, per_device_train_batch_size=a.micro, gradient_accumulation_steps=max(1, a.group // a.micro),
        generation_batch_size=a.group, max_completion_length=a.max_tokens,
        temperature=a.temp, beta=0.0, num_iterations=1, loss_type="dapo", scale_rewards="group",
        mask_truncated_completions=not a.penalize_truncated, bf16=not a.fp32, use_cpu=a.fp32 and not torch.cuda.is_available(), gradient_checkpointing=True,
        logging_steps=1, save_steps=a.save_every, save_total_limit=3, report_to=[], seed=a.seed,
        chat_template_kwargs={"enable_thinking": False} if a.no_think else None,
        use_vllm=a.vllm != "off", vllm_mode="server" if a.vllm == "server" else "colocate",
        vllm_server_base_url=a.vllm_url, vllm_gpu_memory_utilization=a.vllm_mem,
        remove_unused_columns=False,
    )
    trainer = GRPOTrainer(model=model, reward_funcs=make_reward(a.data, a.reward, a.lam, log_path), args=cfg,
                          train_dataset=ds, processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(os.path.join(a.out, "final"))
    with open(os.path.join(a.out, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=1)


if __name__ == "__main__":
    main()
