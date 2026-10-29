"""Evaluation on the GPU with transformers generate (base model or LoRA adapter), same output format as eval_mlx/eval_api.

    python3 -m gpu.eval_hf --model Qwen/Qwen3-8B --split test_id --no-think --out results/gpu/q3_8b_base_test_id.jsonl
    python3 -m gpu.eval_hf --model Qwen/Qwen3-8B --adapter ckpt/q3_8b_grpo_s0/final --split test_sensor --no-think --out ...
Greedy by default; --samples k --temp 1.0 for best-of-k; resumes an interrupted run.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from cuas_gym.cli import ids, load
from cuas_gym.prompt import SYSTEM, build_prompt, parse_answer
from cuas_gym.rewards import hybrid_reward


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--adapter")
    p.add_argument("--data", default="data/gym2"); p.add_argument("--split", default="test_id")
    p.add_argument("--n", type=int, default=1000); p.add_argument("--samples", type=int, default=1)
    p.add_argument("--temp", type=float, default=0.0); p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--batch", type=int, default=8); p.add_argument("--no-think", action="store_true"); p.add_argument("--think", action="store_true")
    p.add_argument("--no-attrs", action="store_true"); p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--load-4bit", action="store_true"); p.add_argument("--ids", help="json list of instance ids to evaluate instead of the split")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"dtype": torch.bfloat16, "device_map": "auto"}
    if a.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    tk = {} if not (a.no_think or a.think) else {"enable_thinking": bool(a.think)}
    name = a.model + (f"+{a.adapter}" if a.adapter else "") + ("" if not tk else ("+think" if a.think else "+nothink"))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            r = json.loads(l); done.add((r["iid"], r["k"]))
    pool = json.load(open(a.ids)) if a.ids else ids(a.data, a.split)
    jobs = [(iid, k) for iid in pool[:a.n] for k in range(a.samples) if (iid, k) not in done]
    t0 = time.time()
    with open(a.out, "a") as f:
        for b in range(0, len(jobs), a.batch):
            chunk = jobs[b:b + a.batch]
            insts = {iid: load(a.data, iid) for iid, _ in chunk}
            texts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                              {"role": "user", "content": build_prompt(insts[iid][0], attrs=not a.no_attrs)}],
                                             tokenize=False, add_generation_prompt=True, **tk) for iid, _ in chunk]
            enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
            t = time.time()
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=a.max_tokens, do_sample=a.temp > 0, temperature=a.temp if a.temp > 0 else None,
                                     top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
            dt = (time.time() - t) / len(chunk)
            for (iid, k), row in zip(chunk, gen[:, enc["input_ids"].shape[1]:]):
                toks = row[row != tok.pad_token_id] if tok.pad_token_id != tok.eos_token_id else row
                text = tok.decode(row, skip_special_tokens=True)
                inst, v = insts[iid]
                act = parse_answer(text)
                r = hybrid_reward(v, act, a.lam)
                sc = v.score(act) if r["valid"] else {"Fn": 0.0, "cov90": 0.0}
                rec = {"iid": iid, "k": k, "model": name, "valid": r["valid"], "error": r["error"], "qr": r["r_out"],
                       "r_proc": r["r_proc"], "Fn": sc["Fn"], "cov90": sc["cov90"], "action": act,
                       "n_tokens": int(len(toks)), "time_s": dt, "text": text}
                f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[{b + len(chunk)}/{len(jobs)}] {time.time() - t0:.0f} s, {dt:.1f} s per answer, "
                  f"mean QR so far {np.mean([json.loads(l)['qr'] for l in open(a.out)]):.3f}", flush=True)
    recs = [json.loads(l) for l in open(a.out)]
    print(f"{a.split}: n={len(recs)} valid={np.mean([r['valid'] for r in recs]):.3f} "
          f"QR mean={np.mean([r['qr'] for r in recs]):.3f} tokens={np.mean([r['n_tokens'] for r in recs]):.0f}")


if __name__ == "__main__":
    main()
