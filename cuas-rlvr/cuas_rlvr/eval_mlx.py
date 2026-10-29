"""Evaluation of a local model (optionally with a LoRA adapter) on CUAS-Gym with MLX on Apple silicon.

    python3 -m cuas_rlvr.eval_mlx --model mlx-community/Qwen2.5-7B-Instruct-4bit --split test_id \
        --out results/mlx_qwen7b_base_test_id.jsonl
    python3 -m cuas_rlvr.eval_mlx --model ... --adapter ckpt/grpo --split test_terrain --out ...
Greedy decoding by default; --samples k with --temp 1.0 for best-of-k. Resumes an interrupted run.
"""
import argparse
import json
import os
import time

import numpy as np

from cuas_gym.cli import ids, load
from .mlx_common import chat_ids, load_model
from cuas_gym.prompt import parse_answer
from cuas_gym.rewards import hybrid_reward


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--adapter")
    p.add_argument("--data", default="data/gym"); p.add_argument("--split", default="test_id")
    p.add_argument("--n", type=int, default=1000); p.add_argument("--samples", type=int, default=1)
    p.add_argument("--temp", type=float, default=0.0); p.add_argument("--max-tokens", type=int, default=1536)
    p.add_argument("--batch", type=int, default=8, help="prompts generated together")
    p.add_argument("--no-attrs", action="store_true"); p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    from mlx_lm.generate import batch_generate
    from mlx_lm.sample_utils import make_sampler
    model, tok, _ = load_model(a.model, a.adapter)
    name = a.model + (f"+{a.adapter}" if a.adapter else "")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            r = json.loads(l); done.add((r["iid"], r["k"]))
    jobs = [(iid, k) for iid in ids(a.data, a.split)[:a.n] for k in range(a.samples) if (iid, k) not in done]
    sampler = make_sampler(temp=a.temp)
    t0 = time.time()
    with open(a.out, "a") as f:
        for b in range(0, len(jobs), a.batch):
            chunk = jobs[b:b + a.batch]
            insts = {iid: load(a.data, iid) for iid, _ in chunk}
            prompts = [chat_ids(tok, insts[iid][0], attrs=not a.no_attrs) for iid, _ in chunk]
            t = time.time()
            res = batch_generate(model, tok, prompts, max_tokens=a.max_tokens, sampler=sampler, return_token_ids=True)
            dt = (time.time() - t) / len(chunk)
            for (iid, k), text, toks in zip(chunk, res.texts, res.token_ids):
                inst, v = insts[iid]
                act = parse_answer(text)
                r = hybrid_reward(v, act, a.lam)
                sc = v.score(act) if r["valid"] else {"Fn": 0.0, "cov90": 0.0}
                rec = {"iid": iid, "k": k, "model": name, "valid": r["valid"], "error": r["error"], "qr": r["r_out"],
                       "r_proc": r["r_proc"], "Fn": sc["Fn"], "cov90": sc["cov90"], "action": act,
                       "n_tokens": len(toks), "time_s": dt, "text": text}
                f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[{b + len(chunk)}/{len(jobs)}] {time.time() - t0:.0f} s, {dt:.1f} s per answer, "
                  f"mean QR so far {np.mean([json.loads(l)['qr'] for l in open(a.out)]):.3f}", flush=True)
    recs = [json.loads(l) for l in open(a.out)]
    print(f"{a.split}: n={len(recs)} valid={np.mean([r['valid'] for r in recs]):.3f} "
          f"QR mean={np.mean([r['qr'] for r in recs]):.3f} tokens={np.mean([r['n_tokens'] for r in recs]):.0f}")


if __name__ == "__main__":
    main()
