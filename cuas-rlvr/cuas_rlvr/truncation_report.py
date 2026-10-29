"""Truncation during GPU GRPO: share of sampled answers cut at the token limit (TRL completions/clipped_ratio), mean
answer length and sampled validity before and after step 120, per run.

    python3 -m cuas_rlvr.truncation_report --ckpt ckpt

Reads ckpt/<run>/trainer_state.json (TRL log history) and ckpt/<run>/log.jsonl (validity of the sampled answers).
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    p = argparse.ArgumentParser(); p.add_argument("--ckpt", default="ckpt"); p.add_argument("--split-step", type=int, default=120)
    a = p.parse_args()
    k = a.split_step
    print(f"{'run':26s} clip<{k} clip>={k} valid<{k} valid>={k} len<{k} len>={k}")
    for f in sorted(glob.glob(os.path.join(a.ckpt, "*", "trainer_state.json"))):
        run = os.path.basename(os.path.dirname(f))
        H = [h for h in json.load(open(f))["log_history"] if "completions/clipped_ratio" in h]
        c = np.array([h["completions/clipped_ratio"] for h in H]); L = np.array([h["completions/mean_length"] for h in H])
        lg = os.path.join(os.path.dirname(f), "log.jsonl")
        v = np.array([json.loads(x)["valid"] for x in open(lg) if x.strip()]) if os.path.exists(lg) else None
        vs = f"{v[:k].mean():9.3f} {v[k:].mean():10.3f}" if v is not None else f"{'-':>9s} {'-':>10s}"
        print(f"{run:26s} {c[:k].mean():8.3f} {c[k:].mean():9.3f} {vs} {L[:k].mean():7.0f} {L[k:].mean():8.0f}")


if __name__ == "__main__":
    main()
