"""Answer length, latency and amortized time of the language-model planners (Table 4 of the article).

    python3 -m cuas_rlvr.cost_table --results results/v2 --latency results/latency --out out/tables/v2

Tokens: range over the per-seed split means of the answer length (named inventory, four test splits).
Latency: mean time of a single request (batch 1) on the first 20 in-distribution instances (results/latency).
Amortized: mean elapsed batch time per instance; range over the per-seed means of the three standard splits,
with the scale split in parentheses.
"""
import argparse
import json
import os

import numpy as np

from .report import SPLITS, _llm

ROWS = [("27B, zero-shot", "q38_27b", ["basenf4"], "lat1_q38_27b_base_nf4.jsonl"),
        ("27B, SFT + GRPO", "q38_27b", ["grpo_s0", "grpo_s1", "grpo_s2"], "lat1_q38_27b_grpo_s0.jsonl"),
        ("8B, SFT + GRPO", "q3_8b", ["grpo_s0", "grpo_s1", "grpo_s2"], "lat1_q3_8b_grpo_s0.jsonl"),
        ("7B, SFT + GRPO", "v2_q7b", ["grpo_s0", "grpo_s1", "grpo_s2"], "lat1_v2_q7b_grpo_s0.jsonl")]
STANDARD = ["test_id", "test_terrain", "test_sensor"]


def span(values, step=1):
    """'a--b' with both ends rounded to a multiple of step, or 'a' when they coincide."""
    lo, hi = (int(round(v / step)) * step for v in (min(values), max(values)))
    return f"{lo}" if lo == hi else f"{lo}--{hi}"


def split_means(results, tag, stages, key):
    """{split: [mean of key over the instances, one value per stage]}."""
    return {sp: [np.mean([r[key] for r in _llm(results, tag, st, sp).values()]) for st in stages] for sp in SPLITS}


def cost_table(results, latency):
    lines = []
    for label, tag, stages, lat in ROWS:
        tok = split_means(results, tag, stages, "n_tokens")
        tim = split_means(results, tag, stages, "time_s")
        lat_s = np.mean([json.loads(l)["time_s"] for l in open(os.path.join(latency, lat))])
        tokens = span([v for sp in SPLITS for v in tok[sp]], step=10)
        amort = f"{span([v for sp in STANDARD for v in tim[sp]])} ({span(tim['test_scale'])})"
        lines.append(f"{label} & {tokens} & {lat_s:.0f} & {amort} \\\\")
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/v2"); p.add_argument("--latency", default="results/latency")
    p.add_argument("--out", default="out/tables/v2")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tex = os.path.join(a.out, "cost.tex")
    with open(tex, "w") as f:
        f.write("\n".join(cost_table(a.results, a.latency)) + "\n")
    print(open(tex).read())


if __name__ == "__main__":
    main()
