"""Tables and statistics of the extra 27B evaluations and of the latency runs (gpu/extra_evals.sh and the Apple-silicon run).

    python3 -m cuas_rlvr.report_extra --results results/v2 --latency results/latency --data data/gym2 --out out/tables/v2

Writes q38.tex (Table S11: 27B zero-shot on the NVFP4/vLLM backend against the geometric heuristic, the supervised 7B
checkpoint and the seed-averaged 7B GRPO model). Prints: 27B zero-shot on the NF4 training backend against the
NVFP4/vLLM run (paired bootstrap), 27B SFT against SFT+GRPO and against zero-shot, single-request latencies, and the
use of the unseen acoustic array (valid answers only) on the decisive instances of the unseen-sensor split.
"""
import argparse
import glob
import json
import os

import numpy as np

from cuas_gym.cli import load
from cuas_gym.prompt import anon_map
from .report import NAMES, SPLITS, _geo, _llm, ci, paired
from .report_v2 import seed_matrix

DECISIVE = ["test_sensor-000000", "test_sensor-000041", "test_sensor-000045", "test_sensor-000051", "test_sensor-000058"]


def fmt(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def num(x, sign=False):
    s = f"{x:+.3f}" if sign else f"{x:.3f}"
    return f"${s}$" if sign or x < 0 else s


def diff_tex(t):
    return f"{num(t[0], True)} [{num(t[1])}, {num(t[2])}]"


def q38_table(results):
    """Rows of Table S11: validity, mean QR with CI, mean answer length and three paired differences per split."""
    lines = []
    for sp in SPLITS:
        b, geo, sft = _llm(results, "q38_27b", "base", sp), _geo(results, sp), _llm(results, "v2_q7b", "sft", sp)
        gids, Q, _ = seed_matrix(results, "v2_q7b", sp)
        grpo = dict(zip(gids, Q.mean(1)))
        ids = sorted(b)
        q = np.array([b[i]["qr"] for i in ids]); lo, hi = ci(q)
        lines.append(f"{NAMES[sp]} & {100 * np.mean([b[i]['valid'] for i in ids]):.0f} & {q.mean():.3f} [{lo:.3f}, {hi:.3f}] & "
                     f"{np.mean([b[i]['n_tokens'] for i in ids]):.0f} & {diff_tex(paired(q, [geo[i] for i in ids]))} & "
                     f"{diff_tex(paired(q, [sft[i]['qr'] for i in ids]))} & {diff_tex(paired(q, [grpo[i] for i in ids]))} \\\\")
    return lines


def array_use(data, d, anon):
    """(valid answers containing an acoustic array, decisive instances with such an answer)."""
    def has(iid):
        r = d[iid]
        if not r["valid"]:
            return False
        lab = anon_map(load(data, iid)[0])["ac"] if anon else "ac"
        return any(t == lab for _, t in r["action"])
    return sum(has(i) for i in d), sum(has(i) for i in DECISIVE if i in d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/v2"); p.add_argument("--latency", default="results/latency")
    p.add_argument("--data", default="data/gym2"); p.add_argument("--out", default="out/tables/v2")
    a = p.parse_args()
    R = a.results
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "q38.tex"), "w") as f:
        f.write("\n".join(q38_table(R)) + "\n")
    print("== Qwen3.8-27B zero-shot: NF4 (training backend) vs NVFP4 (vLLM)")
    for sp in SPLITS:
        b, v = _llm(R, "q38_27b", "basenf4", sp), _llm(R, "q38_27b", "base", sp)
        ids = sorted(b)
        q = np.array([b[i]["qr"] for i in ids]); lo, hi = ci(q)
        print(f"{sp:13s} NF4 {q.mean():.3f} [{lo:.3f}, {hi:.3f}] valid {np.mean([b[i]['valid'] for i in ids]):.2f} "
              f"tokens {np.mean([b[i]['n_tokens'] for i in ids]):.0f} | NF4-NVFP4 {fmt(paired(q, [v[i]['qr'] for i in ids]))}")
    for tag in ("q38_27b", "q38_27banon"):
        print(f"== {tag}: SFT and SFT+GRPO (seed-averaged); base = zero-shot NF4 with the named inventory (for the property-based tag, SFT-base and GRPO-base mix representation and training; GRPO-SFT uses the same representation)")
        for sp in SPLITS:
            ids, Q, _ = seed_matrix(R, tag, sp); m = Q.mean(1)
            s, b = _llm(R, tag, "sft", sp), _llm(R, "q38_27b", "basenf4", sp)
            sq = [s[i]["qr"] for i in ids]; bq = [b[i]["qr"] for i in ids]
            print(f"{sp:13s} SFT {np.mean(sq):.3f} valid {np.mean([s[i]['valid'] for i in ids]):.2f} | GRPO-SFT {fmt(paired(m, sq))} "
                  f"| SFT-base {fmt(paired(sq, bq))} | GRPO-base {fmt(paired(m, bq))}")
    print("== acoustic array on the unseen-sensor split (valid answers only)")
    rows = [("Qwen2.5-7B names", "v2_q7b", 0), ("Qwen2.5-7B properties", "v2anon_q7b", 1), ("Qwen3-8B names", "q3_8b", 0),
            ("Qwen3-8B properties", "q3_8banon", 1), ("Qwen3.8-27B names", "q38_27b", 0), ("Qwen3.8-27B properties", "q38_27banon", 1)]
    for lab, tag, anon in rows:
        per = [array_use(a.data, _llm(R, tag, f"grpo_s{s}", "test_sensor"), anon) for s in range(3)]
        qd = np.mean([[_llm(R, tag, f"grpo_s{s}", "test_sensor")[i]["qr"] for i in DECISIVE] for s in range(3)])
        print(f"{lab:24s} answers {min(x[0] for x in per)}-{max(x[0] for x in per)} decisive {sum(x[1] for x in per)}/15 QR {qd:.3f}")
    for lab, tag, st, anon in [("27B zero-shot NF4", "q38_27b", "basenf4", 0), ("27B SFT names", "q38_27b", "sft", 0),
                               ("27B SFT properties", "q38_27banon", "sft", 1)]:
        d = _llm(R, tag, st, "test_sensor"); u = array_use(a.data, d, anon)
        print(f"{lab:24s} answers {u[0]} decisive {u[1]}/5 QR {np.mean([d[i]['qr'] for i in DECISIVE]):.3f}")
    print("== single-request latency (batch 1)")
    for f in sorted(glob.glob(os.path.join(a.latency, "lat1_*.jsonl"))):
        r = [json.loads(l) for l in open(f)]
        print(f"{os.path.basename(f):32s} n {len(r)} latency {np.mean([x['time_s'] for x in r]):.1f} s (median "
              f"{np.median([x['time_s'] for x in r]):.1f}) tokens {np.mean([x['n_tokens'] for x in r]):.0f} "
              f"QR {np.mean([x['qr'] for x in r]):.3f} valid {np.mean([x['valid'] for x in r]):.2f}")


if __name__ == "__main__":
    main()
