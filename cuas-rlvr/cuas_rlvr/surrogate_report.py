"""Tables of the learned-surrogate baseline and of the results by threat (supplementary material).

    python3 -m cuas_rlvr.surrogate_report --results results/v2 --data data/gym2 --out out/tables/v2
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

from .report import SPLITS, NAMES, _geo, _llm, ci, paired

THREATS = [("fiber", "Fiber-optic quadcopter"), ("fpv", "FPV quadcopter"), ("owa", "Fixed-wing aircraft")]
PLANNERS = [("v2_q7b", "7B / names"), ("v2anon_q7b", "7B / properties"), ("q3_8b", "8B / names"), ("q3_8banon", "8B / properties"),
            ("q38_27b", "27B / names"), ("q38_27banon", "27B / properties")]


def surrogate(results, mode, seeds=(0, 1, 2)):
    runs = {}
    for s in seeds:
        for r in csv.DictReader(open(os.path.join(results, "surrogate", f"surrogate_{mode}_s{s}.csv"))):
            runs.setdefault(s, {})[r["iid"]] = r
    return runs


def seedavg(results, tag):
    out = {}
    for sp in SPLITS:
        runs = [_llm(results, tag, f"grpo_s{s}", sp) for s in (0, 1, 2)]
        for i in runs[0]:
            out[i] = float(np.mean([r[i]["qr"] for r in runs]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/v2"); ap.add_argument("--data", default="data/gym2")
    ap.add_argument("--out", default="out/tables/v2")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    L = []
    for mode in ("properties", "names"):
        runs = surrogate(a.results, mode)
        for s, r in sorted(runs.items()):
            cells = []
            for sp in SPLITS:
                q = [float(x["qr"]) for i, x in r.items() if i.startswith(sp)]
                e = [float(x["mae"]) for i, x in r.items() if i.startswith(sp)]
                cells.append(f"{np.mean(q):.3f} ({np.mean(e):.3f})")
            L.append(f"{mode.capitalize() if s == 0 else ''} & {s} & " + " & ".join(cells) + " \\\\")
        avg = {i: np.mean([float(runs[s][i]["qr"]) for s in runs]) for i in runs[0]}
        cells = []
        for sp in SPLITS:
            q = np.array([avg[i] for i in sorted(avg) if i.startswith(sp)]); lo, hi = ci(q)
            cells.append(f"{q.mean():.3f} [{lo:.3f}, {hi:.3f}]")
        L.append(" & mean & " + " & ".join(cells) + " \\\\")
        L.append("\\midrule")
    open(os.path.join(a.out, "surrogate.tex"), "w").write("\n".join(L[:-1]) + "\n\\bottomrule\n")
    # by threat, pooled over the four splits
    drone = {}
    for f in glob.glob(os.path.join(a.data, "test_*.json")):
        if f.endswith(".ref.json"):
            continue
        d = json.load(open(f)); drone[d["iid"]] = d["drone"]
    gc = {}
    for sp in SPLITS:
        gc.update(_geo(a.results, sp, "geometric_complete"))
    runs = surrogate(a.results, "properties")
    sur = {i: float(np.mean([float(runs[s][i]["qr"]) for s in runs])) for i in runs[0]}
    T = []
    rows = [("Geometric / completed", gc, False)] + [(lab, seedavg(a.results, tag), True) for tag, lab in PLANNERS] + \
           [("Surrogate + greedy", sur, True)]
    for lab, d, diff in rows:
        cells = []
        for key, _ in THREATS:
            ids = sorted(i for i in drone if drone[i] == key)
            m = np.mean([d[i] for i in ids])
            if diff:
                dm, lo, hi = paired([d[i] for i in ids], [gc[i] for i in ids])
                cells.append(f"{m:.3f} & ${dm:+.3f}$ [${lo:+.3f}$, ${hi:+.3f}$]")
            else:
                cells.append(f"{m:.3f} & --")
        T.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    n = {k: sum(1 for i in drone if drone[i] == k) for k, _ in THREATS}
    open(os.path.join(a.out, "threat.tex"), "w").write("\n".join(T) + "\n\\bottomrule\n")
    print(open(os.path.join(a.out, "surrogate.tex")).read()); print(n); print(open(os.path.join(a.out, "threat.tex")).read())


if __name__ == "__main__":
    main()
