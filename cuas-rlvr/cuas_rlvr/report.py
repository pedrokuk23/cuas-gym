"""Tables for the manuscript from the generated data and the baseline results.

    python3 -m cuas_rlvr.report --data data/gym --results results --out out/tables
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

from cuas_gym.cli import ids, load

SPLITS = ["test_id", "test_terrain", "test_sensor", "test_scale"]
NAMES = {"test_id": "In-distribution", "test_terrain": "Unseen terrain", "test_sensor": "Unseen sensor", "test_scale": "Scale",
         "train": "Training"}
METHODS = [("random", "Random feasible", "no"), ("geometric", "Geometric heuristic", "no"),
           ("geometric_complete", "Geometric heuristic + budget completion", "no"), ("greedy", "Greedy (lazy, best of three)", "yes"),
           ("local_search", "Greedy + local search", "yes"), ("genetic", "Genetic algorithm", "yes")]


def ci(x, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = [np.mean(rng.choice(x, len(x))) for _ in range(B)]
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def dataset_table(data):
    lines = []
    for sp in ["train"] + SPLITS:
        I = ids(data, sp)
        if not I:
            continue
        nc, ncell, bud, ns, pv, fn, dr = [], [], [], [], [], [], {}
        for iid in I:
            inst, v = load(data, iid)
            nc.append(inst.P.shape[0]); ncell.append(inst.P.shape[1]); bud.append(inst.budget)
            ns.append(len(v.reference()["rows"])); pv.append(v.reference()["proven"]); fn.append(v.reference()["F"] / v.W)
            dr[inst.drone] = dr.get(inst.drone, 0) + 1
        lines.append(f"{NAMES[sp]} & {len(I)} & {np.mean(nc):.0f} & {np.mean(ncell):.0f} & {np.mean(bud):.1f} & "
                     f"{np.mean(ns):.1f} & {np.mean(fn):.2f} & {100 * np.mean(pv):.0f} \\\\")
    return lines


def baseline_table(results):
    lines = []
    for sp in SPLITS:
        f = os.path.join(results, f"baselines_{sp}.csv")
        if not os.path.exists(f):
            continue
        rows = list(csv.DictReader(open(f)))
        first = True
        methods = [m for m in METHODS if any(x["method"] == m[0] for x in rows)]
        for key, label, sim in methods:
            r = [x for x in rows if x["method"] == key]
            q = np.array([float(x["qr"]) for x in r])
            # the unit of the interval is the instance: repeated random draws are averaged per instance first
            per = {}
            for x in r:
                per.setdefault(x["iid"], []).append(float(x["qr"]))
            lo, hi = ci(np.array([np.mean(v) for v in per.values()]))
            ev = np.mean([float(x["evals"]) for x in r])
            sp_lbl = f"\\multirow{{{len(methods)}}}{{*}}{{{NAMES[sp]}}}" if first else ""
            first = False
            evs = f"{ev:,.0f}".replace(",", "{,}")
            lines.append(f"{sp_lbl} & {label} & {sim} & {q.mean():.3f} [{lo:.3f}, {hi:.3f}] & {100 * (q >= 0.999).mean():.0f} & {evs} \\\\")
        lines.append("\\midrule")
    return lines[:-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/gym"); p.add_argument("--results", default="results"); p.add_argument("--out", default="out/tables")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "dataset.tex"), "w") as f:
        f.write("\n".join(dataset_table(a.data)) + "\n\\bottomrule\n")
    with open(os.path.join(a.out, "baselines.tex"), "w") as f:
        f.write("\n".join(baseline_table(a.results)) + "\n\\bottomrule\n")
    print(open(os.path.join(a.out, "dataset.tex")).read()); print(open(os.path.join(a.out, "baselines.tex")).read())


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------------------------------------------------
# Language-model results (MLX runs)
LLM_STAGES = [("base", "Qwen2.5-7B, zero-shot"), ("sft", "+ SFT"), ("grpo", "+ SFT + GRPO, hybrid reward"),
              ("grpo_outcome", "+ SFT + GRPO, outcome reward")]


def _llm(results, tag, st, sp):
    f = os.path.join(results, f"mlx_{tag}_{st}_{sp}.jsonl")
    if not os.path.exists(f):
        return None
    return {json.loads(l)["iid"]: json.loads(l) for l in open(f)}


def _geo(results, sp, method="geometric"):
    f = os.path.join(results, f"baselines_{sp}.csv")
    return {r["iid"]: float(r["qr"]) for r in csv.DictReader(open(f)) if r["method"] == method}


def paired(a, b, B=5000, seed=0):
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    m = [rng.choice(d, len(d)).mean() for _ in range(B)]
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def llm_table(results, tag="q7b"):
    lines = []
    for sp in SPLITS:
        geo = _geo(results, sp)
        q = np.array([geo[i] for i in sorted(geo)])
        lo, hi = ci(q)
        rows = [("Geometric heuristic", q, None, None)]
        for st, label in LLM_STAGES:
            d = _llm(results, tag, st, sp)
            if d is None:
                continue
            ids = sorted(d)
            rows.append((label, np.array([d[i]["qr"] for i in ids]), np.mean([d[i]["valid"] for i in ids]),
                         np.mean([d[i]["cov90"] for i in ids])))
        first = True
        for label, qq, val, cov in rows:
            lo, hi = ci(qq)
            sp_lbl = f"\\multirow{{{len(rows)}}}{{*}}{{{NAMES[sp]}}}" if first else ""
            first = False
            vs = "--" if val is None else f"{100 * val:.0f}"
            lines.append(f"{sp_lbl} & {label} & {vs} & {qq.mean():.3f} [{lo:.3f}, {hi:.3f}] & {qq.min():.2f} \\\\")
        lines.append("\\midrule")
    return lines[:-1] + ["\\bottomrule"]


def llm_stats(results, tag="q7b"):
    """Paired differences used in the text."""
    out = {}
    for sp in SPLITS:
        geo = _geo(results, sp)
        g = _llm(results, tag, "grpo", sp); o = _llm(results, tag, "grpo_outcome", sp); s = _llm(results, tag, "sft", sp)
        ids = sorted(set(g) & set(o) & set(s) & set(geo))
        out[sp] = {"hyb_vs_out": paired([g[i]["qr"] for i in ids], [o[i]["qr"] for i in ids]),
                   "out_vs_geo": paired([o[i]["qr"] for i in ids], [geo[i] for i in ids]),
                   "hyb_vs_geo": paired([g[i]["qr"] for i in ids], [geo[i] for i in ids]),
                   "out_vs_sft": paired([o[i]["qr"] for i in ids], [s[i]["qr"] for i in ids]), "n": len(ids)}
    return out


def training_curves(ckpt, out_png, runs=(("q7b_grpo", "hybrid reward"), ("q7b_grpo_outcome", "outcome reward")), w=25):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.5, 2.4), dpi=300)
    colors = ["#1f5fa8", "#c4572a"]
    for (name, label), c in zip(runs, colors):
        r = [json.loads(l) for l in open(os.path.join(ckpt, name, "log.jsonl"))]
        q = np.array([x["qr"] for x in r])
        k = np.convolve(q, np.ones(w) / w, mode="valid")
        ax.plot(np.arange(len(q)), q, color=c, alpha=0.15, lw=0.6)
        ax.plot(np.arange(w - 1, len(q)), k, color=c, lw=1.4, label=label)
    ax.set_xlabel("GRPO step", fontsize=8); ax.set_ylabel("Quality ratio of sampled answers", fontsize=8)
    ax.tick_params(labelsize=7); ax.set_ylim(0, 1.05); ax.grid(alpha=0.25, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)


if __name__ == "__main__" and os.environ.get("CUAS_LLM_REPORT"):
    pass


def warm_start(results, data, tag="q7b", stage="grpo_outcome"):
    """Local search started from the model's answer versus started from greedy: quality and simulator evaluations."""
    from cuas_gym import baselines as B
    from cuas_gym.cli import load
    out = {}
    for sp in SPLITS:
        d = _llm(results, tag, stage, sp)
        rows = []
        for iid, x in sorted(d.items()):
            inst, v = load(data, iid)
            Fref = v.reference()["F"]
            v.evals = 0
            G = B.greedy(v); eg = v.evals
            v.evals = 0
            LG = B.local_search(v, G); elg = v.evals + eg
            if x["valid"]:
                r0, _ = v.rows_of(x["action"])
                v.evals = 0
                LM = B.local_search(v, r0); elm = v.evals
                q0 = v.F(r0) / Fref
            else:
                LM, elm, q0 = None, 0, 0.0
            rows.append({"q_model": q0, "q_model_ls": (v.F(LM) / Fref) if LM else np.nan, "ev_model_ls": elm,
                         "q_greedy": v.F(G) / Fref, "ev_greedy": eg, "q_greedy_ls": v.F(LG) / Fref, "ev_greedy_ls": elg})
        out[sp] = rows
    return out
