"""Tables, statistics and figures for the version-2 experiments (three GRPO seeds and the anonymized inventory).

    python3 -m cuas_rlvr.report_v2 --results results/v2 --data data/gym2 --ckpt ckpt --out out
"""
import argparse
import json
import os

import numpy as np

from .report import SPLITS, NAMES, _geo, _llm, ci, paired

SEEDS = (0, 1, 2)


def seed_matrix(results, tag, sp, seeds=SEEDS):
    """ids, QR matrix (instances x seeds), valid matrix."""
    runs = [_llm(results, tag, f"grpo_s{s}", sp) for s in seeds]
    runs = [r for r in runs if r is not None]
    ids = sorted(set.intersection(*[set(r) for r in runs]))
    Q = np.array([[r[i]["qr"] for r in runs] for i in ids])
    V = np.array([[r[i]["valid"] for r in runs] for i in ids])
    return ids, Q, V


def llm_table_v2(results, tag="v2_q7b", anon_tag="v2anon_q7b", base_name="Qwen2.5-7B"):
    lines = []
    for sp in SPLITS:
        geo = _geo(results, sp)
        rows = []
        q = np.array([geo[i] for i in sorted(geo)])
        rows.append(("Geometric heuristic", "--", f"{q.mean():.3f} [{ci(q)[0]:.3f}, {ci(q)[1]:.3f}]", f"{q.min():.2f}"))
        gc = _geo(results, sp, "geometric_complete")
        if gc:
            q = np.array([gc[i] for i in sorted(gc)])
            rows.append(("Geometric heuristic + budget completion", "--", f"{q.mean():.3f} [{ci(q)[0]:.3f}, {ci(q)[1]:.3f}]", f"{q.min():.2f}"))
        for st, label in (("base", f"{base_name}, zero-shot"), ("sft", "+ SFT")):
            d = _llm(results, tag, st, sp)
            if d is None:
                continue
            qq = np.array([d[i]["qr"] for i in sorted(d)]); vv = np.mean([d[i]["valid"] for i in d])
            rows.append((label, f"{100 * vv:.0f}", f"{qq.mean():.3f} [{ci(qq)[0]:.3f}, {ci(qq)[1]:.3f}]", f"{qq.min():.2f}"))
        ids, Q, V = seed_matrix(results, tag, sp)
        per_seed = Q.mean(0); per_inst = Q.mean(1)
        lo, hi = ci(per_inst)
        rows.append(("+ SFT + GRPO (3 seeds)", f"{100 * V.mean():.0f}",
                     f"{per_seed.mean():.3f} $\\pm$ {per_seed.std(ddof=1):.3f} [{lo:.3f}, {hi:.3f}]", f"{per_inst.min():.2f}"))
        if _llm(results, anon_tag, "grpo_s0", sp) is not None:
            ids, Q, V = seed_matrix(results, anon_tag, sp)
            per_seed = Q.mean(0); per_inst = Q.mean(1)
            lo, hi = ci(per_inst)
            sd = f" $\\pm$ {per_seed.std(ddof=1):.3f}" if len(per_seed) > 1 else ""
            rows.append((f"+ SFT + GRPO, anonymized inventory ({len(per_seed)} seeds)", f"{100 * V.mean():.0f}",
                         f"{per_seed.mean():.3f}{sd} [{lo:.3f}, {hi:.3f}]", f"{per_inst.min():.2f}"))
        first = True
        for label, vs, qs, ws in rows:
            sp_lbl = f"\\multirow{{{len(rows)}}}{{*}}{{{NAMES[sp]}}}" if first else ""
            first = False
            lines.append(f"{sp_lbl} & {label} & {vs} & {qs} & {ws} \\\\")
        lines.append("\\midrule")
    return lines[:-1] + ["\\bottomrule"]


def stats_v2(results, tag="v2_q7b", anon_tag="v2anon_q7b"):
    out = {}
    for sp in SPLITS:
        geo = _geo(results, sp); s = _llm(results, tag, "sft", sp); b = _llm(results, tag, "base", sp)
        gc = _geo(results, sp, "geometric_complete")
        ids, Q, V = seed_matrix(results, tag, sp)
        m = Q.mean(1)
        o = {"n": len(ids), "seed_means": Q.mean(0).round(3).tolist(), "seed_valid": V.mean(0).round(3).tolist(),
             "mean_vs_geo": paired(m, [geo[i] for i in ids]), "mean_vs_sft": paired(m, [s[i]["qr"] for i in ids]) if s else None,
             "per_seed_vs_geo": [paired(Q[:, k], [geo[i] for i in ids]) for k in range(Q.shape[1])],
             "worst_seed_min_vs_geo": min(paired(Q[:, k], [geo[i] for i in ids])[1] for k in range(Q.shape[1]))}
        if gc:
            o["mean_vs_geo_complete"] = paired(m, [gc[i] for i in ids])
            o["base_vs_geo_complete"] = paired([b[i]["qr"] for i in ids], [gc[i] for i in ids]) if b else None
        if _llm(results, anon_tag, "grpo_s0", sp) is not None:
            aids, AQ, AV = seed_matrix(results, anon_tag, sp)
            am = dict(zip(aids, AQ.mean(1)))
            o["anon_seed_means"] = AQ.mean(0).round(3).tolist(); o["anon_seed_valid"] = AV.mean(0).round(3).tolist()
            o["anon_vs_mean"] = paired([am[i] for i in ids], m)
            o["anon_vs_geo"] = paired([am[i] for i in ids], [geo[i] for i in ids])
            if gc:
                o["anon_vs_geo_complete"] = paired([am[i] for i in ids], [gc[i] for i in ids])
            sa = _llm(results, anon_tag, "sft", sp)   # SFT checkpoint of the same (property-based) inventory
            o["anon_vs_anon_sft"] = paired([am[i] for i in ids], [sa[i]["qr"] for i in ids]) if sa else None
            o["anon_per_seed_vs_geo"] = [paired(AQ[:, k], [geo[i] for i in aids]) for k in range(AQ.shape[1])]
            o["anon_valid"] = float(AV.mean())
        out[sp] = o
    return out


def curves_v2(ckpt, out_png, tag="v2_q7b", anon_tag="v2anon_q7b", w=25):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), dpi=300)
    runs = [(f"{tag}_grpo_s{s}", f"named, seed {s}") for s in SEEDS] + [(f"{anon_tag}_grpo_s{s}", f"anonymized, seed {s}") for s in SEEDS]
    colors = ["#1f5fa8", "#4f8fd0", "#9ec1e8", "#c4572a", "#e08a5a", "#f2bf9a"]
    for (name, label), c in zip(runs, colors):
        f = os.path.join(ckpt, name, "log.jsonl")
        if not os.path.exists(f):
            continue
        r = [json.loads(l) for l in open(f)]
        for ax, key in zip(axes, ("qr", "valid")):
            y = np.array([x[key] for x in r])
            k = np.convolve(y, np.ones(w) / w, mode="valid")
            ax.plot(np.arange(w - 1, len(y)), k, color=c, lw=1.3, label=label)
    axes[0].set_ylabel("Sampled quality ratio", fontsize=8); axes[1].set_ylabel("Valid answers (fraction)", fontsize=8)
    for ax in axes:
        ax.set_xlabel("GRPO step", fontsize=8); ax.tick_params(labelsize=7); ax.grid(alpha=0.25, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylim(0.5, 1.02); axes[1].set_ylim(0.5, 1.02)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.14, 1, 1)); fig.savefig(out_png, bbox_inches="tight"); plt.close(fig)


def behavior_v2(results, data, tag="v2_q7b", anon_tag="v2anon_q7b"):
    """Budget use and sensor-type shares of the trained models against the reference, per split."""
    from cuas_gym.cli import load
    from cuas_gym.prompt import anon_map
    out = {}
    for sp in SPLITS:
        o = {}
        for lab, tg, st in [("named", tag, "grpo_s0"), ("named", tag, "grpo_s1"), ("named", tag, "grpo_s2"), ("anon", anon_tag, "grpo_s0"), ("anon", anon_tag, "grpo_s1"), ("anon", anon_tag, "grpo_s2")]:
            d = _llm(results, tg, st, sp)
            if d is None:
                continue
            bu, types, nref = [], {}, {}
            for iid, x in d.items():
                inst, v = load(data, iid)
                inv = {v2: k for k, v2 in anon_map(inst).items()} if lab == "anon" else {}
                for t in [inst.configs[r][1] for r in v.reference()["rows"]]:
                    nref[t] = nref.get(t, 0) + 1
                if not x["valid"]:
                    continue
                act = [(s_, inv.get(str(t).lower(), str(t).lower())) for s_, t in x["action"]]
                rows, err = v.rows_of(act)
                if rows is None:
                    continue
                bu.append(v.cost[rows].sum() / v.budget)
                for _, t in act:
                    types[t] = types.get(t, 0) + 1
            tot = sum(types.values()) or 1; rtot = sum(nref.values()) or 1
            o[f"{lab}:{st}"] = {"budget_use": float(np.mean(bu)), "shares": {t: round(c / tot, 3) for t, c in sorted(types.items())},
                                "ref_shares": {t: round(c / rtot, 3) for t, c in sorted(nref.items())}}
        out[sp] = o
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/v2"); p.add_argument("--data", default="data/gym2")
    p.add_argument("--ckpt", default="ckpt"); p.add_argument("--out", default="out")
    p.add_argument("--tag", default="v2_q7b"); p.add_argument("--anon-tag", default="v2anon_q7b")
    p.add_argument("--base-name", default="Qwen2.5-7B"); p.add_argument("--suffix", default="", help="table/figure file suffix, e.g. _q3_8b")
    a = p.parse_args()
    os.makedirs(os.path.join(a.out, "tables", "v2"), exist_ok=True); os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    tex = os.path.join(a.out, "tables", "v2", f"llm{a.suffix}.tex")
    with open(tex, "w") as f:
        f.write("\n".join(llm_table_v2(a.results, a.tag, a.anon_tag, a.base_name)) + "\n")
    print(open(tex).read())
    st = stats_v2(a.results, a.tag, a.anon_tag)
    for sp, o in st.items():
        print(sp, json.dumps({k: (np.round(v, 3).tolist() if isinstance(v, (tuple, list, np.ndarray)) else v) for k, v in o.items()}))
    curves_v2(a.ckpt, os.path.join(a.out, "figures", f"training_curves_v2{a.suffix}.png"), a.tag, a.anon_tag)
    curves_v2(a.ckpt, os.path.join(a.out, "figures", f"training_curves_v2{a.suffix}.pdf"), a.tag, a.anon_tag)
    print(json.dumps(behavior_v2(a.results, a.data, a.tag, a.anon_tag), indent=1))


if __name__ == "__main__":
    main()
