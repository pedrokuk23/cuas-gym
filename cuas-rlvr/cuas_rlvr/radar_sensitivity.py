"""Sensitivity of the benchmark to the radar detection curve (supplementary material, Section S1).

The physics module obtains the radar detection probability by inverting Shnidman's approximation between P_d = 0.1 and
0.99 and continues it with empirical tails outside that range. This script recomputes every test detection matrix with
the exact detection probability of a Swerling 1 or 3 target (square-law detector, N pulses integrated noncoherently),
recomputes the reference of each instance with the same procedure as the released one, and re-scores the references,
the completed geometric heuristic and every trained language-model answer.

    python3 -m cuas_rlvr.radar_sensitivity compute --data data/gym2 --split test_id --matrices ../radar_exact  # per split
    python3 -m cuas_rlvr.radar_sensitivity compare --data data/gym2 --results results/v2 --matrices ../radar_exact \
        --out results/v2/radar_sensitivity.json

`compute` needs the Copernicus tile (as cuas_gym.regen_check) and scipy; the recomputed matrices are not released.
"""
import argparse
import copy
import glob
import json
import os

import numpy as np

from .report import SPLITS, _llm, paired

PLANNERS = [("27B names", "q38_27b", 0), ("27B properties", "q38_27banon", 1), ("8B names", "q3_8b", 0),
            ("8B properties", "q3_8banon", 1), ("7B names", "v2_q7b", 0), ("7B properties", "v2anon_q7b", 1)]


# ---------------------------------------------------------------------------------------------------------------------
# Exact Swerling 1 and 3 detection
def swerling_table(N, pfa, sw, lo=-40.0, hi=60.0, step=0.02):
    """P_d on a grid of SNR per pulse (dB). Conditional on the scan's cross-section ratio x, the integrated statistic
    is noncentral chi-square with 2N degrees of freedom and noncentrality 2 N snr x; x ~ Exp(1) for Swerling 1 and
    Gamma(2, 1/2) for Swerling 3. The average over x is a trapezoid rule in log x."""
    from scipy import special, stats
    T = special.gammainccinv(N, pfa)
    k = 1.0 if sw == 1 else 2.0
    u = np.linspace(np.log(1e-9), np.log(60.0), 4000)
    x = np.exp(u)
    wx = stats.gamma.pdf(x, k, scale=1.0 / k) * x
    db = np.arange(lo, hi + step / 2, step)
    pd = np.array([np.trapezoid(stats.ncx2.sf(2 * T, 2 * N, 2 * N * s * x) * wx, u) for s in 10 ** (db / 10)])
    return db, np.clip(pd, pfa, 1.0)


_TABLES = {}


def swerling_pd(snr, N, pfa, sw):
    """Same signature as cuas_gym.physics.models.shnidman_pd."""
    if sw not in (1, 3):
        raise ValueError("the benchmark threats use Swerling 1 or 3")
    if (N, pfa, sw) not in _TABLES:
        _TABLES[(N, pfa, sw)] = swerling_table(N, pfa, sw)
    db, pd = _TABLES[(N, pfa, sw)]
    snr = np.asarray(snr, dtype=float)
    with np.errstate(divide="ignore"):
        d = 10 * np.log10(np.where(snr > 0, snr, 1e-30))
    return np.where(snr > 0, np.interp(d, db, pd, left=pd[0], right=pd[-1]), 0.0)


def compute(data, split, out):
    """Recompute the matrices of one split with exact Swerling detection and their references."""
    import cuas_gym.physics.models as M
    from cuas_gym.baselines import reference
    from cuas_gym.instances import compute_matrix, load_instance
    from cuas_gym.verifier import Verifier
    M.shnidman_pd = swerling_pd
    os.makedirs(out, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(data, f"{split}-*.npz"))):
        iid = os.path.basename(f)[:-4]
        dst = os.path.join(out, iid + ".npz")
        if os.path.exists(dst):
            continue
        inst = load_instance(data, iid)
        new = copy.copy(inst)
        compute_matrix(new)
        assert new.configs == inst.configs
        ref = reference(Verifier(new), time_limit=20.0)
        np.savez_compressed(dst, P=new.P.astype(np.float16), ref=np.array(ref["rows"]), proven=ref["proven"])
        print(iid, "proven" if ref["proven"] else "", flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# Comparison
def fused(P, w, rows):
    q = np.ones(P.shape[1])
    for r in rows:
        q *= 1.0 - P[r]
    return float(w @ (1.0 - q))


def rows_of(inst, action, anon):
    from cuas_gym.prompt import anon_map
    inv = {lab: t for t, lab in anon_map(inst).items()} if anon else {}
    row = {(int(s), str(t)): i for i, (s, t) in enumerate(inst.configs)}
    return [row[(int(s), inv.get(str(t).strip().lower(), str(t).strip().lower()))] for s, t in action]


def compare(data, results, matrices):
    from cuas_gym.baselines import geometric_complete
    from cuas_gym.instances import load_instance
    from cuas_gym.verifier import Verifier
    out = {}
    for sp in SPLITS:
        ids = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(matrices, f"{sp}-*.npz")))
        runs = {lab: [_llm(results, tag, f"grpo_s{s}", sp) for s in range(3)] for lab, tag, _ in PLANNERS}
        dP, old_ref, same, rad_old, rad_new = [], [], 0, 0, 0
        q0 = {k: [] for k in ["Geometric, completed"] + [p[0] for p in PLANNERS]}
        q1 = {k: [] for k in q0}
        for iid in ids:
            inst = load_instance(data, iid)
            a = np.load(os.path.join(matrices, iid + ".npz"))
            P0, P1, w = inst.P.astype(np.float64), a["P"].astype(np.float64), inst.w.astype(np.float64)
            rad = np.array([t == "radar" for _, t in inst.configs])
            pos = P0[rad] > 0
            if pos.any():
                dP.append(float(np.abs(P1[rad][pos] - P0[rad][pos]).mean()))
            ref0 = json.load(open(os.path.join(data, iid + ".ref.json")))["rows"]
            ref1 = [int(r) for r in a["ref"]]
            F0, F1 = fused(P0, w, ref0), fused(P1, w, ref1)
            old_ref.append(fused(P1, w, ref0) / F1)
            same += sorted(ref0) == sorted(ref1)
            rad_old += any(rad[r] for r in ref0); rad_new += any(rad[r] for r in ref1)
            g = geometric_complete(Verifier(inst))
            q0["Geometric, completed"].append(fused(P0, w, g) / F0); q1["Geometric, completed"].append(fused(P1, w, g) / F1)
            for lab, _, anon in PLANNERS:
                s0, s1 = [], []
                for run in runs[lab]:
                    r = run[iid]
                    rows = rows_of(inst, r["action"], anon) if r["valid"] else None
                    s0.append(fused(P0, w, rows) / F0 if rows else 0.0); s1.append(fused(P1, w, rows) / F1 if rows else 0.0)
                q0[lab].append(np.mean(s0)); q1[lab].append(np.mean(s1))
        geo0, geo1 = np.array(q0["Geometric, completed"]), np.array(q1["Geometric, completed"])
        res = {"n": len(ids), "mean_abs_change_radar_pd": float(np.mean(dP)),
               "released_reference_qr_under_exact": {"mean": float(np.mean(old_ref)), "min": float(np.min(old_ref))},
               "same_reference_network": same, "references_with_radar": {"released": rad_old, "exact": rad_new},
               "planners": {}}
        for k in q0:
            a0, a1 = np.array(q0[k]), np.array(q1[k])
            row = {"qr_released": float(a0.mean()), "qr_exact": float(a1.mean()), "exact_minus_released": list(paired(a1, a0))}
            if k != "Geometric, completed":
                row["vs_geometric_released"] = list(paired(a0, geo0)); row["vs_geometric_exact"] = list(paired(a1, geo1))
            res["planners"][k] = row
        out[sp] = res
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["compute", "compare"])
    p.add_argument("--data", default="data/gym2"); p.add_argument("--results", default="results/v2")
    p.add_argument("--split", default="test_id"); p.add_argument("--matrices", default="../radar_exact")
    p.add_argument("--out", default="results/v2/radar_sensitivity.json")
    a = p.parse_args()
    if a.mode == "compute":
        compute(a.data, a.split, a.matrices)
        return
    res = compare(a.data, a.results, a.matrices)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    for sp, r in res.items():
        print(sp, json.dumps({k: v for k, v in r.items() if k != "planners"}))
        for k, v in r["planners"].items():
            print(f"  {k:22s} released {v['qr_released']:.4f} exact {v['qr_exact']:.4f} "
                  f"change {v['exact_minus_released'][0]:+.4f} [{v['exact_minus_released'][1]:+.4f}, {v['exact_minus_released'][2]:+.4f}]")


if __name__ == "__main__":
    main()
