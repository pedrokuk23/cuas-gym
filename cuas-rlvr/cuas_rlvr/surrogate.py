"""Learned non-LLM baseline: a gradient-boosted surrogate of the detection matrix followed by greedy search.

The surrogate sees only what the language model sees (site position, elevation, relative height and the eight
horizon angles; the inventory's cost, data-sheet range and line-of-sight need; threat type and altitude; the
region of interest as cells and weights) and never the physics module. It is trained by regression on the detection
matrices of the 300 training instances, the same instances used to train the language models. At test time it predicts
a matrix, the greedy search of the simulator-based baselines runs on the prediction, and the verifier scores the
chosen network on the true matrix.

    python3 -m cuas_rlvr.surrogate --data data/gym2 --out results/v2/surrogate
Variants: --mode names (sensor type identity as a feature; the unseen type has no identity) or properties.
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np

from cuas_gym.cli import ids, load
from cuas_gym.baselines import greedy

SPLITS = ["test_id", "test_terrain", "test_sensor", "test_scale"]
KNOWN = ["radar", "eo", "rid", "rf"]
DRONES = ["fiber", "fpv", "owa"]


def features(inst, mode):
    """Feature matrix (n_configs * n_cells, k) in row-major order of P."""
    xy = np.asarray(inst.cell_xy, float) / 1000.0
    site = {s["id"]: s for s in inst.sites}
    blocks = []
    for sid, t in inst.configs:
        s = site[sid]
        dx, dy = xy[:, 0] - s["x"], xy[:, 1] - s["y"]
        d = np.hypot(dx, dy) + 1e-3
        brg = (np.degrees(np.arctan2(dx, dy)) % 360.0) / 45.0          # 0 = north, clockwise, in sectors
        hz = np.asarray(s["horizon"], float)
        k0 = np.floor(brg).astype(int) % 8; fr = brg - np.floor(brg)
        hdir = (1 - fr) * hz[k0] + fr * hz[(k0 + 1) % 8]
        rng_km = float(inst.drange[t])
        elev_ang = np.degrees(np.arctan2(inst.alt - s["rel"] - 10.0, d * 1000.0))
        cols = [d, d / max(rng_km, 1e-3), np.full_like(d, rng_km), np.full_like(d, float(inst.needlos[t])),
                np.full_like(d, float(inst.cost[t])), hdir, np.full_like(d, hz.mean()), np.full_like(d, hz.max()),
                elev_ang, elev_ang - hdir, np.full_like(d, float(inst.alt)), np.full_like(d, float(s["rel"])),
                np.full_like(d, float(s["elev"]))]
        cols += [np.full_like(d, float(inst.drone == k)) for k in DRONES]
        if mode == "names":
            cols += [np.full_like(d, float(t == k)) for k in KNOWN]
        blocks.append(np.stack(cols, 1))
    return np.concatenate(blocks, 0).astype(np.float32)


class _V:
    """Verifier view with a predicted matrix (same costs, weights and budget)."""
    def __init__(self, v, P):
        self.P, self.w, self.cost, self.budget, self.evals = P, v.w, v.cost, v.budget, 0

    def miss(self, rows):
        q = np.ones(self.P.shape[1])
        for r in rows:
            q *= 1.0 - self.P[r]
        return q

    def F(self, rows):
        return float(self.w @ (1.0 - self.miss(rows)))

    def all_gains(self, q):
        return self.P @ (q * self.w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/gym2"); ap.add_argument("--out", default="results/v2/surrogate")
    ap.add_argument("--mode", default="properties", choices=["names", "properties"])
    ap.add_argument("--rows", type=int, default=2_000_000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingRegressor
    rng = np.random.default_rng(a.seed)
    tr = ids(a.data, "train")
    per = a.rows // len(tr)
    X, Y = [], []
    t0 = time.time()
    for iid in tr:
        inst, v = load(a.data, iid)
        F = features(inst, a.mode); y = np.asarray(inst.P, np.float32).ravel()
        k = rng.choice(len(y), min(per, len(y)), replace=False)
        X.append(F[k]); Y.append(y[k])
    X, Y = np.concatenate(X), np.concatenate(Y)
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=100,
                                      l2_regularization=1.0, random_state=a.seed)
    m.fit(X, Y)
    t_fit = time.time() - t0
    os.makedirs(a.out, exist_ok=True)
    rows_out = []
    for sp in SPLITS:
        for iid in ids(a.data, sp):
            inst, v = load(a.data, iid)
            t1 = time.time()
            Ph = np.clip(m.predict(features(inst, a.mode)), 0.0, 1.0).reshape(inst.P.shape)
            S = greedy(_V(v, Ph))
            dt = time.time() - t1
            F = v.F(S); Fs = v.reference()["F"]
            mae = float(np.abs(Ph - inst.P).mean())
            rows_out.append({"iid": iid, "split": sp, "qr": F / Fs, "valid": 1, "mae": mae, "time_s": dt,
                             "action": json.dumps(v.action_of(S))})
    f = os.path.join(a.out, f"surrogate_{a.mode}_s{a.seed}.csv")
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
    for sp in SPLITS:
        q = [r["qr"] for r in rows_out if r["split"] == sp]; e = [r["mae"] for r in rows_out if r["split"] == sp]
        tt = [r["time_s"] for r in rows_out if r["split"] == sp]
        print(f"{a.mode} s{a.seed} {sp}: QR {np.mean(q):.3f}  MAE(P) {np.mean(e):.3f}  {np.mean(tt):.2f} s/instance")
    print(f"fit {t_fit:.0f} s on {len(Y)} rows")


if __name__ == "__main__":
    main()
