"""Recompute the detection matrix of released instances with the physics module and compare it with the stored one.

    python3 -m cuas_gym.regen_check --data data/gym2 [--split test_sensor] [--n 20]

Also regenerates each instance from its split and seed (sites, inventory, budget, data-sheet ranges) and checks that
the metadata match. Stored matrices are float16, so differences up to about 1e-3 are rounding.
"""
import argparse
import copy
import glob
import json
import os
import time

import numpy as np

from .instances import compute_matrix, generate, load_instance


def check(folder, iid, meta=True):
    inst = load_instance(folder, iid)
    re = copy.copy(inst)
    compute_matrix(re)
    out = {"iid": iid, "configs_equal": re.configs == inst.configs,
           "max_abs_dP": float(np.abs(re.P.astype(np.float16).astype(np.float32) - inst.P).max()),
           "max_abs_dP_raw": float(np.abs(re.P - inst.P).max()),
           "w_equal": bool(np.array_equal(re.w, inst.w)), "xy_max_diff": float(np.abs(re.cell_xy - inst.cell_xy).max())}
    if meta:
        split, seed = iid.rsplit("-", 1)
        g = generate(int(seed), split, compute=False, version=getattr(inst, "version", 1))
        keys = ("roi", "sites", "types", "cost", "rmax", "needlos", "budget", "alt", "quad_w", "center", "drone", "variant", "drange")
        m0 = json.loads(json.dumps(inst.meta())); m1 = json.loads(json.dumps(g.meta() | {"configs": []}))
        out["meta_equal"] = all(m0.get(k) == m1.get(k) for k in keys)
        out["meta_diff"] = [k for k in keys if m0.get(k) != m1.get(k)]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/gym2"); p.add_argument("--split", default=None); p.add_argument("--n", type=int, default=0)
    p.add_argument("--out", default=None); p.add_argument("--no-meta", action="store_true")
    a = p.parse_args()
    ids = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(a.data, "*.npz")))
    if a.split:
        ids = [i for i in ids if i.rsplit("-", 1)[0] == a.split]
    if a.n:
        ids = ids[:a.n]
    res, t0 = [], time.time()
    for i in ids:
        r = check(a.data, i, not a.no_meta)
        res.append(r)
        if not (r["configs_equal"] and r["max_abs_dP"] <= 1e-3 and r.get("meta_equal", True)):
            print("MISMATCH", r)
    dmax = max(r["max_abs_dP"] for r in res)
    ok = sum(r["configs_equal"] and r["max_abs_dP"] <= 1e-3 and r.get("meta_equal", True) for r in res)
    print(f"{ok}/{len(res)} instances reproduced; max |dP| after float16 rounding {dmax:.2e}; {time.time() - t0:.0f} s")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
