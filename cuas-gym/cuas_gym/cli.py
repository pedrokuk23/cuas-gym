"""Command line of CUAS-Gym.

    python3 -m cuas_gym.cli generate --split train --n 200 --out data/gym
    python3 -m cuas_gym.cli baselines --data data/gym --split test_id --out results/baselines_test_id.csv
"""
import argparse
import csv
import glob
import json
import os
import time

import numpy as np

from . import SPLITS, Verifier, generate, load_instance, save_instance
from . import baselines as B


def cmd_generate(a):
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    for k in range(a.start, a.start + a.n):
        iid = f"{a.split}-{k:06d}"
        if os.path.exists(os.path.join(a.out, iid + ".npz")) and not a.force:
            continue
        inst = generate(k, a.split, version=a.version)
        v = Verifier(inst)
        ref = v.reference()
        inst_meta_ref = {"rows": [int(r) for r in ref["rows"]], "F": ref["F"], "proven": ref["proven"]}
        save_instance(inst, a.out)
        with open(os.path.join(a.out, iid + ".ref.json"), "w") as f:
            json.dump(inst_meta_ref, f)
        print(f"{iid}: configs {inst.P.shape[0]}, cells {inst.P.shape[1]}, budget {inst.budget:g}, drone {inst.drone}, "
              f"ref Fn {ref['F'] / v.W:.3f} proven {ref['proven']}  ({time.time() - t0:.0f} s)", flush=True)


def load(data, iid):
    inst = load_instance(data, iid)
    v = Verifier(inst)
    with open(os.path.join(data, iid + ".ref.json")) as f:
        r = json.load(f)
    v._ref = {"rows": r["rows"], "F": r["F"], "proven": r["proven"]}
    return inst, v


def ids(data, split):
    return sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(data, f"{split}-*.npz")))


def cmd_baselines(a):
    rows_out = []
    for iid in ids(a.data, a.split):
        inst, v = load(a.data, iid)
        Fref = v.reference()["F"]
        rng = np.random.default_rng(0)
        methods = {
            "random": lambda: B.random_feasible(v, rng),
            "geometric": lambda: B.geometric(v),
            "geometric_complete": lambda: B.geometric_complete(v),
            "greedy": lambda: B.greedy(v),
            "local_search": lambda: B.local_search(v, B.greedy(v)),
            "genetic": lambda: B.genetic(v, rng),
        }
        for name, fn in methods.items():
            reps = 10 if name == "random" else 1
            for _ in range(reps):
                v.evals = 0
                t = time.time()
                S = fn()
                ev, dt = v.evals, time.time() - t
                sc = v.score(v.action_of(S))
                rows_out.append({"iid": iid, "method": name, "qr": sc["F"] / Fref, "Fn": sc["Fn"], "cov90": sc["cov90"],
                                 "cost": sc["cost"], "budget": inst.budget, "n_sensors": len(S), "evals": ev, "time_s": dt,
                                 "proven_ref": v.reference()["proven"], "drone": inst.drone})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader(); w.writerows(rows_out)
    summarize(rows_out)


def summarize(rows):
    by = {}
    for r in rows:
        by.setdefault(r["method"], []).append(r)
    print(f"{'method':14s} {'QR mean':>8s} {'QR min':>7s} {'opt %':>6s} {'evals':>9s}")
    for m, rs in by.items():
        q = np.array([r["qr"] for r in rs])
        print(f"{m:14s} {q.mean():8.3f} {q.min():7.3f} {100 * (q >= 0.999).mean():6.1f} {np.mean([r['evals'] for r in rs]):9.0f}")


def main():
    p = argparse.ArgumentParser(prog="cuas_gym")
    sp = p.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("generate"); g.add_argument("--split", choices=list(SPLITS), required=True)
    g.add_argument("--n", type=int, default=10); g.add_argument("--start", type=int, default=0)
    g.add_argument("--out", default="data/gym"); g.add_argument("--force", action="store_true")
    g.add_argument("--version", type=int, default=1, choices=[1, 2])
    b = sp.add_parser("baselines"); b.add_argument("--data", default="data/gym"); b.add_argument("--split", required=True)
    b.add_argument("--out", required=True)
    a = p.parse_args()
    {"generate": cmd_generate, "baselines": cmd_baselines}[a.cmd](a)


if __name__ == "__main__":
    main()
