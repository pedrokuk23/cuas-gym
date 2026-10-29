"""Which sensor types a model chooses, and how it does where a type is decisive.

    python3 -m cuas_rlvr.probe_types results/v2/mlx_v2_q7b_grpo_s0_test_sensor.jsonl \
        --ceiling results/v2/ceiling_test_sensor_ac.json --type ac [--anon]

With --anon, the answers use letter labels and are mapped back to types with the instance's permutation.
"""
import argparse
import collections
import json
import os

import numpy as np

from cuas_gym.cli import load
from cuas_gym.prompt import anon_map


def types_of(action, lab_inv):
    pl = action if isinstance(action, list) else (action or {}).get("placements", [])
    out = []
    for p in pl:
        t = p[1] if isinstance(p, (list, tuple)) else p.get("type")
        t = str(t).strip().lower()
        out.append(lab_inv.get(t, t))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+"); p.add_argument("--data", default="data/gym2")
    p.add_argument("--ceiling"); p.add_argument("--type", default="ac"); p.add_argument("--anon", action="store_true")
    p.add_argument("--thr", type=float, default=0.9)
    a = p.parse_args()
    ceil = json.load(open(a.ceiling)) if a.ceiling else {}
    for f in a.files:
        recs = [json.loads(l) for l in open(f)]
        cnt, with_t, n_valid = collections.Counter(), 0, 0
        qr_dec, qr_rest, qr_ref_dec = [], [], []
        cache = {}
        for r in recs:
            iid = r["iid"]
            if iid not in cache:
                inst, _ = load(a.data, iid)
                cache[iid] = {lab: t for t, lab in anon_map(inst).items()} if a.anon else {}
            if r["valid"]:
                n_valid += 1
                ts = types_of(r["action"], cache[iid])
                cnt.update(ts)
                with_t += a.type in ts
            if iid in ceil:
                (qr_dec if ceil[iid] < a.thr else qr_rest).append(r["qr"])
                if ceil[iid] < a.thr:
                    qr_ref_dec.append(ceil[iid])
        tot = sum(cnt.values()) or 1
        print(os.path.basename(f))
        print(f"  n={len(recs)} valid={n_valid / len(recs):.2f} QR={np.mean([r['qr'] for r in recs]):.3f}  "
              f"type shares: " + ", ".join(f"{t} {c / tot:.2f}" for t, c in cnt.most_common()))
        print(f"  answers with '{a.type}': {with_t}/{n_valid} valid answers")
        if ceil:
            print(f"  instances where ceiling without '{a.type}' < {a.thr}: n={len(qr_dec)}, model QR {np.mean(qr_dec) if qr_dec else float('nan'):.3f} "
                  f"(ceiling {np.mean(qr_ref_dec) if qr_ref_dec else float('nan'):.3f}); other instances: n={len(qr_rest)}, model QR {np.mean(qr_rest):.3f}")


if __name__ == "__main__":
    main()
