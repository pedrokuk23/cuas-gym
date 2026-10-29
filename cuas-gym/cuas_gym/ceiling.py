"""Ceiling without a sensor type: best placement when one type is excluded, as a fraction of the reference. Tells how
much a split depends on that type (per instance). By default greedy search plus local search (a feasible network, a
lower bound on the restricted optimum); with --certify, branch and bound proves the restricted optimum for the listed
instances and writes the certificate (rows, objective, QR, proof status, time limit).

    python3 -m cuas_gym.ceiling --data data/gym2 --split test_sensor --type ac --out results/v2/ceiling_test_sensor_ac.json
    python3 -m cuas_gym.ceiling --data data/gym2 --split test_sensor --type ac --certify --below 0.9 \
        --from-json results/v2/ceiling_test_sensor_ac.json --out results/v2/ceiling_certificates_ac.json
"""
import argparse
import copy
import json

import numpy as np

from .cli import ids, load
from .baselines import greedy, local_search, branch_and_bound
from .verifier import Verifier


def ceiling_without(inst, v, typ):
    keep = [i for i, c in enumerate(inst.configs) if c[1] != typ]
    if len(keep) == len(inst.configs):
        return 1.0
    inst2 = copy.copy(inst)
    inst2.configs = [inst.configs[i] for i in keep]
    inst2.P = inst.P[keep]
    v2 = Verifier(inst2)
    rows = local_search(v2, greedy(v2))
    return float(v2.F(rows) / v.F(v.reference()["rows"]))


def certify_without(inst, v, typ, time_limit=600.0):
    keep = [i for i, c in enumerate(inst.configs) if c[1] != typ]
    inst2 = copy.copy(inst)
    inst2.configs = [inst.configs[i] for i in keep]
    inst2.P = inst.P[keep]
    v2 = Verifier(inst2)
    inc = local_search(v2, greedy(v2))
    rows, proven = branch_and_bound(v2, inc, time_limit)
    Fref = v.F(v.reference()["rows"])
    return {"rows": [int(r) for r in rows], "action": [[int(s), str(t)] for s, t in (inst2.configs[r] for r in rows)],
            "F": float(v2.F(rows)), "F_reference": float(Fref), "qr": float(v2.F(rows) / Fref), "proven": bool(proven),
            "time_limit_s": time_limit, "excluded_type": typ, "tolerance": 1e-9}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/gym2"); p.add_argument("--split", default="test_sensor")
    p.add_argument("--type", default="ac"); p.add_argument("--out", required=True)
    p.add_argument("--certify", action="store_true"); p.add_argument("--below", type=float, default=0.9)
    p.add_argument("--from-json", help="greedy+local-search ceilings; certify the instances below --below")
    p.add_argument("--time-limit", type=float, default=600.0)
    a = p.parse_args()
    if a.certify:
        pre = json.load(open(a.from_json))
        todo = sorted(i for i, q in pre.items() if q < a.below)
        cert = {}
        for iid in todo:
            inst, v = load(a.data, iid)
            cert[iid] = certify_without(inst, v, a.type, a.time_limit)
            print(iid, f"QR {cert[iid]['qr']:.10f} proven {cert[iid]['proven']}", flush=True)
        json.dump(cert, open(a.out, "w"), indent=1)
        return
    out = {}
    for iid in ids(a.data, a.split):
        inst, v = load(a.data, iid)
        out[iid] = ceiling_without(inst, v, a.type)
    json.dump(out, open(a.out, "w"), indent=0)
    x = np.array(list(out.values()))
    print(f"{a.split} without {a.type}: mean {x.mean():.3f} min {x.min():.3f} below 0.9: {(x < 0.9).mean():.2f} "
          f"below 0.8: {(x < 0.8).mean():.2f} (n={len(x)})")


if __name__ == "__main__":
    main()
