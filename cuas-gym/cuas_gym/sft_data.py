"""Supervised fine-tuning data: prompt -> short rationale + reference answer, for the training split.

The rationale is written from facts the model can see (drone emissions, horizon angles, ranges, costs), so that SFT
teaches the format and a sensible reasoning style without leaking the detection matrix.

    python3 -m cuas_gym.sft_data --data data/gym --out data/sft_train.jsonl
"""
import argparse
import json

import numpy as np

from .cli import ids, load
from .baselines import greedy_order
from .prompt import SYSTEM, build_prompt, anon_on, anon_map


def rationale(inst, action):
    from .catalog import DRONES
    d = next(x for x in DRONES if x["key"] == inst.drone)
    site = {s["id"]: s for s in inst.sites}
    L = []
    if d["ridPt"] is None and "rid" in inst.types:
        L.append("The drone does not broadcast Remote ID, so Remote ID receivers are useless here.")
    if d["c2Pt"] is None and "rf" in inst.types:
        L.append("It has no radio emissions, so passive RF receivers cannot detect it.")
    elif d["c2Pt"] is not None and "rf" in inst.types:
        L.append("Its radio link can be picked up by passive RF receivers, which do not need line of sight.")
    v2 = getattr(inst, "version", 1) == 2
    if anon_on():
        lab = anon_map(inst)
        L = []
        useless = [lab[t] for t in inst.types if inst.drange[t] <= 0]
        if useless:
            L.append("Types " + ", ".join(sorted(useless)) + " have zero range against this drone and are useless here.")
        L.append("Ranges and costs: " + ", ".join(f"{lab[t]} {inst.drange[t]:g} km for cost {inst.cost[t]:g}"
                                                  for t in sorted(inst.types, key=lambda t: lab[t])) + ".")
    elif v2:
        L.append("Detection ranges against this drone: " + ", ".join(f"{t} {inst.drange[t]:g} km for cost {inst.cost[t]:g}"
                                                             for t in inst.types) + ".")
    elif d["rcs"] < 0.05 and "radar" in inst.types:
        L.append(f"Its radar cross section is small ({d['rcs']:g} m2), which shortens the radar range.")
    best_w = int(np.argmax(inst.quad_w))
    L.append(f"The {['NE', 'NW', 'SW', 'SE'][best_w]} quadrant has the highest weight.")
    cost = 0.0
    for k, (s, t) in enumerate(action, 1):
        st = site[s]
        hz = float(np.mean(st["horizon"]))
        why = f"open view (mean horizon {hz:.1f} deg)" if inst.needlos[t] else "no line of sight needed"
        if v2:
            why = f"range {inst.drange[t]:g} km, " + why
        cost += inst.cost[t]
        if anon_on():
            L.append(f"site {s} ({st['x']:.1f}, {st['y']:.1f}) km: {lab[t]}, {why}; cost so far {cost:g} of {inst.budget:g}.")
        elif v2:
            L.append(f"site {s} ({st['x']:.1f}, {st['y']:.1f}) km: {t}, {why}; cost so far {cost:g} of {inst.budget:g}.")
        else:
            L.append(f"{k}. {t} at site {s} ({st['x']:.1f}, {st['y']:.1f}) km, {why}; cost so far {cost:g} of {inst.budget:g}.")
    return " ".join(L[:-len(action)]) + "\n" + "\n".join(L[-len(action):])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/gym"); p.add_argument("--split", default="train")
    p.add_argument("--out", default="data/sft_train.jsonl")
    a = p.parse_args()
    n = 0
    with open(a.out, "w") as f:
        for iid in ids(a.data, a.split):
            inst, v = load(a.data, iid)
            act = v.action_of(greedy_order(v, v.reference()["rows"]))
            if getattr(inst, "version", 1) == 2:          # v2 answers are sets listed by site id (no repeats)
                order = {t: k for k, t in enumerate(inst.types)}
                act = sorted(act, key=lambda x: (x[0], order[x[1]]))
            lab = anon_map(inst) if anon_on() else {}
            ans = json.dumps({"placements": [{"site": s, "type": lab.get(t, t)} for s, t in act]})
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(inst)},
                    {"role": "assistant", "content": rationale(inst, act) + "\n<answer>" + ans + "</answer>"}]
            f.write(json.dumps({"iid": iid, "messages": msgs}) + "\n")
            n += 1
    print(f"{n} examples -> {a.out}")


if __name__ == "__main__":
    main()
