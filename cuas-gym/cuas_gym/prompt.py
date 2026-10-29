"""Observation text for the language model and parser of its answer."""
import json
import os
import re

import numpy as np

DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
LABEL = {"radar": "radar", "eo": "EO/IR camera", "rid": "Remote ID receiver", "rf": "passive RF receiver", "ac": "acoustic array"}

ANON_LETTERS = "abcdefgh"


def anon_on():
    """Anonymized inventory (CUAS_ANON=1): sensor types are described by their properties only, under letter
    labels permuted per instance, so that the model cannot rely on the type name. Version-2 instances only."""
    return os.environ.get("CUAS_ANON", "0") == "1"


def anon_map(inst):
    """type -> letter label, a permutation fixed by the instance seed."""
    rng = np.random.default_rng(int(getattr(inst, "seed", 0)) * 7919 + 17)
    order = rng.permutation(len(inst.types))
    return {t: ANON_LETTERS[k] for t, k in zip(inst.types, order)}


SYSTEM = ("You are an expert in air surveillance planning. You place sensors on candidate sites to detect small drones "
          "over terrain. Think step by step about line of sight, sensor range and overlap, then give your answer.")


def build_prompt(inst, attrs=True):
    """attrs=False removes the terrain-derived attributes (relative height and horizon angles): ablation."""
    L = []
    L.append("Task: choose sensors and sites to maximize the weighted probability of detecting a small drone over the "
             "region of interest (ROI). Detections of different sensors are fused: a cell is missed only if every sensor misses it.")
    L.append(f"Target: {drone_text(inst.drone)}, flying {inst.alt:g} m above ground.")
    xy = _roi_km(inst)
    L.append("ROI polygon (x east, y north, km from the ROI center): " + ", ".join(f"({x:.1f}, {y:.1f})" for x, y in xy))
    L.append("Quadrant weights (importance of each part of the ROI): " +
             ", ".join(f"{q}={w}" for q, w in zip(["NE", "NW", "SW", "SE"], inst.quad_w)))
    L.append(f"Budget: {inst.budget:g} cost units.")
    anon = anon_on()
    if anon:
        assert getattr(inst, "version", 1) == 2, "anonymized inventory needs version-2 instances"
        lab = anon_map(inst)
        L.append("Inventory (sensor types are identified by a letter and described by their properties; any type may be "
                 "installed on any site, and a site may host several types):")
    else:
        L.append("Inventory (any type may be installed on any site; a site may host several types):")
    for t in sorted(inst.types, key=lambda t: lab[t]) if anon else inst.types:
        los = "needs line of sight" if inst.needlos[t] else "does not need line of sight"
        if anon:
            L.append(f"- {lab[t]}: cost {inst.cost[t]:g}, detection range against this drone at this altitude on open "
                     f"terrain {inst.drange[t]:g} km, {los}")
        elif getattr(inst, "version", 1) == 2:
            L.append(f"- {t}: {inst.variant[t]['label']}, cost {inst.cost[t]:g}, detection range against this drone at this "
                     f"altitude on open terrain {inst.drange[t]:g} km, {los}")
        else:
            L.append(f"- {t}: {LABEL.get(t, t)}, cost {inst.cost[t]:g}, nominal range {inst.rmax[t] / 1000:g} km, {los}")
    if attrs:
        L.append("Candidate sites: id | x km | y km | ground elevation m | height above ROI median m | horizon elevation angle "
                 "(deg) toward " + " ".join(DIRS) + " within 3 km (lower = more open view)")
        for s in inst.sites:
            L.append(f"{s['id']} | {s['x']:.2f} | {s['y']:.2f} | {s['elev']} | {s['rel']:+d} | " + " ".join(f"{a:g}" for a in s["horizon"]))
    else:
        L.append("Candidate sites: id | x km | y km | ground elevation m")
        for s in inst.sites:
            L.append(f"{s['id']} | {s['x']:.2f} | {s['y']:.2f} | {s['elev']}")
    L.append("Adding a sensor never reduces detection, and unspent budget is wasted.")
    if anon:
        L.append("Answer format: reason briefly, then end with exactly one line\n"
                 '<answer>{"placements": [{"site": <site id>, "type": "<sensor letter>"}, ...]}</answer>\n'
                 "listing each placement once, sorted by site id, with a total cost within the budget.")
    elif getattr(inst, "version", 1) == 2:
        L.append("Answer format: reason briefly, then end with exactly one line\n"
                 '<answer>{"placements": [{"site": <site id>, "type": "<sensor type>"}, ...]}</answer>\n'
                 "listing each placement once, sorted by site id, with a total cost within the budget.")
    else:
        L.append("Answer format: reason briefly, then end with exactly one line\n"
                 '<answer>{"placements": [{"site": <site id>, "type": "<sensor type>"}, ...]}</answer>\n'
                 "listing the chosen placements in order of importance, with a total cost within the budget.")
    return "\n".join(L)


def drone_text(key):
    from .catalog import DRONES
    d = next(x for x in DRONES if x["key"] == key)
    rid = "broadcasts Remote ID" if d["ridPt"] is not None else "no Remote ID broadcast"
    link = "radio control/video link" if d["c2Pt"] is not None else "no radio emissions"
    return f"{d['name']}, radar cross section {d['rcs']:g} m2, size {d['dc']:g} m, {rid}, {link}"


def _roi_km(inst):
    """ROI vertices in km in the same local frame as the sites (centred on inst.center)."""
    import math
    lat0, lon0 = inst.center
    my, mx = 111320.0, 111320.0 * math.cos(math.radians(lat0))
    return [((lo - lon0) * mx / 1000, (la - lat0) * my / 1000) for la, lo in inst.roi]


_ANS = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)


def parse_answer(text):
    """Returns a list of (site, type) or None when no valid JSON answer is found."""
    m = _ANS.findall(text or "")
    cands = m[::-1] if m else re.findall(r"\{[^{}]*\"placements\"\s*:\s*\[.*?\]\s*\}", text or "", re.S)[::-1]
    for c in cands:
        try:
            d = json.loads(c.strip())
            pl = d["placements"] if isinstance(d, dict) else d
            out = [(int(p["site"]), str(p["type"]).strip().lower()) for p in pl]
            return out
        except (ValueError, KeyError, TypeError):
            continue
    return None
