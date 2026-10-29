"""CUAS-Gym v2: sensor variants per instance and their catalog detection range.

Each instance draws one variant per sensor type (different power, optics, sensitivity or array size, and a cost that
grows with performance). The language model is told, for each variant, its detection range against the instance's
drone at the instance's altitude on open (flat) terrain, as a planner would read it from a data sheet; terrain masking
is left for the model to reason about.
"""
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(_HERE, "datasheet_cache.json")

# (label, model parameters, common overrides, cost)
VARIANTS = {
    "radar": [("compact radar (200 W)", {"pt": 200}, {"rMax": 5000}, 4),
              ("medium radar (1 kW)", {"pt": 1000}, {"rMax": 8000}, 6),
              ("long-range radar (5 kW)", {"pt": 5000}, {"rMax": 12000}, 8)],
    "eo": [("day camera, 2 deg zoom", {"band": 0, "det": 1, "hfov": 2, "npix": 1920, "nec": 0.005, "ee": 0.5, "kth": 5, "cmin": 0.02}, {"rMax": 6000}, 2),
           ("uncooled thermal camera, 6 deg", {"band": 1, "det": 1, "hfov": 6, "npix": 640, "netd": 0.05, "mrt": 0.1, "ee": 0.5, "kth": 5}, {"rMax": 6000}, 3),
           ("cooled thermal camera, 3 deg", {"band": 1, "det": 1, "hfov": 3, "npix": 1280, "netd": 0.02, "mrt": 0.04, "ee": 0.5, "kth": 5}, {"rMax": 10000}, 5)],
    "rf": [("RF detector", {"sens": -100}, {"rMax": 8000}, 2),
           ("sensitive RF detector", {"sens": -110}, {"rMax": 12000}, 4)],
    "rid": [("Remote ID receiver", {}, {}, 1)],
    "ac": [("small microphone array", {"nl": 35, "ag": 15, "alpha": -1, "dt": 8, "sig": 3, "gG": 1, "c0met": 0}, {"rMax": 6000}, 1),
           ("large microphone array", {"nl": 35, "ag": 30, "alpha": -1, "dt": 8, "sig": 3, "gG": 1, "c0met": 0}, {"rMax": 6000}, 2),
           ("very large microphone array", {"nl": 30, "ag": 35, "alpha": -1, "dt": 8, "sig": 3, "gG": 1, "c0met": 0}, {"rMax": 6000}, 3)],
}


def flat_range(typ, p, common, drone, alt, R=14000, cell=200):
    """Largest distance (km) out to which the mean probability of detection in a ring stays >= 0.5, flat terrain."""
    from .physics import Study, drone_values
    from .physics.scenario import Scenario
    d = R / 111320.0
    lat0, lon0 = -23.3, -45.6
    roi = [[lat0 - d, lon0 - d], [lat0 - d, lon0 + d], [lat0 + d, lon0 + d], [lat0 + d, lon0 - d]]
    s = dict({"id": 1, "type": typ, "name": "s", "lat": lat0, "lon": lon0, "deployed": True, "p": dict(p)}, **common)
    sc = {"areas": [{"id": 1, "role": "coverage", "pts": roi, "reqs": [], "threats": []}], "sensors": [s],
          "settings": dict(drone_values(drone), cell=cell, alt=alt, dem="flat", altRef="AGL")}
    st = Study(Scenario(sc), [], download=False)
    r = st.analyze(alt=alt)
    g = r["grid"]; idx = g.idx
    pd = r["agg"]["per"][0]["pd"][idx]
    cx = g.x0 + (idx % g.nx + 0.5) * g.sx; cy = g.y0 + (idx // g.nx + 0.5) * g.sy
    x0, y0 = g.pr.fwd(lat0, lon0)
    dist = np.hypot(cx - x0, cy - y0)
    best = 0.0
    for b in np.arange(0, R, cell):
        m = (dist >= b) & (dist < b + cell)
        if m.any() and pd[m].mean() >= 0.5:
            best = b + cell
    return round(best / 1000, 1)


_cache = None


def datasheet_range(typ, k, drone, alt):
    global _cache
    if _cache is None:
        _cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    key = f"{typ}|{k}|{drone}|{alt:g}"
    if key not in _cache:
        _, p, c, _ = VARIANTS[typ][k]
        _cache[key] = flat_range(typ, p, c, drone, alt)
        with open(CACHE, "w") as f:
            json.dump(_cache, f, indent=0, sort_keys=True)
    return _cache[key]
