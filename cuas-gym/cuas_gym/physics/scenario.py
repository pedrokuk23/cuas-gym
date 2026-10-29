"""Scenario description (areas, sensors, settings) and its default settings."""
import copy
import json

import numpy as np

from .models import BASE, TYPES
from .terrain import sample

# default settings; a scenario overrides them
DEFAULTS = {"cell": 100, "mesh": "intersect", "sub": 2, "dem": "geotiff", "kEarth": 1.3333, "losStep": 1, "losDem": True, "adapt": True,
            "alt": 122, "altRef": "AGL", "kofn": 2, "rcs": 0.1, "dc": 0.35, "c0": 0.3, "dT": 3, "sl": 80, "ridPt": 40, "ridF": 2400,
            "c2Pt": 25, "c2F": 2437, "rain": 0, "night": False, "task": "det", "visAtm": 0, "noiseAdd": 0, "airT": 15, "airRH": 70, "fluct": -1, "drone": "custom", "dwell": "single",
            "dwellSpeed": 15, "tobs": 30, "bare": 0, "intOn": False, "intSide": -10, "intDuty": 10, "intBand": 0.1, "intRej": 60, "intRR": 20,
            "lifeYears": 0, "budget": 40, "gMode": "ratio", "spSame": 0, "spAny": 0, "spAccess": 0, "optRoutes": True}
NUMERIC = {k for k, v in DEFAULTS.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def _num(v, d):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


class Scenario:
    def __init__(self, obj):
        if not isinstance(obj, dict) or not isinstance(obj.get("areas"), list):
            raise ValueError("not a scenario description")
        self.raw = obj
        st = dict(DEFAULTS)
        for k, v in (obj.get("settings") or {}).items():
            if k in DEFAULTS:
                st[k] = _num(v, DEFAULTS[k]) if k in NUMERIC else (v in (True, "true", 1)) if isinstance(DEFAULTS[k], bool) else v
        self.settings = st
        self.areas = [{"id": a["id"], "name": a.get("name", f"Area {a['id']}"), "role": a.get("role") or "coverage", "pts": a["pts"],
                       "reqs": [dict(r) for r in (a.get("reqs") or [])],
                       "threats": [{"id": e["id"], "min": float(e.get("min") or 0), "max": float(e["max"]) if e.get("max") not in (None, "") else float(e.get("min") or 0)}
                                   for e in (a.get("threats") or [])]} for a in obj["areas"]]
        self.threats = [dict(t) for t in (obj.get("threats") or [])]
        self.sensors = []
        nid = 1
        for s in obj.get("sensors") or []:
            self.sensors.append(self._sensor(s, nid))
            nid = max(nid, self.sensors[-1]["id"]) + 1
        self.obstacles = obj.get("obstacles") or []
        self.routes = obj.get("routes") or []
        self.classes = [k for k in (obj.get("classes") or []) if k.get("on")]

    @staticmethod
    def _sensor(s, nid):
        T = TYPES[s["type"]]
        out = copy.deepcopy(BASE)
        out.update({"id": s.get("id", nid), "type": s["type"], "name": s.get("name", T["label"]), "lat": float(s["lat"]), "lon": float(s["lon"])})
        out.update(copy.deepcopy(T["common"]))
        out.update({k: v for k, v in s.items() if k != "p"})
        if s.get("active") is not None and s.get("deployed") is None:
            out["deployed"] = bool(s["active"])
        p = dict(T["fields"]); p.update(s.get("p") or {})
        out["p"] = p
        for k in ("mast", "az", "sector", "elMin", "elMax", "rMin", "rMax", "pdMin", "cost", "opex", "revisit", "maxLooks"):
            if k in out:
                out[k] = float(out[k])
        out["ground"] = None
        out["_geo"] = None
        return out

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def coverage_areas(self):
        return [a for a in self.areas if a["role"] == "coverage" and len(a["pts"]) >= 3]

    def target(self):
        st = self.settings
        tobs = st["cell"] / max(0.1, st["dwellSpeed"]) if st["dwell"] == "cross" else max(0, st["tobs"]) if st["dwell"] == "hover" else 0
        return {"rcs": st["rcs"], "dc": st["dc"], "c0": st["c0"], "dT": st["dT"], "ridPt": st["ridPt"], "ridF": st["ridF"], "c2Pt": st["c2Pt"],
                "c2F": st["c2F"], "sl": st["sl"], "rain": st["rain"] or 0, "night": bool(st["night"]), "task": st["task"],
                "visAtm": st["visAtm"] or 0, "noiseAdd": st["noiseAdd"] or 0, "tobs": tobs,
                "fluct": st["fluct"], "airT": st["airT"], "airRH": st["airRH"]}

    def target_classes(self):
        base = self.target()
        out = []
        for k in self.classes:
            T = dict(base, rcs=float(k["rcs"]), dc=float(k["dc"]), sl=float(k["sl"]),
                     ridPt=float(k.get("ridPt", base["ridPt"])) if k.get("rid") else -300,
                     c2Pt=float(k.get("c2Pt", base["c2Pt"])) if k.get("link") else -300,
                     dT=base["dT"] if k.get("dT") in (None, "") else float(k["dT"]),
                     fluct=base["fluct"] if k.get("fluct") in (None, "") else float(k["fluct"]))
            out.append({"name": k.get("name"), "alt": None if k.get("alt") in ("", None) else float(k["alt"]), "T": T})
        return out

    def set_ground(self, g, tiles):
        """Ground under each sensor: surveyed value, else terrain, lowered to bare earth when that option is on."""
        R = float(self.settings.get("bare") or 0)
        for s in self.sensors:
            if s.get("ground") is not None:
                continue
            gs = s.get("gset")
            if gs not in ("", None):
                s["ground"] = float(gs)
                continue
            v = sample(tiles, np.array([s["lat"]]), np.array([s["lon"]]))[0] if tiles else np.nan
            s["ground"] = float(v) if np.isfinite(v) else 0.0
            if R > 0:
                s["ground"] = min(s["ground"], bare_at(s["lat"], s["lon"], R, g, tiles))


def bare_at(lat, lon, R, g, tiles):
    from .geo import Proj, cell_of
    pr = Proj(lat, lon)
    m = np.inf
    for a in range(-3, 4):
        for b in range(-3, 4):
            if a * a + b * b > 9:
                continue
            la, lo = pr.inv(a * R / 3, b * R / 3)
            v = sample(tiles, np.array([la]), np.array([lo]))[0] if tiles else np.nan
            if not np.isfinite(v) and g is not None:
                c = cell_of(g, float(la), float(lo))
                if c >= 0:
                    v = (g.G if g.G is not None else g.H)[c]
            if np.isfinite(v) and v < m:
                m = float(v)
    return m
