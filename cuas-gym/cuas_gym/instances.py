"""Procedural instances on real terrain.

An instance holds the region of interest, the candidate sites, the sensor inventory with costs, the budget, the target
altitude and the sector weights, plus the matrix P[c, v] of detection probabilities of configuration c in cell v
computed by the physics module (cuas_gym.physics). Instances are saved as one .npz (arrays) + one .json (metadata).
"""
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# Generating new instances needs the Copernicus GLO-30 tile S24 W046 (data/dem/, or downloaded by
# cuas_gym.physics.terrain.download_copernicus). Loading, scoring and training on the released instances do not.
from .physics import Study, drone_values  # noqa: E402
from .physics.models import TYPES  # noqa: E402
from .physics.scenario import Scenario  # noqa: E402
from .physics.terrain import Tile, sample  # noqa: E402

DEM_DIR = os.environ.get("CUAS_DEM_DIR", os.path.normpath(os.path.join(_HERE, "..", "data", "dem")))

# Terrain regions inside the Copernicus tile S24 W046 (lat -24..-23, lon -46..-45).
#   A: Paraiba valley and surrounding hills (Sao Jose dos Campos, Jacarei, Taubate): rolling terrain
#   B: Serra do Mar escarpment and the coast (Caraguatatuba, Ubatuba, Sao Sebastiao): steep terrain
REGIONS = {
    "A": {"lat": (-23.42, -23.05), "lon": (-45.95, -45.25)},
    "B": {"lat": (-23.80, -23.48), "lon": (-45.85, -45.15)},
}

# Splits of the benchmark (see the article's Section III). 'types' = sensor types the inventory is drawn from.
SPLITS = {
    "train":        {"region": "A", "types": ["radar", "eo", "rid", "rf"], "sites": (10, 18), "budget": (8, 22)},
    "test_id":      {"region": "A", "types": ["radar", "eo", "rid", "rf"], "sites": (10, 18), "budget": (8, 22)},
    "test_terrain": {"region": "B", "types": ["radar", "eo", "rid", "rf"], "sites": (10, 18), "budget": (8, 22)},
    "test_sensor":  {"region": "A", "types": ["radar", "eo", "rid", "rf", "ac"], "sites": (10, 18), "budget": (8, 22), "force": "ac"},
    "test_scale":   {"region": "A", "types": ["radar", "eo", "rid", "rf"], "sites": (28, 40), "budget": (30, 50)},
}
# disjoint seed ranges so that no test instance repeats a training instance
SEED_OFFSET = {"train": 0, "test_id": 1_000_000, "test_terrain": 2_000_000, "test_sensor": 3_000_000, "test_scale": 4_000_000}

CELL = 300.0          # grid cell (m)
ALTS = (60, 90, 120)  # target altitude AGL (m)
# Threat drones: non-cooperative types of the physics module's drone library (no Remote ID), so that each sensor type matters
DRONES = ("fpv", "fiber", "owa")


@dataclass
class Instance:
    iid: str
    split: str
    seed: int
    roi: list                 # [[lat, lon], ...] polygon of the region of interest
    sites: list               # [{"id", "lat", "lon", "x", "y", "elev", "rel", "horizon": [8 angles deg]}]
    types: list               # sensor types in the inventory
    cost: dict                # type -> cost
    rmax: dict                # type -> nominal maximum range (m)
    needlos: dict             # type -> requires line of sight
    budget: float
    alt: float
    quad_w: list              # weights of the NE, NW, SW, SE quadrants of the region of interest
    center: list = None       # [lat, lon] origin of the local frame (x east, y north)
    drone: str = "fpv"        # threat drone (key of the drone library)
    version: int = 1          # 1: default sensor parameters; 2: one variant per type, with its data-sheet range
    variant: dict = None      # v2: type -> {"k", "label"}
    drange: dict = None       # v2: type -> detection range (km) against the drone at this altitude, flat terrain
    configs: list = field(default_factory=list)   # [(site_id, type)] row order of P
    P: np.ndarray = None      # (n_configs, n_cells) detection probability
    w: np.ndarray = None      # (n_cells,) cell weights
    cell_xy: np.ndarray = None  # (n_cells, 2) cell centers, metres from the ROI center (x east, y north)

    def meta(self):
        d = {k: getattr(self, k) for k in ("iid", "split", "seed", "roi", "sites", "types", "cost", "rmax", "needlos",
                                            "budget", "alt", "quad_w", "center", "drone", "version", "variant", "drange")}
        d["configs"] = [list(c) for c in self.configs]
        return d


def _tile():
    if not os.path.isdir(DEM_DIR):
        raise RuntimeError(f"no terrain folder {DEM_DIR}: put the Copernicus GLO-30 tile there (or set CUAS_DEM_DIR), e.g. "
                           "cuas_gym.physics.terrain.download_copernicus('Copernicus_DSM_COG_10_S24_00_W046_00_DEM', DEM_DIR)")
    for f in sorted(os.listdir(DEM_DIR)):
        if f.endswith(".tif"):
            return Tile(os.path.join(DEM_DIR, f))
    raise RuntimeError(f"no GeoTIFF tile in {DEM_DIR}")


_TILE = None


def tile():
    global _TILE
    if _TILE is None:
        _TILE = _tile()
    return _TILE


class _Local:
    """Equirectangular projection around a point (metres)."""
    def __init__(self, lat0, lon0):
        self.lat0, self.lon0 = lat0, lon0
        self.my = 111320.0
        self.mx = 111320.0 * math.cos(math.radians(lat0))

    def fwd(self, lat, lon):
        return (np.asarray(lon) - self.lon0) * self.mx, (np.asarray(lat) - self.lat0) * self.my

    def inv(self, x, y):
        return self.lat0 + np.asarray(y) / self.my, self.lon0 + np.asarray(x) / self.mx


def _elev(lat, lon):
    v = sample([tile()], np.atleast_1d(np.asarray(lat, float)), np.atleast_1d(np.asarray(lon, float)))
    return np.where(np.isfinite(v), v, 0.0)


def _horizon(pr, x, y, z0, mast=10.0, R=3000.0, n=8, steps=30):
    """Horizon elevation angle (deg) in n directions from a site, terrain sampled every R/steps."""
    out = []
    d = np.linspace(R / steps, R, steps)
    for k in range(n):
        a = 2 * math.pi * k / n           # 0 = north, clockwise
        xs, ys = x + d * math.sin(a), y + d * math.cos(a)
        la, lo = pr.inv(xs, ys)
        h = _elev(la, lo)
        ang = np.degrees(np.arctan2(h - (z0 + mast), d))
        out.append(round(float(ang.max()), 1))
    return out


def _roi(rng, pr, radius):
    k = int(rng.integers(6, 9))
    th = (np.arange(k) + rng.uniform(-0.3, 0.3, k)) * 2 * math.pi / k + rng.uniform(0, 2 * math.pi)
    r = radius * rng.uniform(0.7, 1.1, k)
    xs, ys = r * np.sin(th), r * np.cos(th)
    la, lo = pr.inv(xs, ys)
    return [[round(float(a), 5), round(float(b), 5)] for a, b in zip(la, lo)], np.c_[xs, ys]


def _inside(px, py, poly):
    n, c, j = len(poly), False, len(poly) - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        if (yi > py) != (yj > py) and px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi:
            c = not c
        j = i
    return c


def generate(seed=0, split="train", compute=True, version=1):
    """Sample one instance of a split. compute=False skips the detection matrix (metadata only)."""
    cfg = SPLITS[split]
    rng = np.random.default_rng(SEED_OFFSET[split] + int(seed))
    reg = REGIONS[cfg["region"]]
    for _ in range(200):  # reject regions mostly over the sea
        lat0, lon0 = rng.uniform(*reg["lat"]), rng.uniform(*reg["lon"])
        pr = _Local(lat0, lon0)
        radius = rng.uniform(4000, 7000)
        roi, poly = _roi(rng, pr, radius)
        gx, gy = np.meshgrid(np.linspace(-radius, radius, 9), np.linspace(-radius, radius, 9))
        la, lo = pr.inv(gx.ravel(), gy.ravel())
        if (_elev(la, lo) > 2).mean() > 0.9:
            break
    # candidate sites: inside the ROI bounding box grown by 1 km, on land, at least 500 m apart
    n_sites = int(rng.integers(cfg["sites"][0], cfg["sites"][1] + 1))
    xs, ys, tries = [], [], 0
    lim = radius * 1.1 + 1000
    while len(xs) < n_sites and tries < 5000:
        tries += 1
        x, y = rng.uniform(-lim, lim, 2)
        if any(math.hypot(x - a, y - b) < 500 for a, b in zip(xs, ys)):
            continue
        la, lo = pr.inv(x, y)
        if _elev(la, lo)[0] <= 2:
            continue
        xs.append(float(x)); ys.append(float(y))
    lat_s, lon_s = pr.inv(np.array(xs), np.array(ys))
    z = _elev(lat_s, lon_s)
    roi_z = _elev(*pr.inv(gx.ravel(), gy.ravel()))
    sites = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        sites.append({"id": i + 1, "lat": round(float(lat_s[i]), 5), "lon": round(float(lon_s[i]), 5),
                      "x": round(x / 1000, 2), "y": round(y / 1000, 2), "elev": round(float(z[i])),
                      "rel": round(float(z[i] - np.median(roi_z))),
                      "horizon": _horizon(pr, x, y, float(z[i]))})
    types = list(cfg["types"])
    if "force" not in cfg:
        drop = [t for t in ("rid", "rf") if rng.random() < 0.3]   # some inventories lack one passive type
        types = [t for t in types if t not in drop]
    cost = {t: float(TYPES[t]["common"]["cost"]) for t in types}
    rmax = {t: float(TYPES[t]["common"]["rMax"]) for t in types}
    needlos = {t: bool(TYPES[t]["common"].get("needLOS", True) is not False) for t in types}
    budget = float(rng.integers(cfg["budget"][0], cfg["budget"][1] + 1))
    alt = float(rng.choice(ALTS))
    quad_w = [int(v) for v in rng.choice([1, 1, 2, 3], 4)]
    drone = str(rng.choice(DRONES))
    iid = f"{split}-{seed:06d}"
    inst = Instance(iid, split, int(seed), roi, sites, types, cost, rmax, needlos, budget, alt, quad_w, [round(float(lat0), 6), round(float(lon0), 6)], drone)
    if version == 2:
        from .variants import VARIANTS, datasheet_range
        inst.version, inst.variant, inst.drange = 2, {}, {}
        for t in types:
            k = int(rng.integers(len(VARIANTS[t])))
            if t == cfg.get("force"):          # the unseen type is drawn among its stronger variants
                k = int(rng.integers(1, len(VARIANTS[t])))
            label, _, common, c = VARIANTS[t][k]
            inst.variant[t] = {"k": k, "label": label}
            inst.cost[t] = float(c)
            inst.rmax[t] = float(common.get("rMax", inst.rmax[t]))
            inst.drange[t] = datasheet_range(t, k, drone, alt)
    inst._pr = pr
    inst._poly = poly
    if compute:
        compute_matrix(inst)
    return inst


def scenario_dict(inst):
    sensors, nid = [], 1
    for s in inst.sites:
        for t in inst.types:
            sen = {"id": nid, "type": t, "name": f"S{s['id']}-{t}", "lat": s["lat"], "lon": s["lon"], "deployed": True}
            if inst.version == 2:
                from .variants import VARIANTS
                _, p, common, _ = VARIANTS[t][inst.variant[t]["k"]]
                sen["p"] = dict(p)
                sen.update(common)
            sensors.append(sen)
            nid += 1
    return {"format": "multi-sensor-scenario", "areas": [{"id": 1, "name": "ROI", "role": "coverage", "pts": inst.roi, "reqs": [], "threats": []}],
            "sensors": sensors, "settings": dict(drone_values(inst.drone), cell=CELL, alt=inst.alt, altRef="AGL", budget=inst.budget, dem="geotiff")}


def compute_matrix(inst):
    """Detection probability of every configuration in every ROI cell, by the physics module (terrain LOS on the full-
    resolution DEM, sensor models with their default parameters)."""
    st = Study(Scenario(scenario_dict(inst)), [tile()], download=False)
    g, layers = st.layers(sensors=st.scn.sensors, alt=inst.alt)
    idx = g.idx
    inst.configs = [(int(o["s"]["name"].split("-")[0][1:]), o["s"]["type"]) for o in layers]
    inst.P = np.stack([o["pd"][idx] for o in layers]).astype(np.float32)
    cx = g.x0 + (idx % g.nx + 0.5) * g.sx
    cy = g.y0 + (idx // g.nx + 0.5) * g.sy
    la, lo = g.pr.inv(cx, cy)
    pr = _Local(*inst.center)
    x, y = pr.fwd(np.asarray(la, float), np.asarray(lo, float))
    inst.cell_xy = np.c_[x, y].astype(np.float32)
    q = np.where(y >= 0, np.where(x >= 0, 0, 1), np.where(x < 0, 2, 3))  # NE, NW, SW, SE
    inst.w = np.array(inst.quad_w, np.float32)[q]
    return inst


def save_instance(inst, folder):
    os.makedirs(folder, exist_ok=True)
    np.savez_compressed(os.path.join(folder, inst.iid + ".npz"), P=inst.P.astype(np.float16), w=inst.w, cell_xy=inst.cell_xy)
    with open(os.path.join(folder, inst.iid + ".json"), "w") as f:
        json.dump(inst.meta(), f)


def load_instance(folder, iid):
    with open(os.path.join(folder, iid + ".json")) as f:
        m = json.load(f)
    a = np.load(os.path.join(folder, iid + ".npz"))
    configs = [tuple(c) for c in m.pop("configs")]
    inst = Instance(**m)
    inst.configs = [(int(s), str(t)) for s, t in configs]
    inst.P, inst.w, inst.cell_xy = a["P"].astype(np.float32), a["w"], a["cell_xy"]
    return inst
