"""Physics module of CUAS-Gym: terrain, grid, line of sight and sensor detection models.

It computes, for a scenario (region of interest, sensors at candidate sites, threat and settings), the detection
probability of every sensor in every grid cell. CUAS-Gym uses it once per instance to build the detection matrix.

    from cuas_gym.physics import Study, Scenario, Tile
    st = Study(Scenario(scenario_dict), [Tile("Copernicus_DSM_COG_10_S24_00_W046_00_DEM.tif")])
    grid, layers = st.layers(alt=90)          # one probability layer per sensor (layers[i]["pd"][grid.idx])

Only numpy is required; itm.py additionally needs the optional itmlogic package.
"""
import os

from .engine import aggregate, make_ctx, overall, sensor_layer, zone_report
from .geo import Proj, build_grid, in_poly
from .models import DRONES, TYPES, compute_interference
from .scenario import Scenario
from .terrain import Tile, download_copernicus, find_tiles, needed_tiles

__all__ = ["Study", "Scenario", "Tile", "drone_values", "TYPES", "DRONES"]


def drone_values(key):
    """Target settings for a drone of the library (key or part of its name). None ERP = does not transmit (-300 dBm)."""
    k = str(key).lower()
    d = next((d for d in DRONES if d["key"] == k), None) or next((d for d in DRONES if k in d["name"].lower()), None)
    if d is None:
        raise KeyError(f"unknown drone {key!r}; choose one of: " + ", ".join(x["key"] for x in DRONES))
    return {"drone": d["key"], "rcs": d["rcs"], "dc": d["dc"], "sl": d["sl"], "dT": d["dT"], "fluct": d["fluct"],
            "ridPt": -300 if d["ridPt"] is None else d["ridPt"], "c2Pt": -300 if d["c2Pt"] is None else d["c2Pt"]}


class Study:
    """A scenario on terrain: builds the grid, the line-of-sight geometry and the detection layers."""

    def __init__(self, scenario, dem_files=(), download=False, log=None, **overrides):
        self.scn = scenario if isinstance(scenario, Scenario) else Scenario.load(scenario)
        self.log = log or (lambda m: None)
        files, folders = [], []
        for d in dem_files or ():
            (folders if isinstance(d, str) and os.path.isdir(d) else files).append(d)
        self.tiles = [t if isinstance(t, Tile) else Tile(t) for t in files]
        if folders:
            self.tiles += find_tiles(self.points(), folders)
        if download and not self.tiles and self.scn.settings.get("dem") != "flat":
            dest = folders[0] if folders else os.path.join(os.getcwd(), "dem")
            self.tiles += [Tile(download_copernicus(n, dest, self.log)) for n in needed_tiles(self.points())]
        self.scn.settings.update(overrides)
        self.grid = None
        self._gkey = None

    def points(self):
        return [tuple(p) for a in self.scn.coverage_areas() for p in a["pts"]] + [(s["lat"], s["lon"]) for s in self.scn.sensors]

    def ensure_grid(self):
        st = self.scn.settings
        key = (st["cell"], st["mesh"], st["sub"], st["bare"], st["losDem"], str(self.scn.obstacles))
        if self.grid is None or key != self._gkey:
            if st["dem"] == "geotiff" and not self.tiles:
                raise ValueError("this scenario uses GeoTIFF terrain: pass the tile files (dem_files)")
            self.grid = build_grid(self.scn.coverage_areas(), list(self.scn.sensors), st["cell"], st["mesh"], st["sub"], self.tiles,
                                   src="flat" if st["dem"] == "flat" else "geotiff", bare=st["bare"], obstacles=self.scn.obstacles,
                                   native=bool(st["losDem"]))
            self._gkey = key
            for s in self.scn.sensors:
                s["ground"] = None
                s["_geo"] = None
            self.scn.set_ground(self.grid, self.tiles)
        return self.grid

    def interference(self, for_opt=False):
        st = self.scn.settings
        compute_interference(self.scn.sensors, self.scn.target(), {"on": bool(st["intOn"]), "sidelobe": float(st["intSide"]),
            "duty": min(1.0, max(1e-4, float(st["intDuty"]) / 100)), "band": max(0.0, float(st["intBand"])), "rej": max(0.0, float(st["intRej"])),
            "rrej": max(0.0, float(st["intRR"])), "all": for_opt})

    def ctx(self, for_opt=False, force_alt=None):
        """force_alt: one height for every threat; None = each area's height bands."""
        g, st = self.ensure_grid(), self.scn.settings
        self.interference(for_opt)
        return g, make_ctx(g, st, self.scn.target(), self.scn.target_classes(), self.scn.areas, self.scn.threats, force_alt)

    def layers(self, sensors=None, alt=None, for_opt=False):
        """Grid and one detection layer per sensor ({"s": sensor, "pd": probability per cell, ...})."""
        g, ctx = self.ctx(for_opt, force_alt=alt)
        st = self.scn.settings
        alt = st["alt"] if alt is None else alt
        sensors = [s for s in self.scn.sensors if s.get("deployed")] if sensors is None else sensors
        return g, [sensor_layer(g, s, alt, st["altRef"], ctx) for s in sensors]

    def cost_of(self, s):
        return float(s.get("cost") or 0) + float(self.scn.settings.get("lifeYears") or 0) * float(s.get("opex") or 0)

    def analyze(self, alt=None, sensors=None):
        """Fused coverage (k-of-n aggregation), requirement report by zone and overall metrics for the deployed sensors."""
        g, per = self.layers(sensors, alt)
        agg = aggregate(g, per, int(self.scn.settings["kofn"]))
        return {"grid": g, "agg": agg, "overall": overall(g, agg, self.cost_of), "report": zone_report(g, agg, self.scn.areas), "routes": []}
