"""Boulder validation with the Longley-Rice (ITM) path loss, as the MathWorks example uses.

For each cell and radar site, the ITM median loss over the terrain profile (itmlogic package) replaces free space:
the one-way excess loss over free space is counted twice (two-way radar path), and the cell is covered when the SNR at
the required 4.9904 dB still holds. Compared with the module's own line-of-sight + free-space coverage from
validate_boulder.py, this isolates the effect of the propagation model.

    pip install itmlogic
    python3 -m cuas_gym.physics.tests.boulder_itm_itm --tile ../dem_validation/Copernicus_DSM_COG_10_N39_00_W106_00_DEM.tif
"""
import argparse
import json
import math
import os

import numpy as np

from .. import Study
from ..scenario import Scenario
from ..terrain import Tile
from ...engine import site_geom, ground_ref
from ...geo import cell_center, hat
from ..itm import itm_loss

HERE = os.path.dirname(os.path.abspath(__file__))
SNR_REQ_DB = 4.9904


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default=os.path.join(HERE, "data", "scenario-matlab-boulder-radar.json"))
    p.add_argument("--tile", required=True); p.add_argument("--cell", type=float, default=1000)
    p.add_argument("--climate", type=int, default=5); p.add_argument("--out", default="results/validation_boulder_itm.json")
    a = p.parse_args()
    sc = json.load(open(a.scenario)); sc["settings"]["cell"] = a.cell; sc["settings"]["dem"] = "geotiff"
    st = Study(Scenario(sc), [Tile(a.tile)], download=False)
    out = {}
    for alt in (500, 250):
        r = st.analyze(alt=alt)
        g, ctx = st.ctx(False, force_alt=alt)
        G = ground_ref(g); idx = g.idx
        per = r["agg"]["per"]
        names = [o["s"]["name"] for o in per]
        cov_sky, cov_itm, excess = [], [], []
        for o in per:
            s = o["s"]; geo = site_geom(g, s, ctx)
            pd = np.asarray(o["pd"][idx]); sky = pd >= 0.9
            f_mhz = float(s["p"]["f"]) * 1000.0
            itm_ok = np.zeros(len(idx), bool); ex = np.full(len(idx), np.nan)
            # SNR margin at the cell with free space: from Pd back to SNR is not exposed, so use the radar equation directly
            lam = 299792458.0 / (f_mhz * 1e6)
            pt, gdb, rcs = float(s["p"]["pt"]), float(s["p"]["g"]), float(sc["settings"]["rcs"])
            bw = float(s["p"]["b"]) * 1e6; n_p = int(s["p"]["n"])
            k, T0 = 1.380649e-23, 290.0
            for j, c in enumerate(idx):
                D = float(geo["D"][c])
                if D < 50 or D > float(s["rMax"]):
                    continue
                zt = float(G[c]) + alt
                hr = zt - float(g.H[c])
                dz = zt - geo["zs"] - D * D / (2 * ctx["kR"])
                R = max(1.0, math.hypot(D, dz))
                snr1 = pt * (10 ** (gdb / 10)) ** 2 * lam ** 2 * rcs / ((4 * math.pi) ** 3 * k * T0 * bw * R ** 4)
                snr_db = 10 * math.log10(snr1)   # single-pulse SNR: the example sizes Pt so that it equals 4.9904 dB at 35 km (10 pulses enter through Albersheim)
                npts = max(3, int(D // max(10.0, ctx["step"])) + 1)
                t = np.linspace(0, 1, npts); cx, cy = cell_center(g, c)
                prof = hat(g, geo["xs"] + (cx - geo["xs"]) * t, geo["ys"] + (cy - geo["ys"]) * t)
                prof = np.where(np.isnan(prof), float(g.H[c]), prof); prof[0] = float(s["ground"])
                try:
                    L, Lfs, _ = itm_loss(prof, D / (npts - 1), s["mast"], max(0.5, hr), f_mhz, a.climate)
                except Exception:
                    continue
                e = max(0.0, L - Lfs); ex[j] = e
                itm_ok[j] = (snr_db - 2 * e) >= SNR_REQ_DB
            cov_sky.append(sky); cov_itm.append(itm_ok); excess.append(ex)
            print(f"alt {alt}: {s['name']}: LOS+free space {sky.mean():.3f}  ITM {itm_ok.mean():.3f}  median excess loss {np.nanmedian(ex):.1f} dB", flush=True)
        def greedy(cov):
            chosen, union = [], np.zeros_like(cov[0], bool)
            for _ in range(3):
                kk = max((i for i in range(len(cov)) if i not in chosen), key=lambda i: (union | cov[i]).sum())
                chosen.append(kk); union = union | cov[kk]
            return [names[i] for i in chosen], float(union.mean())
        gs, us = greedy(cov_sky); gi, ui = greedy(cov_itm)
        out[alt] = {"sky": {"greedy3": gs, "coverage": us}, "itm": {"greedy3": gi, "coverage": ui},
                    "single_itm": {n: float(c.mean()) for n, c in zip(names, cov_itm)}}
        print(f"alt {alt}: greedy three LOS+FS {gs} -> {us:.3f}; ITM {gi} -> {ui:.3f}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
