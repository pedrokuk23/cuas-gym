"""Validation of the physics module against the MathWorks example "Planning Radar Network Coverage over Terrain" (Radar Toolbox).

Rebuilds the example (five candidate radar sites near Boulder, CO; 10 GHz, 38 dB, Pd 0.9, Pfa 1e-6, 10 pulses, RCS
0.1 m2, 35 km design range; targets at 500 and 250 m above ground over the 39.5-40 N x 105.6-105.1 W window), runs
the physics module on the Copernicus GLO-30 tile N39 W106, and reports, for each altitude: coverage fraction of each site, the
three sites that a greedy selection picks (as in the example), and the union coverage of those three.

    python3 -m cuas_gym.physics.tests.boulder --tile ../dem/Copernicus_DSM_COG_10_N39_00_W106_00_DEM.tif
"""
import argparse
import json
import os

import numpy as np

from .. import Study
from ..scenario import Scenario
from ..terrain import Tile

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default=os.path.join(HERE, "data", "scenario-matlab-boulder-radar.json"))
    p.add_argument("--tile", required=True); p.add_argument("--cell", type=float, default=1000)
    p.add_argument("--out", default="results/validation_boulder.json")
    a = p.parse_args()
    sc = json.load(open(a.scenario))
    sc["settings"]["cell"] = a.cell
    sc["settings"]["dem"] = "geotiff"
    out = {}
    st = Study(Scenario(sc), [Tile(a.tile)], download=False)
    for alt in (500, 250):
        r = st.analyze(alt=alt)
        g = r["grid"]; idx = g.idx
        per = r["agg"]["per"]
        cov = [np.asarray(o["pd"][idx] >= 0.9) for o in per]       # covered = Pd >= 0.9 (SNR >= 4.99 dB in the example)
        names = [o["s"]["name"] for o in per]
        single = {n: float(c.mean()) for n, c in zip(names, cov)}
        # greedy selection of three sites by marginal coverage, as in the example
        chosen, union = [], np.zeros_like(cov[0], bool)
        for _ in range(3):
            k = max((i for i in range(len(cov)) if i not in chosen), key=lambda i: (union | cov[i]).sum())
            chosen.append(k); union = union | cov[k]
        allu = np.zeros_like(cov[0], bool)
        for c in cov:
            allu |= c
        out[alt] = {"cells": int(idx.size), "single": single, "greedy3": [names[i] for i in chosen],
                    "greedy3_coverage": float(union.mean()), "all5_coverage": float(allu.mean()),
                    "elev_min": float(np.min(g.H[idx])), "elev_max": float(np.max(g.H[idx])), "elev_mean": float(np.mean(g.H[idx]))}
        print(f"alt {alt} m AGL: {idx.size} cells; single-site coverage " + ", ".join(f"{n}: {v:.3f}" for n, v in single.items()))
        print(f"  greedy three: {[names[i] for i in chosen]} -> union {union.mean():.3f}; all five: {allu.mean():.3f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
