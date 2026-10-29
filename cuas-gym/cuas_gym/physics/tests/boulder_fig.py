"""Figure for the Boulder validation: coverage of the three greedily chosen sites at 500 and 250 m AGL.

    python3 -m cuas_gym.physics.tests.boulder_fig_fig --tile ../dem_validation/Copernicus_DSM_COG_10_N39_00_W106_00_DEM.tif --out ../paper/figures/boulder.pdf
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
    p.add_argument("--tile", required=True); p.add_argument("--cell", type=float, default=500); p.add_argument("--out", required=True)
    a = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    sc = json.load(open(a.scenario)); sc["settings"]["cell"] = a.cell; sc["settings"]["dem"] = "geotiff"
    st = Study(Scenario(sc), [Tile(a.tile)], download=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), dpi=300, sharey=True)
    for ax, alt in zip(axes, (500, 250)):
        r = st.analyze(alt=alt); g = r["grid"]; idx = g.idx; per = r["agg"]["per"]
        cov = [np.asarray(o["pd"][idx] >= 0.9) for o in per]; names = [o["s"]["name"] for o in per]
        chosen, union = [], np.zeros_like(cov[0], bool)
        for _ in range(3):
            k = max((i for i in range(len(cov)) if i not in chosen), key=lambda i: (union | cov[i]).sum())
            chosen.append(k); union |= cov[k]
        H = np.full(g.nx * g.ny, np.nan); H[idx] = g.H[idx]; H = H.reshape(g.ny, g.nx)
        C = np.full(g.nx * g.ny, np.nan); C[idx] = union; C = C.reshape(g.ny, g.nx)
        ext = [g.x0 / 1000, (g.x0 + g.nx * g.sx) / 1000, g.y0 / 1000, (g.y0 + g.ny * g.sy) / 1000]
        ls = LightSource(azdeg=315, altdeg=45)
        Hf = np.where(np.isnan(H), np.nanmean(H), H)
        ax.imshow(ls.hillshade(Hf, vert_exag=1.5, dx=g.sx, dy=g.sy), cmap="gray", extent=ext, origin="lower", vmin=0, vmax=1)
        ax.imshow(np.where(C == 1, 1.0, np.nan), cmap="Greens", vmin=0, vmax=1.6, extent=ext, origin="lower", alpha=0.55, interpolation="nearest")
        for i, o in enumerate(per):
            s = o["s"]; x, y = (float(v) / 1000 for v in g.pr.fwd(s["lat"], s["lon"]))
            sel = i in chosen
            ax.scatter(x, y, marker="^", s=46 if sel else 28, c="#ff5a1f" if sel else "white", edgecolors="black", lw=0.6, zorder=5)
            ax.annotate(str(i + 1), (x, y), xytext=(5, 2), textcoords="offset points", fontsize=7, color="black", zorder=6)
        ax.set_title(f"{alt} m above ground: sites {', '.join(str(k + 1) for k in sorted(chosen))}, {100 * union.mean():.0f}% covered", fontsize=8)
        ax.set_xlabel("km east", fontsize=7); ax.tick_params(labelsize=6); ax.set_aspect("equal")
    axes[0].set_ylabel("km north", fontsize=7)
    fig.tight_layout(); fig.savefig(a.out, bbox_inches="tight"); fig.savefig(a.out.replace(".pdf", ".png"), bbox_inches="tight")
    print("saved", a.out)


if __name__ == "__main__":
    main()
