"""Figure: one CUAS-Gym instance with terrain, candidate sites and the fused detection probability of three
placements (reference, trained model with the named inventory, trained model with the anonymized inventory).

    python3 -m cuas_rlvr.figure_instance --data data/gym2 --iid test_sensor-000000 \
        --named results/v2/mlx_v2_q7b_grpo_s0_test_sensor.jsonl --anon results/v2/mlx_v2anon_q7b_grpo_s0_test_sensor.jsonl \
        --out ../paper/figures/instance_example.pdf
"""
import argparse
import json

import numpy as np
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType, not Type 3 (IEEE PDF requirements)

from cuas_gym.cli import load
from cuas_gym.instances import _Local, _elev
from cuas_gym.prompt import anon_map

MARK = {"radar": ("s", "radar"), "eo": ("^", "EO/IR"), "rf": ("o", "passive RF"), "ac": ("D", "acoustic"), "rid": ("v", "Remote ID")}


def fused(v, rows):
    q = np.ones(v.P.shape[1])
    for r in rows:
        q *= 1 - v.P[r]
    return 1 - q


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/gym2"); p.add_argument("--iid", required=True)
    p.add_argument("--named", required=True); p.add_argument("--anon", required=True); p.add_argument("--out", required=True)
    a = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    inst, v = load(a.data, a.iid)
    rec_n = next(json.loads(l) for l in open(a.named) if json.loads(l)["iid"] == a.iid)
    rec_a = next(json.loads(l) for l in open(a.anon) if json.loads(l)["iid"] == a.iid)
    inv = {lab: t for t, lab in anon_map(inst).items()}
    act_a = [(s, inv.get(str(t).lower(), str(t).lower())) for s, t in rec_a["action"]]
    ref_rows = v.reference()["rows"]; Fref = v.F(ref_rows)
    panels = [("Reference (branch and bound)", ref_rows),
              ("Trained 7B, names inventory", v.rows_of(rec_n["action"])[0]),
              ("Trained 7B, properties inventory", v.rows_of(act_a)[0])]
    # terrain hillshade on a grid around the ROI
    loc = _Local(*inst.center)
    xy = inst.cell_xy / 1000.0
    pad = 1.5
    x0, x1 = xy[:, 0].min() - pad, xy[:, 0].max() + pad; y0, y1 = xy[:, 1].min() - pad, xy[:, 1].max() + pad
    gx = np.linspace(x0, x1, 220); gy = np.linspace(y0, y1, 220)
    GX, GY = np.meshgrid(gx, gy)
    lat, lon = loc.inv(GX.ravel() * 1000, GY.ravel() * 1000)
    Z = _elev(lat, lon).reshape(GX.shape)
    ls = LightSource(azdeg=315, altdeg=45)
    shade = ls.hillshade(Z, vert_exag=2, dx=(gx[1] - gx[0]) * 1000, dy=(gy[1] - gy[0]) * 1000)
    roi = np.array([loc.fwd(la, lo) for la, lo in inst.roi]) / 1000.0
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), dpi=300, sharex=True, sharey=True)
    site = {s["id"]: s for s in inst.sites}
    for ax, (title, rows) in zip(axes, panels):
        ax.imshow(shade, cmap="gray", extent=[x0, x1, y0, y1], origin="lower", vmin=0, vmax=1, alpha=0.9)
        cov = fused(v, rows)
        # coverage as an image on the cell grid (cells are a regular grid; cells outside the ROI stay transparent)
        cell = 0.3
        ix = np.round((xy[:, 0] - xy[:, 0].min()) / cell).astype(int); iy = np.round((xy[:, 1] - xy[:, 1].min()) / cell).astype(int)
        img = np.full((iy.max() + 1, ix.max() + 1), np.nan); img[iy, ix] = cov
        ext = [xy[:, 0].min() - cell / 2, xy[:, 0].max() + cell / 2, xy[:, 1].min() - cell / 2, xy[:, 1].max() + cell / 2]
        sc = ax.imshow(img, cmap="viridis", vmin=0, vmax=1, extent=ext, origin="lower", alpha=0.8, interpolation="nearest")
        ax.plot(np.r_[roi[:, 0], roi[0, 0]], np.r_[roi[:, 1], roi[0, 1]], color="white", lw=0.9)
        ax.scatter([s["x"] for s in inst.sites], [s["y"] for s in inst.sites], facecolors="white", edgecolors="black", s=12, lw=0.5, zorder=4)
        used = {}
        for r in rows:
            sid, t = inst.configs[r]
            used.setdefault(sid, []).append(t)
        for sid, ts in used.items():
            s = site[sid]
            for k, t in enumerate(ts):
                m, _ = MARK[t]
                ax.scatter(s["x"] + 0.25 * k, s["y"], marker=m, s=34, c="#ff5a1f", edgecolors="black", lw=0.5, zorder=5)
        F = v.F(rows)
        ax.set_title(f"{title}\n$F/F^\\star$ = {F / Fref:.2f}", fontsize=7)
        ax.set_xlabel("km east", fontsize=7); ax.tick_params(labelsize=6)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    axes[0].set_ylabel("km north", fontsize=7)
    cb = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.01); cb.set_label("fused probability of detection", fontsize=7); cb.ax.tick_params(labelsize=6)
    hs = [plt.Line2D([], [], marker=MARK[t][0], color="#ff5a1f", markeredgecolor="black", lw=0, markersize=5, label=MARK[t][1]) for t in inst.types if t in MARK]
    hs.append(plt.Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor="black", lw=0, markersize=4, label="candidate site"))
    fig.legend(handles=hs, loc="lower center", ncol=len(hs), fontsize=6, frameon=False, bbox_to_anchor=(0.45, -0.02))
    fig.subplots_adjust(left=0.07, right=0.9, top=0.86, bottom=0.2, wspace=0.08)
    fig.savefig(a.out, bbox_inches="tight"); fig.savefig(a.out.replace(".pdf", ".png"), bbox_inches="tight")
    print("drone", inst.drone, "alt", inst.alt, "budget", inst.budget, "types", inst.types, "variants", {t: inst.variant[t]["label"] for t in inst.types})
    for title, rows in panels:
        print(title, [(inst.configs[r]) for r in rows], round(v.F(rows) / Fref, 3))


if __name__ == "__main__":
    main()
