"""Figure: fused detection probability of three networks on one CUAS-Gym instance, with the selected sensors
on the map (site numbers) and the network, quality ratio, and verified cost of each panel.

    # Fig. 1: unseen-sensor instance, reference and the two trained 7B planners
    python3 -m cuas_rlvr.figure_map_rationale --data data/gym2 --iid test_sensor-000000 \
        --panel "Reference=ref" \
        --panel "7B, names=named:results/v2/mlx_v2_q7b_grpo_s0_test_sensor.jsonl" \
        --panel "7B, properties=anon:results/v2/mlx_v2anon_q7b_grpo_s0_test_sensor.jsonl" \
        --out paper_artifacts/figures/map_7b.pdf
    # Fig. 2: unseen-terrain instance, the two trained 27B planners and the completed geometric heuristic
    python3 -m cuas_rlvr.figure_map_rationale --data data/gym2 --iid test_terrain-000002 \
        --panel "27B, names=named:results/v2/mlx_q38_27b_grpo_s0_test_terrain.jsonl" \
        --panel "27B, properties=anon:results/v2/mlx_q38_27banon_grpo_s0_test_terrain.jsonl" \
        --panel "Geometric, completed=geo" --out paper_artifacts/figures/map_27b.pdf

The map layers are opaque RGB images (terrain shading blended with the colour map in numpy), so the PDF carries no
transparency masks; text and markers stay vector, with embedded TrueType fonts.
"""
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource, Normalize

try:
    from cuas_gym.cli import load
    from cuas_gym.instances import _Local, _elev
    from cuas_gym.prompt import anon_map
    from cuas_gym.baselines import geometric_complete
except ImportError:  # same modules under a relative package layout
    from .cli import load
    from .instances import _Local, _elev
    from .prompt import anon_map
    from .baselines import geometric_complete

NAME = {"radar": "Radar", "eo": "EO/IR", "rf": "RF", "rid": "Remote ID", "ac": "Acoustic"}
MARK = {"radar": "s", "eo": "^", "rf": "o", "rid": "v", "ac": "D"}
COL = {"radar": "#CC79A7", "eo": "#0072B2", "rf": "#009E73", "rid": "#E69F00", "ac": "#D55E00"}
INK, RULE = "#1f2328", "#c0c4c8"
SITE = "#E8000B"  # candidate sites: red, distinct from the grey terrain, the colour map, and the sensor markers


def fused(v, rows):
    q = np.ones(v.P.shape[1])
    for r in rows:
        q *= 1 - v.P[r]
    return 1 - q


def panel_rows(inst, v, iid, spec):
    if spec == "ref":
        return v.reference()["rows"]
    if spec == "geo":
        return list(geometric_complete(v))
    kind, path = spec.split(":", 1)
    rec = next(json.loads(l) for l in open(path) if json.loads(l)["iid"] == iid)
    act = rec["action"]
    if kind == "anon":
        inv = {lab: t for t, lab in anon_map(inst).items()}
        act = [(s, inv.get(str(t).lower(), str(t).lower())) for s, t in act]
    return v.rows_of(act)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/gym2"); p.add_argument("--iid", required=True)
    p.add_argument("--panel", action="append", required=True, help='"Title=ref|geo|named:path|anon:path" (three panels)')
    p.add_argument("--out", required=True); p.add_argument("--pad", type=float, default=1.3)
    a = p.parse_args()
    inst, v = load(a.data, a.iid)
    Fref = v.F(v.reference()["rows"])
    panels = [(t, panel_rows(inst, v, a.iid, s)) for t, s in (x.split("=", 1) for x in a.panel)]
    site = {s["id"]: s for s in inst.sites}
    loc = _Local(*inst.center)
    xy = inst.cell_xy / 1000.0
    roi = np.array([loc.fwd(la, lo) for la, lo in inst.roi]) / 1000.0
    x0, x1 = roi[:, 0].min() - a.pad, roi[:, 0].max() + a.pad
    y0, y1 = roi[:, 1].min() - a.pad, roi[:, 1].max() + a.pad
    n_shown = sum(x0 <= s["x"] <= x1 and y0 <= s["y"] <= y1 for s in inst.sites)
    gx = np.linspace(x0, x1, 600); gy = np.linspace(y0, y1, int(600 * (y1 - y0) / (x1 - x0)))
    GX, GY = np.meshgrid(gx, gy)
    lat, lon = loc.inv(GX.ravel() * 1000, GY.ravel() * 1000)
    Z = _elev(lat, lon).reshape(GX.shape)
    shade = LightSource(azdeg=315, altdeg=45).hillshade(Z, vert_exag=2, dx=(gx[1] - gx[0]) * 1000, dy=(gy[1] - gy[0]) * 1000)
    base = np.dstack([0.25 + 0.7 * shade] * 3)
    cell = 0.3
    cx0, cy0 = xy[:, 0].min() - cell / 2, xy[:, 1].min() - cell / 2
    ix = np.floor((GX - cx0) / cell).astype(int); iy = np.floor((GY - cy0) / cell).astype(int)
    ixc = np.round((xy[:, 0] - xy[:, 0].min()) / cell).astype(int); iyc = np.round((xy[:, 1] - xy[:, 1].min()) / cell).astype(int)
    cmap, norm = plt.get_cmap("viridis"), Normalize(0, 1)

    def layer(rows):
        g = np.full((iyc.max() + 1, ixc.max() + 1), np.nan); g[iyc, ixc] = fused(v, rows)
        ok = (ix >= 0) & (iy >= 0) & (ix < g.shape[1]) & (iy < g.shape[0])
        val = np.full(GX.shape, np.nan); val[ok] = g[iy[ok], ix[ok]]
        rgb = base.copy(); m = np.isfinite(val)
        rgb[m] = 0.85 * cmap(norm(val[m]))[:, :3] + 0.15 * base[m]
        return np.clip(rgb, 0, 1)

    W, H = 7.16, 3.1
    fy = lambda inch: inch / H  # vertical positions in inches from the bottom
    fig = plt.figure(figsize=(W, H), dpi=300)
    L, R, gap = 0.065, 0.985, 0.045
    pw = (R - L - 2 * gap) / 3
    map_bot, map_h = fy(0.66), fy(1.84)
    used_types, out = set(), []
    for k, (title, rows) in enumerate(panels):
        used = [inst.configs[r] for r in rows]
        cost = sum(inst.cost[t] for _, t in used)
        qr = v.F(rows) / Fref
        out.append((title, used, round(qr, 3), cost))
        left = L + k * (pw + gap)
        tx = left - (0.045 if k == 0 else 0.01)
        fig.text(tx, fy(3.09), f"{'ABC'[k]}  {title}", fontsize=8.4, fontweight="bold", color=INK, va="top")
        fig.text(tx, fy(2.89), f"QR {qr:.3f}", fontsize=8.2, fontweight="bold", color=INK, va="top")
        fig.text(left + pw, fy(2.89), f"Cost {cost:g}/{inst.budget:g}", fontsize=7.2, color=INK, va="top", ha="right")
        ax = fig.add_axes([left, map_bot, pw, map_h])
        ax.imshow(layer(rows), extent=[x0, x1, y0, y1], origin="lower", interpolation="bilinear")
        ax.plot(np.r_[roi[:, 0], roi[0, 0]], np.r_[roi[:, 1], roi[0, 1]], color="white", lw=0.9)
        ax.scatter([s["x"] for s in inst.sites], [s["y"] for s in inst.sites], s=14, facecolors=SITE, edgecolors="white", lw=0.5, zorder=4)
        bysite = {}
        for sid, t in used:
            bysite.setdefault(sid, []).append(t); used_types.add(t)
        for sid, ts in bysite.items():
            s = site[sid]
            for j, t in enumerate(ts):
                ax.scatter(s["x"] + 0.35 * j, s["y"], marker=MARK[t], s=46, c=COL[t], edgecolors=INK, lw=0.6, zorder=6)
            # label to the right, or to the left when another selected site sits close to the right
            crowd = any(o != sid and 0 < site[o]["x"] - s["x"] < 2.0 and abs(site[o]["y"] - s["y"]) < 0.8 for o in bysite)
            ax.annotate(str(sid), (s["x"] + (0 if crowd else 0.35 * (len(ts) - 1)), s["y"]), xytext=(-5 if crowd else 5, 1),
                        textcoords="offset points", fontsize=7.3, fontweight="bold", color=INK, zorder=7, va="center",
                        ha="right" if crowd else "left",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("auto")
        ax.tick_params(labelsize=7.3, length=2, pad=1.5)
        ax.set_xlabel("East (km)", fontsize=7.6, labelpad=1)
        if k == 0:
            ax.set_ylabel("North (km)", fontsize=7.6, labelpad=1)
        else:
            ax.set_yticklabels([])
        for sp in ax.spines.values():
            sp.set_color(INK); sp.set_linewidth(0.6)
        # network below the map: type at sites
        groups = {}
        for sid, t in used:
            groups.setdefault(t, []).append(sid)
        txt = "; ".join(f"{NAME[t]} at {', '.join(str(x) for x in sorted(s))}" for t, s in groups.items())
        fig.text(tx, fy(2.69), txt, fontsize=7.5, color=INK, va="top")
    order = ["radar", "eo", "rf", "rid", "ac"]
    hs = [plt.Line2D([], [], marker=MARK[t], color=COL[t], markeredgecolor=INK, lw=0, markersize=5.5, label=NAME[t])
          for t in order if t in used_types]
    hs.append(plt.Line2D([], [], marker="o", markerfacecolor=SITE, markeredgecolor="white", lw=0, markersize=4,
                         label=f"Candidate site ({n_shown} of {len(inst.sites)} shown)"))
    hs.append(plt.Line2D([], [], color="#9aa0a6", lw=1.4, label="Region of interest"))
    fig.legend(handles=hs, loc="center left", ncol=len(hs), fontsize=7.3, frameon=False, bbox_to_anchor=(0.005, fy(0.225)),
               handletextpad=0.3, columnspacing=0.8)
    fig.text(0.65, fy(0.285), "Fused detection probability", fontsize=7.3, color=INK, va="bottom")
    cax = fig.add_axes([0.65, fy(0.19), 0.335, fy(0.075)])
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=7.2, length=2, pad=1); cb.outline.set_linewidth(0.5)
    fig.savefig(a.out); fig.savefig(a.out.replace(".pdf", ".png"), dpi=600)
    for o in out:
        print(o)
    print("drone", inst.drone, "alt", inst.alt, "budget", inst.budget, "cost", inst.cost, "range", inst.drange,
          "los", inst.needlos, "quad_w", inst.quad_w, "shown", n_shown, "proven", v.reference().get("proven"))


if __name__ == "__main__":
    main()
