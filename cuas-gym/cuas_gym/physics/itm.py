"""Longley–Rice Irregular Terrain Model (ITM 1.2.2, point-to-point mode) for comparison with the radio-link model of this module.

Optional: needs the `itmlogic` package (MIT licence, pure Python: pip install itmlogic). This module's own model for Remote ID
and passive RF is free-space loss plus single knife-edge diffraction (ITU-R P.526) over the dominant obstacle of the
4/3-Earth profile; this module computes the ITM median loss over the same terrain profiles so the two can be compared.

ITM is meant for 20 MHz–20 GHz, 1–2000 km and antennas 0.5–3000 m above ground: below 1 km its result is extrapolated
(flagged in the output).
"""
import math

import numpy as np

CLIMATES = {1: "equatorial", 2: "continental subtropical", 3: "maritime subtropical", 4: "desert",
            5: "continental temperate", 6: "maritime temperate over land", 7: "maritime temperate over sea"}


def available():
    try:
        import itmlogic  # noqa: F401
        return True
    except ImportError:
        return False


def itm_loss(heights, step, h_tx, h_rx, f_mhz, climate=5, eps=15.0, sgm=0.005, en0=314.0, pol=1):
    """Median ITM basic transmission loss (dB) over a terrain profile.

    heights: ground elevation (m) from the transmitter to the receiver, equally spaced by `step` (m);
    h_tx, h_rx: antenna heights above ground (m); f_mhz: frequency; climate: ITM code 1–7; eps, sgm: ground
    permittivity and conductivity (S/m; defaults: average ground); en0: surface refractivity (N-units); pol: 0
    horizontal, 1 vertical. Returns (loss_dB, free_space_dB, warning_code)."""
    from itmlogic.preparatory_subroutines.qlrps import qlrps
    from itmlogic.preparatory_subroutines.qlrpfl import qlrpfl
    from itmlogic.statistics.avar import avar
    z = [float(v) for v in heights]
    n = len(z) - 1
    if n < 2:
        raise ValueError("profile too short")
    prop = {"fmhz": float(f_mhz), "hg": [max(0.5, float(h_tx)), max(0.5, float(h_rx))], "ipol": int(pol), "eps": eps, "sgm": sgm,
            "klimx": int(climate), "klim": int(climate), "mdvarx": 12, "mdvar": 12, "lvar": 5, "kwx": 0, "mdp": -1}
    zsys = sum(z) / len(z)
    prop["wn"], prop["gme"], prop["ens"], prop["zgnd"] = qlrps(prop["fmhz"], zsys, en0, prop["ipol"], eps, sgm)
    prop["pfl"] = [n, float(step)] + z
    prop = qlrpfl(prop)
    fs = 32.45 + 20 * math.log10(prop["fmhz"]) + 20 * math.log10(max(prop["dist"], 1.0) / 1000)
    a, prop = avar(0.0, 0.0, 0.0, prop)
    return fs + a, fs, int(prop.get("kwx", 0))


def compare(study, names=None, max_points=300, alt=None, climate=5, seed=1):
    """For each Remote ID / passive RF sensor (deployed, or `names`), sample up to max_points cells within its range and
    compare this module's path loss (free space + knife edge) with the ITM median over the same profile.
    Returns a list of dicts, one per sensor, with per-point arrays and a summary."""
    from .engine import site_geom, ground_ref
    from .geo import cell_center, hat
    from .models import C_LIGHT, diff_loss
    if not available():
        raise RuntimeError("the comparison needs the itmlogic package: python3 -m pip install itmlogic")
    g, ctx = study.ctx()
    st = study.scn.settings
    T = study.scn.target()
    alt = float(st["alt"] if alt is None else alt)
    sens = [s for s in study.scn.sensors if s["type"] in ("rid", "rf") and ((names is None and s.get("deployed")) or (names and s["name"] in names))]
    rng = np.random.default_rng(seed)
    out = []
    for s in sens:
        geo = site_geom(g, s, ctx)
        f = T["ridF"] if s["type"] == "rid" else T["c2F"]
        erp = T["ridPt"] if s["type"] == "rid" else T["c2Pt"]
        lam = C_LIGHT / (f * 1e6)
        cells = g.idx[(geo["D"][g.idx] > 50) & (geo["D"][g.idx] <= s["rMax"])]
        if len(cells) > max_points:
            cells = np.sort(rng.choice(cells, max_points, replace=False))
        G = ground_ref(g)
        rows = []
        for c in cells:
            D = float(geo["D"][c])
            zt = (float(G[c]) + alt) if st["altRef"] == "AGL" else alt
            hr = zt - float(g.H[c])
            if hr < 0.5:
                continue
            npts = max(3, int(D // max(10.0, ctx["step"])) + 1)
            t = np.linspace(0, 1, npts)
            cx, cy = cell_center(g, c)
            prof = hat(g, geo["xs"] + (cx - geo["xs"]) * t, geo["ys"] + (cy - geo["ys"]) * t)
            prof = np.where(np.isnan(prof), float(g.H[c]), prof)
            prof[0] = float(s["ground"])
            itm, fs_itm, kwx = itm_loss(prof, D / (npts - 1), s["mast"], hr, f, climate)
            dz = zt - geo["zs"] - D * D / (2 * ctx["kR"])
            R = max(1.0, math.hypot(D, dz))
            fspl = 20 * math.log10(R / 1000) + 20 * math.log10(f) + 32.44
            ts = float(geo["TS"][c])
            ex = {"h": np.array([(float(geo["ZP"][c]) - zt) * ts]), "d1": np.array([max(1.0, ts * D)]), "d2": np.array([max(1.0, (1 - ts) * D)])}
            vis = zt >= max(float(g.H[c]), float(geo["ZP"][c]))
            p = s["p"]
            if int(p.get("diff", 1)) == 1:
                obs = float(diff_loss(ex, lam)[0])
            else:
                obs = 0.0 if vis else float(p.get("nlos", 20))
            sky = fspl + obs
            budget = erp + float(p.get("gr", 0)) - float(p.get("lm", 0)) - float(p.get("sens", -90)) - (s.get("desens") or 0)
            rows.append((D, sky, itm, kwx, sky <= budget, itm <= budget))
        if not rows:
            out.append({"name": s["name"], "n": 0})
            continue
        A = np.array([r[:4] for r in rows], float)
        okS = np.array([r[4] for r in rows])
        okI = np.array([r[5] for r in rows])
        diff = A[:, 2] - A[:, 1]
        out.append({"name": s["name"], "type": s["type"], "f": f, "n": len(rows), "D": A[:, 0], "sky": A[:, 1], "itm": A[:, 2], "kwx": A[:, 3],
                    "median": float(np.median(diff)), "rms": float(np.sqrt(np.mean(diff ** 2))), "p90": float(np.percentile(np.abs(diff), 90)),
                    "agree": float(np.mean(okS == okI)), "covSky": float(np.mean(okS)), "covItm": float(np.mean(okI)),
                    "under1km": float(np.mean(A[:, 0] < 1000)), "climate": CLIMATES.get(int(climate), str(climate))})
    return out
