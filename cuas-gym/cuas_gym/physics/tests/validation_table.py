"""Checks of the physics module against independent references (Table S2 of the supplementary material).

    python3 -m cuas_gym.physics.tests.validation_table [--latex out.tex]

Each row: model, check, reference value, value returned by this module, source of the reference.
"""
import argparse
import math

import numpy as np

from .. import Study
from .. import models as M
from ..engine import loc_err, site_geom
from ..geo import Proj
from ..scenario import Scenario


def one(f, *a):
    return float(np.atleast_1d(f(*a))[0])


def marcum_pd(snr_db, pfa):
    """Exact non-fluctuating Pd, 1 pulse: Marcum Q1(sqrt(2 SNR), sqrt(-2 ln Pfa)) (needs scipy)."""
    try:
        from scipy.stats import ncx2
    except ImportError:
        return float("nan")
    snr = 10 ** (snr_db / 10)
    return float(ncx2.sf(-2 * math.log(pfa), 2, 2 * snr))


def swerling3_pd(snr_db, pfa, n=200000):
    """Swerling 3, 1 pulse, by numerical integration of the non-fluctuating Pd over the chi-square(4) law of the RCS."""
    from scipy.stats import ncx2
    snr = 10 ** (snr_db / 10)
    x = np.linspace(1e-6, 30, n)            # RCS / mean RCS
    pdf = 4 * x * np.exp(-2 * x)            # chi-square with 4 degrees of freedom, unit mean
    pd = ncx2.sf(-2 * math.log(pfa), 2, 2 * snr * x)
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(pd * pdf, x))


def sensor(typ, **p):
    s = Scenario._sensor({"type": typ, "lat": 0, "lon": 0, "p": p}, 1)
    s["rMax"] = 1e7
    s["pdMin"] = 0.5
    return s


def target(**o):
    T = {"rcs": 1, "dc": 0.5, "c0": 0.5, "dT": 3, "sl": 80, "ridPt": 20, "ridF": 2400, "c2Pt": 20, "c2F": 2400, "rain": 0, "night": False,
         "task": "det", "visAtm": 0, "noiseAdd": 0, "tobs": 0}
    T.update(o)
    return T


def shadow_height(D_km, mast, ridge=None, cell=20.0, k=4 / 3):
    """Lowest visible target height (m above flat ground) at D_km north of a site, from the engine's line-of-sight
    computation (site_geom) on flat terrain with an effective earth radius k x 6371 km, optionally behind a ridge
    (dict: distance in km, height and width in m)."""
    lat0, lon0 = 0.0, 0.0
    pr = Proj(lat0, lon0)
    ll = lambda x, y: [float(v) for v in pr.inv(x, y)]
    D, w = D_km * 1000, cell * 1.5
    roi = [ll(-w, D - w), ll(w, D - w), ll(w, D + w), ll(-w, D + w)]
    obs = []
    if ridge:
        a, b = ridge["km"] * 1000 - ridge["w"] / 2, ridge["km"] * 1000 + ridge["w"] / 2
        obs = [{"pts": [ll(-500, a), ll(500, a), ll(500, b), ll(-500, b)], "h": ridge["h"]}]
    sc = {"areas": [{"id": 1, "role": "coverage", "pts": roi, "reqs": [], "threats": []}],
          "sensors": [{"id": 1, "type": "radar", "name": "s", "lat": lat0, "lon": lon0, "deployed": True, "mast": mast, "p": {}}],
          "settings": {"cell": cell, "dem": "flat", "alt": 0, "altRef": "AGL", "kEarth": k}, "obstacles": obs}
    st = Study(Scenario(sc), [])
    g, ctx = st.ctx(False, force_alt=0)
    G = site_geom(g, st.scn.sensors[0], ctx)
    k = int(np.argmin(np.abs(G["D"][g.idx] - D)))
    return float(G["ZP"][g.idx[k]]), float(G["D"][g.idx[k]])


def rows():
    R = []
    R.append(("Radar detection", "Marcum (non-fluctuating): $P_d$ at 13.2~dB, $P_{fa}=10^{-6}$, 1 pulse",
              f"{marcum_pd(13.2, 1e-6):.3f}", f"{one(M.albersheim_pd, M.db2lin(13.2), 1, 1e-6):.3f}",
              "exact Marcum $Q$ (numerical); module via Albersheim's approximation \\cite{albersheim1981closed}"))
    R.append(("", "Swerling~1: $P_d$ at 21.14~dB", f"{1e-6 ** (1 / (1 + M.db2lin(21.14))):.3f}",
              f"{one(M.shnidman_pd, np.array([M.db2lin(21.14)]), 1, 1e-6, 1):.3f}", "exact closed form $P_{fa}^{1/(1+\\mathrm{SNR})}$; module via Shnidman"))
    R.append(("", "Swerling~3: $P_d$ at 17.30~dB", f"{swerling3_pd(17.30, 1e-6):.3f}",
              f"{one(M.shnidman_pd, np.array([10 ** 1.730]), 1, 1e-6, 3):.3f}", "numerical integration of the $\\chi^2_4$ law; module via Shnidman"))
    lo, hi = -5.0, 20.0
    for _ in range(60):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if one(M.albersheim_pd, M.db2lin(m), 10, 1e-6) < 0.9 else (lo, m)
    R.append(("", "Albersheim: SNR for $P_d=0.9$, $P_{fa}=10^{-6}$, 10 pulses", "4.9904~dB", f"{(lo + hi) / 2:.4f}~dB",
              "\\cite{albersheim1981closed,mathworks2024radar}"))
    s = sensor("radar", pt=3.1521e5, g=38, f=10, b=M.C_LIGHT / (2 * 5) / 1e6, nf=0, l=0, n=10, pfa=1e-6, model=0, gamma=-99, mp=0, bwEl=0)
    s["pdMin"] = 0.9
    R.append(("", "Radar equation: range at $P_d=0.9$ with $P_t = 3.1521\\times10^{5}$~W", "35.0~km",
              f"{M.range_at_pd(s, target(rcs=0.1)) / 1000:.1f}~km", "\\cite{mathworks2024radar}"))
    R.append(("Propagation", "Knife-edge loss, $\\nu=0$", "6.0~dB", f"{one(M.knife_edge, np.array([0.0])):.2f}~dB", "\\cite{itu526}"))
    R.append(("", "Knife-edge loss, $\\nu=1$", "13.9~dB", f"{one(M.knife_edge, np.array([1.0])):.2f}~dB", "\\cite{itu526}"))
    R.append(("", "Rain, 10~GHz, 25~mm/h", "0.696~dB/km", f"{M.rain_db_per_km(10, 25):.3f}~dB/km", "$k$, $\\alpha$ of \\cite{itu838}"))
    R.append(("", "Rain, 18~GHz, 25~mm/h (interpolated coefficients)", "2.30~dB/km", f"{M.rain_db_per_km(18, 25):.2f}~dB/km", "\\cite{itu838}"))
    R.append(("Acoustic", "Air absorption, 1~kHz, 20$^\\circ$C, 70\\% RH", "5.0~dB/km", f"{M.air_absorb(1000, 20, 70):.2f}~dB/km", "\\cite{iso9613}"))
    R.append(("", "Air absorption, 4~kHz, 20$^\\circ$C, 70\\% RH", "22.9~dB/km", f"{M.air_absorb(4000, 20, 70):.2f}~dB/km", "\\cite{iso9613}"))
    R.append(("", "Air absorption, 500~Hz, 10$^\\circ$C, 70\\% RH", "1.9~dB/km", f"{M.air_absorb(500, 10, 70):.2f}~dB/km", "\\cite{iso9613}"))
    kR = 4 / 3 * 6371e3
    ref = (60e3 - math.sqrt(2 * kR * 100)) ** 2 / (2 * kR)
    R.append(("Line of sight", "Smooth earth ($k=4/3$), 100~m mast: lowest visible height at 60~km", f"{ref:.1f}~m",
              f"{shadow_height(60, 100)[0]:.1f}~m", "$(D-\\sqrt{2kRh})^2/2kR$"))
    ref2 = 10 + (50 - 10) * 6.01 / 1.99
    zr, _ = shadow_height(6.01, 10, {"km": 1.99, "h": 50, "w": 20}, cell=8.0, k=1e9)   # flat earth, 20-m-wide ridge
    R.append(("", "50~m ridge at 1.99~km, 10~m mast, flat earth: shadow height at 6.01~km", f"{ref2:.1f}~m",
              f"{zr:.1f}~m", "similar triangles"))
    sg, D = 2 * math.pi / 180, 1000
    w = 1 / (sg * sg * D * D)
    R.append(("Localization", "Two bearings at $90^\\circ$, $\\sigma=2^\\circ$, 1~km", f"{math.sqrt(2) * sg * D:.1f}~m", f"{float(loc_err(w, 0, w)):.1f}~m", "$\\sqrt{2}\\,\\sigma D$"))
    return R


def main():
    p = argparse.ArgumentParser(); p.add_argument("--latex", default=None)
    a = p.parse_args()
    R = rows()
    for r in R:
        print(" | ".join(r[:4]))
    if a.latex:
        with open(a.latex, "w") as f:
            for r in R:
                f.write(" & ".join(r) + " \\\\\n")


if __name__ == "__main__":
    main()
