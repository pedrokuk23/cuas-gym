"""Unit checks of the physics models against independent references.
Run from the repository root: python3 -m cuas_gym.physics.tests.test_models"""
import math
import os
import sys

import numpy as np

from .. import models as M  # noqa: E402
from ..engine import lm_of, loc_err  # noqa: E402
from ..scenario import Scenario  # noqa: E402

dB = lambda x: 10 * math.log10(x)


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


one = lambda f, *a: float(np.asarray(f(*a)).ravel()[0])


def test_radar_pd():
    assert abs(one(M.albersheim_pd, M.db2lin(13.2), 1, 1e-6) - 0.9) < 0.02
    assert abs(1e-6 ** (1 / (1 + M.db2lin(21.14))) - 0.9) < 0.005
    assert abs(one(M.shnidman_pd, np.array([M.db2lin(21.14)]), 1, 1e-6, 1) - 0.9) < 0.02
    assert abs(one(M.shnidman_pd, np.array([M.db2lin(13.2)]), 1, 1e-6, 0) - 0.9) < 0.03


def test_radar_equation():
    s = sensor("radar", pt=200, g=38, f=18, b=1, nf=5, l=6, n=1, pfa=1e-6, model=1, gamma=-99, mp=0, bwEl=0)
    lam = M.C_LIGHT / 18e9
    R = (200 * M.db2lin(38) ** 2 * lam ** 2 * 0.1 / ((4 * math.pi) ** 3 * M.K_BOLTZ * M.T0 * 1e6 * M.db2lin(5) * M.db2lin(6) * M.db2lin(21.14))) ** 0.25
    assert abs(float(M.pd_single(s, np.array([R]), True, target(rcs=0.1), None)[0]) - 0.9) < 0.005


def test_matlab_radar_network_example():
    """MathWorks "Planning Radar Network Coverage over Terrain": albersheim(0.9, 1e-6, 10) = 4.9904 dB and
    radareqpow → 3.1521e5 W for that SNR at 35 km (10 GHz, 38 dB, RCS 0.1 m², range resolution 5 m)."""
    lo, hi = -5.0, 20.0
    for _ in range(60):
        m = (lo + hi) / 2
        if one(M.albersheim_pd, M.db2lin(m), 10, 1e-6) < 0.9:
            lo = m
        else:
            hi = m
    assert abs((lo + hi) / 2 - 4.9904) < 0.01
    s = sensor("radar", pt=3.1521e5, g=38, f=10, b=M.C_LIGHT / (2 * 5) / 1e6, nf=0, l=0, n=10, pfa=1e-6, model=0, gamma=-99, mp=0, bwEl=0)
    s["pdMin"] = 0.9
    assert abs(M.range_at_pd(s, target(rcs=0.1)) / 1000 - 35) < 0.05


def test_fresnel_and_iso9613():
    lam = M.C_LIGHT / 2.4e9
    ex = {"h": np.array([-3.0]), "d1": np.array([500.0]), "d2": np.array([500.0])}
    assert abs(float(M.fresnel_r1(ex, lam)[0]) - math.sqrt(lam * 250)) < 1e-3
    s = sensor("rid", gr=0, sens=-90, lm=0, sig=0, diff=0, nlos=20, mp=0, fres=1)
    R = 1000 * 10 ** ((90 - 32.44 - 20 * math.log10(2400)) / 20) * 1.01
    assert float(M.pd_single(s, np.array([R]), True, target(), ex)[0]) == 0
    ex4 = {"h": np.array([-4.0]), "d1": np.array([500.0]), "d2": np.array([500.0])}
    assert float(M.pd_single(s, np.array([R]), True, target(), ex4)[0]) == 1
    assert abs(float(M.iso_ground(500, 0, 1.5, 1.5, 50)) + 3) < 1e-9
    ref = 14 * math.exp(-0.46 * 2.25) * (1 - math.exp(-4)) + 14 * math.exp(-0.46 * 1e4) * (1 - math.exp(-4))
    assert abs(float(M.iso_ground(500, 1, 1.5, 100, 200)) - ref) < 1e-6
    assert abs(float(M.iso_cmet(3, 1.5, 100, 2030)) - 1.5) < 1e-9


def test_propagation():
    s = sensor("rid", gr=0, sens=-90, lm=0, sig=0, diff=0, mp=0)
    Rref = 1000 * 10 ** ((20 + 90 - 32.44 - 20 * math.log10(2400)) / 20)
    assert abs(M.range_at_pd(s, target()) - Rref) < 10
    assert abs(float(M.knife_edge(0)) - 6.0) < 0.1 and abs(float(M.knife_edge(1)) - 13.9) < 0.2 and float(M.knife_edge(-0.78)) == 0
    assert abs(M.rain_db_per_km(10, 25) - 0.01217 * 25 ** 1.2571) < 0.005
    assert abs(M.rain_db_per_km(18, 25) - 0.07078 * 25 ** 1.0818) < 0.05
    ex = {"dH": np.array([3200.0]), "hs": np.array([10.0]), "ht": np.array([10.0])}
    assert abs(dB(float(M.two_ray({"mp": 1, "rho": 1, "rough": 0}, ex, 0.125)[0])) - 6.02) < 0.05


def test_eo_ac():
    s = sensor("eo", hfov=2, npix=1920, n50=1, vis=1e6, cmin=0)
    ifov = (2 * math.pi / 180) / 1920
    assert abs(float(M.pd_single(s, np.array([0.5 / (2 * ifov)]), True, target(dc=0.5, c0=1), None)[0]) - 0.5) < 1e-3
    a = sensor("ac", nl=45, ag=20, alpha=0, dt=0, sig=0)
    assert abs(M.range_at_pd(a, target(sl=80)) - 10 ** (55 / 20)) < 6


def test_literature_models():
    # Swerling 3 (one dominant scatterer, small multirotor frame): exact SNR 17.30 dB for Pd 0.9, Pfa 1e-6
    assert abs(float(M.shnidman_pd(np.array([10 ** 1.730]), 1, 1e-6, 3)[0]) - 0.9) < 0.02
    q = dict(n=1, pfa=1e-6, gamma=-99, mp=0, bwEl=0)
    R = np.array([2500.0])
    a = float(M.pd_single(sensor("radar", model=3, sw=3, **q), R, True, target(rcs=0.02), None)[0])
    b = float(M.pd_single(sensor("radar", model=0, **q), R, True, target(rcs=0.02, fluct=3), None)[0])
    assert abs(a - b) < 1e-12
    # ISO 9613-1 air absorption (dB/km)
    for f, tc, ref, tol in ((1000, 20, 5.0, 0.1), (4000, 20, 22.9, 0.4), (500, 10, 1.9, 0.1)):
        assert abs(M.air_absorb(f, tc, 70) - ref) < tol
    # point-target detection: P = 0.5 where SNR = threshold
    s = sensor("eo", band=1, det=1, hfov=6, npix=640, netd=0.05, ee=0.5, kth=5, vis=1e9)
    Rp = 0.3 * math.sqrt(6) / ((6 * math.pi / 180) / 640)
    assert abs(float(M.pd_single(s, np.array([Rp]), True, target(dc=0.3, dT=3), {"elT": np.array([5.0])})[0]) - 0.5) < 1e-3


def test_fusion_localization():
    assert abs(1 - math.exp(-(float(lm_of(0.5)) * 2)) - 0.75) < 1e-6
    sg, D = 2 * math.pi / 180, 1000
    w = 1 / (sg * sg * D * D)
    assert abs(float(loc_err(w, 0, w)) - math.sqrt(2) * sg * D) < 0.01


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("PASS", k)
