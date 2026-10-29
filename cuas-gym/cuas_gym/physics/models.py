"""Sensor detection models, vectorized with NumPy.

Every function takes arrays of slant range R (m), line-of-sight flags and the dominant-obstacle geometry `ex`
(dict of arrays, or None) and returns the single-look detection probability.
"""
import json
import math
import os

import numpy as np

C_LIGHT, K_BOLTZ, T0, R_EARTH = 299792458.0, 1.380649e-23, 290.0, 6371000.0

with open(os.path.join(os.path.dirname(__file__), "types.json"), encoding="utf-8") as _f:
    _T = json.load(_f)
TYPES = _T["types"]
TK = list(TYPES)
BASE = _T["base"]
DRONES = _T.get("drones", [])
PASSIVE_MASK = sum(1 << i for i, t in enumerate(TK) if TYPES[t]["passive"])
SELF_LOC = sum(1 << i for i, t in enumerate(TK) if t in ("radar", "rid"))


def db2lin(d):
    return np.power(10.0, np.asarray(d, dtype=float) / 10.0)


def erfc(x):
    """Numerical Recipes erfcc (relative error < 1.2e-7)."""
    x = np.asarray(x, dtype=float)
    z = np.abs(x)
    t = 1 / (1 + 0.5 * z)
    r = t * np.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (
        0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    return np.where(x >= 0, r, 2 - r)


def Phi(x):
    return 0.5 * erfc(-np.asarray(x, dtype=float) / math.sqrt(2))


def albersheim_pd(snr, N, pfa):
    snr = np.asarray(snr, dtype=float)
    A = math.log(0.62 / pfa)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        snr_db = 10 * np.log10(np.where(snr > 0, snr, 1))
        X = np.power(10.0, (snr_db + 5 * math.log10(N)) / (6.2 + 4.54 / math.sqrt(N + 0.44)))
        B = (X - A) / (0.12 * A + 1.7)
        pd = 1 / (1 + np.exp(-B))
    return np.where(snr > 0, pd, 0.0)


def shnidman_req(pd, N, pfa, sw):
    pd = np.asarray(pd, dtype=float)
    K = math.inf if sw == 0 else 1 if sw == 1 else N if sw == 2 else 2 if sw == 3 else 2 * N
    alpha = 0 if N < 40 else 0.25
    eta = math.sqrt(-0.8 * math.log(4 * pfa * (1 - pfa))) + np.sign(pd - 0.5) * np.sqrt(-0.8 * np.log(4 * pd * (1 - pd)))
    Xinf = eta * (eta + 2 * math.sqrt(N / 2 + (alpha - 0.25)))
    Cdb = 0.0
    if math.isfinite(K):
        C1 = (((17.7006 * pd - 18.4496) * pd + 14.5339) * pd - 3.525) / K
        C2 = (np.exp(27.31 * pd - 25.14) + (pd - 0.8) * (0.7 * math.log(1e-5 / pfa) + (2 * N - 20) / 80)) / K
        Cdb = np.where(pd <= 0.872, C1, C1 + C2)
    return np.power(10.0, Cdb / 10) * Xinf / N


def shnidman_pd(snr, N, pfa, sw):
    snr = np.asarray(snr, dtype=float)
    lo, hi = float(shnidman_req(0.1, N, pfa, sw)), float(shnidman_req(0.99, N, pfa, sw))
    a, b = np.full(snr.shape, 0.1), np.full(snr.shape, 0.99)
    for _ in range(40):
        m = (a + b) / 2
        big = shnidman_req(m, N, pfa, sw) > snr
        b = np.where(big, m, b)
        a = np.where(big, a, m)
    out = (a + b) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(snr <= lo, 0.1 * (snr / lo) ** 2, out)
        out = np.where(snr >= hi, 1 - 0.01 * (hi / np.where(snr > 0, snr, 1)), out)
    return np.where(snr > 0, out, 0.0)


def two_ray(p, ex, lam):
    if int(p.get("mp", 0)) != 1 or ex is None:
        return 1.0
    dH, hs, ht = ex["dH"], ex["hs"], ex["ht"]
    ok = (dH > 0) & (hs > 0) & (ht > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        psi = np.arctan((hs + ht) / np.where(dH > 0, dH, 1))
        sig = float(p.get("rough") or 0)
        rho = min(1.0, max(0.0, float(p.get("rho") or 0))) * np.exp(-2 * (2 * math.pi * sig * np.sin(psi) / lam) ** 2)
        dphi = 4 * math.pi * hs * ht / (lam * np.where(dH > 0, dH, 1))
        F = np.maximum(1e-4, 1 + rho * rho - 2 * rho * np.cos(dphi))
    return np.where(ok, F, 1.0)


RAIN_TAB = [(1, 2.59e-5, 0.9691), (2, 8.47e-5, 1.0664), (4, 1.071e-4, 1.6009), (6, 7.056e-4, 1.5900), (8, 4.115e-3, 1.3905),
            (10, 1.217e-2, 1.2571), (12, 2.386e-2, 1.1825), (15, 4.481e-2, 1.1233), (20, 9.164e-2, 1.0568), (25, 0.1571, 0.9991),
            (30, 0.2403, 0.9485), (35, 0.3374, 0.9047), (40, 0.4431, 0.8673)]


def rain_db_per_km(f_ghz, rate):
    """ITU-R P.838-3, horizontal polarization, log-interpolated."""
    if not rate > 0:
        return 0.0
    f = min(40.0, max(1.0, f_ghz))
    i = 0
    while i < len(RAIN_TAB) - 2 and RAIN_TAB[i + 1][0] < f:
        i += 1
    f0, k0, a0 = RAIN_TAB[i]
    f1, k1, a1 = RAIN_TAB[i + 1]
    u = math.log(f / f0) / math.log(f1 / f0)
    k = math.exp(math.log(k0) + u * (math.log(k1) - math.log(k0)))
    return k * rate ** (a0 + u * (a1 - a0))


def knife_edge(nu):
    nu = np.asarray(nu, dtype=float)
    clear = nu <= -0.78
    v = np.where(clear, 0.0, nu)   # evaluate the logarithm only where it is used (np.where computes both branches)
    return np.where(clear, 0.0, 6.9 + 20 * np.log10(np.sqrt((v - 0.1) ** 2 + 1) + v - 0.1))


def fresnel_r1(ex, lam):
    """First Fresnel zone radius at the dominant obstacle (m)."""
    return np.sqrt(lam * ex["d1"] * ex["d2"] / (ex["d1"] + ex["d2"]))


def diff_loss(ex, lam):
    d1, d2 = ex["d1"], ex["d2"]
    ok = (d1 > 0) & (d2 > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        nu = ex["h"] * np.sqrt(2 * (d1 + d2) / (lam * np.where(ok, d1 * d2, 1)))
    return np.where(ok, knife_edge(nu), 0.0)


def air_absorb(f, tC=15.0, rh=70.0):
    """ISO 9613-1 atmospheric absorption of sound (dB/km) at f (Hz), air temperature tC (°C), relative humidity rh (%)."""
    T = 273.15 + (tC if tC is not None and math.isfinite(tC) else 15.0)
    Tr = T / 293.15
    h = max(0.0, rh if rh is not None and math.isfinite(rh) else 70.0) * 10 ** (-6.8346 * (273.16 / T) ** 1.261 + 4.6151)
    frO = 24 + 4.04e4 * h * (0.02 + h) / (0.391 + h)
    frN = Tr ** -0.5 * (9 + 280 * h * math.exp(-4.170 * (Tr ** (-1 / 3) - 1)))
    f2 = f * f
    return 8686 * f2 * (1.84e-11 * math.sqrt(Tr) + Tr ** -2.5 * (0.01275 * math.exp(-2239.1 / T) / (frO + f2 / frO) + 0.1068 * math.exp(-3352 / T) / (frN + f2 / frN)))


ISO_BANDS = (63, 125, 250, 500, 1000, 2000, 4000, 8000)


def iso_ground(f, G, hs, hr, dp):
    """ISO 9613-2 ground attenuation A_gr = A_s + A_r + A_m (general method, Table 3) in the octave band of f (Hz)."""
    band = 63
    for x in ISO_BANDS:
        if abs(math.log2(f / x)) < abs(math.log2(f / band)):
            band = x
    hs, hr, dp = np.asarray(hs, float), np.asarray(hr, float), np.asarray(dp, float)
    e = 1 - np.exp(-dp / 50)

    def end(h):
        if band == 63:
            return np.full(np.broadcast(h, dp).shape, -1.5)
        if band == 125:
            return -1.5 + G * (1.5 + 3.0 * np.exp(-0.12 * (h - 5) ** 2) * e + 5.7 * np.exp(-0.09 * h * h) * (1 - np.exp(-2.8e-6 * dp * dp)))
        if band == 250:
            return -1.5 + G * (1.5 + 8.6 * np.exp(-0.09 * h * h) * e)
        if band == 500:
            return -1.5 + G * (1.5 + 14.0 * np.exp(-0.46 * h * h) * e)
        if band == 1000:
            return -1.5 + G * (1.5 + 5.0 * np.exp(-0.9 * h * h) * e)
        return np.full(np.broadcast(h, dp).shape, -1.5 * (1 - G))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(dp <= 30 * (hs + hr), 0.0, 1 - 30 * (hs + hr) / dp)
    return end(hs) + end(hr) + (-3 * q if band == 63 else -3 * q * (1 - G))


def iso_cmet(c0, hs, hr, dp):
    """ISO 9613-2 meteorological correction C0·(1 − 10(hs + hr)/dp) beyond dp = 10(hs + hr)."""
    if not c0 > 0:
        return 0.0
    hs, hr, dp = np.asarray(hs, float), np.asarray(hr, float), np.asarray(dp, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(dp > 10 * (hs + hr), c0 * (1 - 10 * (hs + hr) / dp), 0.0)


def beam_gain(th, tilt, bw):
    return np.exp(-2.7726 * ((th - tilt) / bw) ** 2) if bw > 0 else 1.0


def eff_revisit(s):
    base = max(0.05, float(s.get("revisit") or 1))
    p = s["p"]
    if s["type"] == "eo" and int(p.get("scan", 0)) == 1:
        hf = max(0.01, float(p.get("hfov") or 2))
        sec = min(360.0, float(s.get("sector") or 360))
        return max(base, math.ceil(sec / hf) * max(0.01, float(p.get("fovDwell") or 0.5)))
    return base


def pd_sensor(s, R, los, T, ex):
    pd = pd_single(s, R, los, T, ex)
    tobs = T.get("tobs", 0) or 0
    if not tobs > 0:
        return pd
    n = max(1.0, min(float(s.get("maxLooks") or 10), tobs / eff_revisit(s)))
    return np.where((pd > 0) & (pd < 1), 1 - np.power(1 - pd, n), pd)


def pd_single(s, R, los, T, ex):
    p, typ = s["p"], s["type"]
    R = np.asarray(R, dtype=float)
    los = np.broadcast_to(np.asarray(los, dtype=bool), R.shape)
    g = lambda k, d=0.0: float(p.get(k, d) if p.get(k, d) is not None else d)
    if typ == "radar":
        lam, G, N = C_LIGHT / (g("f") * 1e9), db2lin(g("g")), max(1, int(g("n", 1)))
        fT = beam_gain(ex["elT"], g("tilt"), g("bwEl")) if ex is not None and g("bwEl") > 0 else 1.0
        rain = db2lin(-2 * rain_db_per_km(g("f"), T["rain"]) * R / 1000)
        F2 = two_ray(p, ex, lam)
        snr = F2 * F2 * g("pt") * G * G * fT * fT * lam * lam * T["rcs"] * rain / (
            (4 * math.pi) ** 3 * R ** 4 * K_BOLTZ * T0 * g("b") * 1e6 * db2lin(g("nf")) * db2lin(g("l")))
        sinr = snr / db2lin(s.get("desens") or 0)
        if g("gamma", -99) > -90 and ex is not None:
            fG = beam_gain(ex["elG"], g("tilt"), g("bwEl")) if g("bwEl") > 0 else 1.0
            Ac = R * (g("bwAz") * math.pi / 180) * g("rres")
            scr = T["rcs"] * fT * fT / (db2lin(g("gamma")) * Ac * fG * fG) * db2lin(g("mti"))
            sinr = 1 / (1 / sinr + 1 / scr)
        if T["task"] == "id":
            return np.zeros(R.shape)
        if T["task"] == "cls":
            sinr = sinr / db2lin(g("clsMargin", 6))
        m = int(g("model"))
        fl = T.get("fluct")
        if fl is not None and float(fl) >= 0:      # the target's own fluctuation overrides each radar's model
            pd = shnidman_pd(sinr, N, g("pfa"), max(0, min(4, math.floor(float(fl) + 0.5))))
            return np.where(los, pd, 0.0)
        if m == 1:
            pd = np.power(g("pfa"), 1 / (1 + N * sinr))
        elif m == 3:
            pd = shnidman_pd(sinr, N, g("pfa"), max(0, min(4, math.floor(g("sw", 1) + 0.5))))
        elif m == 2:
            pd = np.clip(0.5 * erfc(math.sqrt(-math.log(g("pfa"))) - np.sqrt(N * sinr + 0.5)), 0, 1)
        else:
            pd = albersheim_pd(sinr, N, g("pfa"))
        return np.where(los, pd, 0.0)
    if typ == "eo":
        ir = int(g("band")) == 1
        if T["night"] and not ir:
            return np.zeros(R.shape)
        ifov = (g("hfov") * math.pi / 180) / g("npix")
        sig = 3.912 / max(0.01, g("vis")) + (3.912 / (T["visAtm"] * (3 if ir else 1)) if T["visAtm"] > 0 else 0) + \
            (1.076 * T["rain"] ** 0.67 / 4.343 if T["rain"] > 0 else 0)
        bg = 1.0
        if ex is not None:
            bg = np.where(ex["elT"] <= 0, min(1.0, max(0.0, g("cterr", 0.5))), 1.0)
        tau = np.exp(-sig * R / 1000)
        if int(g("det")) == 1 and T["task"] not in ("cls", "id"):   # point target: signal in the brightest pixel vs noise
            with np.errstate(divide="ignore", over="ignore"):
                fill = np.minimum(1.0, max(0.0, g("ee", 0.5)) * (T["dc"] / (R * ifov)) ** 2)
            c = T.get("dT", 3) / max(1e-6, g("netd", 0.05)) if ir else T["c0"] / max(1e-6, g("nec", 0.005))
            return np.where(los, Phi(c * bg * tau * fill - g("kth", 5)), 0.0)
        n50 = g("n50id", 6) if T["task"] == "id" else g("n50rec", 3) if T["task"] == "cls" else g("n50")
        r = ((T["dc"] / (R * ifov)) / 2) / n50
        E = 2.7 + 0.7 * r
        with np.errstate(divide="ignore", over="ignore"):
            pj = np.where(r > 0, 1 / (1 + np.power(r, -E)), 0.0)
        if ir:
            ok = T.get("dT", 3) * bg * tau >= g("mrt", 0.1)
        else:
            ok = T["c0"] * bg * tau >= g("cmin")
        return np.where(los & ok, pj, 0.0)
    if typ in ("rid", "rf"):
        erp, f = (T["ridPt"], T["ridF"]) if typ == "rid" else (T["c2Pt"], T["c2F"])
        lam = C_LIGHT / (f * 1e6)
        fspl = 20 * np.log10(np.maximum(R, 1) / 1000) + 20 * math.log10(f) + 32.44
        if int(g("diff")) == 1 and ex is not None:
            obs = diff_loss(ex, lam)
        else:
            clear = los
            if int(g("fres", 0)) == 1 and ex is not None:   # optional criterion: 60 % of the 1st Fresnel zone clear
                clear = los & (-ex["h"] >= 0.6 * fresnel_r1(ex, lam))
            obs = np.where(clear, 0.0, g("nlos"))
        if ex is not None:
            mp = np.where(los & ~(ex["h"] > 0), 10 * np.log10(two_ray(p, ex, lam)), 0.0)
        else:
            mp = np.where(los, 10 * np.log10(two_ray(p, None, lam)), 0.0)
        task = (g("idMargin", 10) if T["task"] == "id" else g("clsMargin", 5) if T["task"] == "cls" else 0) if typ == "rf" else 0
        pr = erp + g("gr") - fspl - g("lm") - obs + mp - rain_db_per_km(f / 1000, T["rain"]) * R / 1000 - (s.get("desens") or 0) - task
        return Phi((pr - g("sens")) / g("sig")) if g("sig") > 0 else np.where(pr >= g("sens"), 1.0, 0.0)
    if typ == "ac":
        if int(g("barrier")) == 1 and ex is not None:
            hpos = ex["h"] > 0
            with np.errstate(divide="ignore", invalid="ignore"):
                delta = ex["h"] ** 2 / 2 * (1 / ex["d1"] + 1 / ex["d2"])
                Nf = 2 * delta / (343 / (g("fa") or 200))
                bar = np.where(hpos, np.minimum(25, 10 * np.log10(3 + 20 * np.where(hpos, Nf, 0))), 0.0)
        else:
            bar = np.where(los, 0.0, g("nlos"))
        nl = g("nl") + (T.get("noiseAdd") or 0) + (10 * math.log10(1 + T["rain"] / 0.5) if T["rain"] > 0 else 0) - (5 if T["night"] else 0)
        if T["task"] == "id":
            return np.zeros(R.shape)
        alpha = air_absorb(g("fa") or 200, T.get("airT", 15), T.get("airRH", 70)) if g("alpha") < 0 else g("alpha")
        gm = 0.0
        if ex is not None and "dH" in ex:
            hs, hr, dp = np.maximum(0, ex["hs"]), np.maximum(0, ex["ht"]), ex["dH"]
            if g("gG", -1) >= 0:
                agr = iso_ground(g("fa") or 200, min(1.0, g("gG", -1)), hs, hr, dp)
                bar = np.where(dp > 0, np.where((bar > 0) & (int(g("barrier")) == 1), np.maximum(bar, agr), bar + agr), bar)
            gm = np.where(dp > 0, iso_cmet(g("c0met", 0), hs, hr, dp), 0.0)
        snr = T["sl"] - 20 * np.log10(np.maximum(R, 1)) - alpha * R / 1000 + g("ag") - nl - bar - gm - (g("clsMargin", 6) if T["task"] == "cls" else 0)
        return Phi((snr - g("dt")) / g("sig")) if g("sig") > 0 else np.where(snr >= g("dt"), 1.0, 0.0)
    return np.zeros(R.shape)


def range_at_pd(s, T):
    """Free-space range where Pd is still above the threshold (binary search)."""
    lo, hi = 1.0, 1e6
    one = lambda r: float(pd_sensor(s, np.array([r]), True, T, None)[0])
    if one(lo) < s["pdMin"]:
        return 0.0
    for _ in range(60):
        m = math.sqrt(lo * hi)
        if one(m) >= s["pdMin"]:
            lo = m
        else:
            hi = m
    return lo


def compute_interference(sensors, T, o):
    """Radars desensitize receivers nearby (port of computeInterference). o: on, sidelobe, duty, band, rej, rrej, all."""
    for s in sensors:
        s["desens"] = 0.0
        s["_intf"] = None
    if not o or not o.get("on"):
        return
    srcs = [s for s in sensors if s["type"] == "radar" and (s.get("deployed") or o.get("all"))]
    for v in sensors:
        if v["type"] not in ("radar", "rid", "rf"):
            continue
        pv = v["p"]
        fv = float(pv["f"]) if v["type"] == "radar" else (T["ridF"] if v["type"] == "rid" else T["c2F"]) / 1000
        N = -174 + 10 * math.log10(float(pv["b"]) * 1e6) + float(pv["nf"]) if v["type"] == "radar" else float(pv["sens"]) - 10
        Isum, worst = 0.0, None
        for r in srcs:
            if r is v:
                continue
            k = math.cos((r["lat"] + v["lat"]) / 2 * math.pi / 180)
            d = max(1.0, math.hypot((r["lon"] - v["lon"]) * 111320 * k, (r["lat"] - v["lat"]) * 110574))
            fs = float(r["p"]["f"])
            inband = abs(fs - fv) <= o["band"]
            fspl = 20 * math.log10(d / 1000) + 20 * math.log10(fs * 1000) + 32.44
            gr = o["sidelobe"] if v["type"] == "radar" else float(pv.get("gr") or 0)
            I = 10 * math.log10(float(r["p"]["pt"]) * 1000 * o["duty"]) + o["sidelobe"] + gr - fspl - (0 if inband else o["rej"]) - (o["rrej"] if v["type"] == "radar" else 0)
            Isum += 10 ** (I / 10)
            if worst is None or I > worst["I"]:
                worst = {"r": r["name"], "d": d, "I": I, "inband": inband}
        if Isum:
            v["desens"] = 10 * math.log10(1 + Isum / 10 ** (N / 10))
            v["_intf"] = worst
