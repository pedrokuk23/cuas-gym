"""Coverage engine: line-of-sight geometry per site, detection layers, fusion and requirement checks."""
import math

import numpy as np

from .geo import cell_center, hat
from .models import PASSIVE_MASK, R_EARTH, SELF_LOC, TK, pd_sensor
from .terrain import pixel_m

ALL_MASK = (1 << len(TK)) - 1
METRICS = {"any": "Cov", "same": "Same ≥2", "diff": "Mixed ≥2", "prob": "P(det)", "loc": "Loc"}
LMCAP = 13.8
BRG_DEF = {"rf": 3, "ac": 5, "eo": 0.2}


def lm_of(pd):
    pd = np.asarray(pd, dtype=float)
    with np.errstate(divide="ignore"):
        v = np.minimum(LMCAP, -np.log(1 - np.minimum(pd, 1 - 1e-6)))
    return np.where(pd > 0, v, 0.0)


def make_ctx(g, settings, T, classes=None, areas=None, threats=None, force=None):
    native = g.native
    mult = max(0.25, float(settings.get("losStep", 1) or 1))
    px = max(10.0, pixel_m(g.tiles, g.pr) or 30) if native else 0
    ctx = {"T": T, "classes": classes or [], "threats": threats or [], "kR": (float(settings.get("kEarth", 4 / 3)) or 4 / 3) * R_EARTH,
           "step": 2 * px * mult if native else 0.5 * g.sx * mult}
    ctx["plan"] = threat_plan(g, areas or [], ctx["threats"], ctx, force=force)
    THREAT_NAMES.update({t["id"]: t.get("name", t["id"]) for t in ctx["threats"]})
    slots = req_slots(areas or []) if ctx["plan"] else []
    prepare_slots(g, areas or [], slots)
    ctx["slot_plans"] = [threat_plan(g, areas, ctx["threats"], ctx, only=t, force=force) for t in slots]
    return ctx


def req_slots(areas):
    """Threats with requirements of their own (and flown in some coverage area): one extra layer each."""
    cov = [a for a in areas if (a.get("role") or "coverage") == "coverage"]
    flown = {e["id"] for a in cov for e in (a.get("threats") or [])}
    out = []
    for a in cov:
        for r in a.get("reqs") or []:
            t = r.get("threat")
            if t and t in flown and t not in out:
                out.append(t)
    return out


def prepare_slots(g, areas, slots):
    """g.slots, g.nCell (cells once per slot) and g.xz (zones, then the zones of per-threat requirements)."""
    g.slots = list(slots)
    g.nCell = g.nIn * (1 + len(slots))
    by_id = {a["id"]: a for a in areas}
    g.xz = [{"zi": zi, "z": z, "slot": 0} for zi, z in enumerate(g.zones)]
    if slots:
        for zi, z in enumerate(g.zones):
            seen = set()
            for e in eff_req(z, by_id):
                if e.get("threat") and e["threat"] not in seen:
                    seen.add(e["threat"])
                    g.xz.append({"zi": zi, "z": z, "slot": slots.index(e["threat"]) + 1})


def xz_index(g, zi, threat):
    if not threat or not getattr(g, "xz", None):
        return zi
    j = g.slots.index(threat) + 1 if threat in g.slots else -1
    return next((k for k, x in enumerate(g.xz) if x["zi"] == zi and x["slot"] == j), zi)


def agg_for(agg, e):
    return agg["slot"][e["threat"]] if e.get("threat") and agg.get("slot") and e["threat"] in agg["slot"] else agg


THREAT_NAMES = {}

# ---------- threats per area (port of threatPlan / planLayer) ----------
ALT_STEP, ALT_MAXN = 50, 6


def alt_levels(a, b):
    a = float(a) if a not in (None, "") else 0.0
    b = float(b) if b not in (None, "") else a
    if not math.isfinite(a):
        a = 0.0
    if not math.isfinite(b) or b <= a:
        return [a]
    n = min(ALT_MAXN, math.ceil((b - a) / ALT_STEP) + 1)
    return [math.floor((a + (b - a) * i / (n - 1)) * 10 + 0.5) / 10 for i in range(n)]


def threat_T(base, th):
    erp = lambda v: -300.0 if v in (None, "") else float(v)
    pick = lambda k: base[k] if th.get(k) in (None, "") else float(th[k])
    return dict(base, rcs=float(th["rcs"]), dc=float(th["dc"]), sl=float(th["sl"]), dT=pick("dT"), fluct=pick("fluct"), c0=pick("c0"),
                ridPt=erp(th.get("ridPt")), c2Pt=erp(th.get("c2Pt")))


def area_threats(a, threats):
    ids = {t["id"] for t in threats}
    return [e for e in (a.get("threats") or []) if e.get("id") in ids]


def threat_plan(g, areas, threats, ctx, only=None, force=None):
    """Combos {T, alt (None = analysis altitude), cells}; None when no coverage area lists a threat."""
    if not any((a.get("role") or "coverage") == "coverage" and area_threats(a, threats) for a in areas):
        return None
    by_id = {a["id"]: a for a in areas}
    combos = {}

    def add(key, T, alt, cells):
        combos.setdefault(key, {"key": key, "T": T, "alt": alt, "cells": []})["cells"].append(np.asarray(cells))
    for z in g.zones:
        ent = [e for aid in z["areaIds"] if aid in by_id for e in area_threats(by_id[aid], threats)]
        if not ent:
            if only is not None:
                continue
            add("base", ctx["T"], force, z["cells"])
            for i, k in enumerate(ctx["classes"]):
                add("cls%d" % i, k["T"], force if force is not None else k.get("alt"), z["cells"])
            continue
        for e in ent:
            if only is not None and e["id"] != only:
                continue
            th = next(t for t in threats if t["id"] == e["id"])
            T = threat_T(ctx["T"], th)
            for h in ([force] if force is not None else alt_levels(e.get("min"), e.get("max"))):
                add(f"{e['id']}@{h}", T, h, z["cells"])
    out = []
    for k in combos.values():
        k["cells"] = np.concatenate(k["cells"]).astype(np.int64)
        out.append(k)
    return {"combos": out}


def plan_layer(g, s, alt, alt_ref, ctx, plan=None):
    N, G = g.nx * g.ny, site_geom(g, s, ctx)
    pd, bit, los, seen = np.full(N, np.inf, np.float32), np.ones(N, np.uint8), np.ones(N, np.uint8), np.zeros(N, bool)
    gr = ground_ref(g)
    for k in (plan or ctx["plan"])["combos"]:
        c = k["cells"]
        a = k["alt"] if k["alt"] is not None else alt
        zt = gr[c] + a if alt_ref == "AGL" else np.full(len(c), float(a))
        v, valid, vis = detect(s, k["T"], ctx["kR"], G["D"][c], G["AZ"][c], g.H[c].astype(float), G["zs"], zt, G["ZP"][c], G["TS"][c])
        p = np.where(valid, v, 0.0).astype(np.float32)
        pd[c] = np.minimum(pd[c], p)
        bit[c] &= (valid & (v >= s["pdMin"])).astype(np.uint8)
        los[c] &= (valid & vis).astype(np.uint8)
        seen[c] = True
    pd[~seen] = 0
    bit[~seen] = 0
    los[~seen] = 0
    return {"s": s, "pd": pd, "bit": bit, "los": los, "geo": G}


def site_geom(g, s, ctx):
    """Distance, azimuth and the 'shadow ceiling' ZP for every cell (exact profiles; a faster option interpolates
    smooth blocks: 'adaptive line of sight')."""
    key = (g.ver, s["lat"], s["lon"], s["mast"], s["ground"], ctx["kR"], ctx["step"])
    if s.get("_geo") is not None and s["_geo"]["key"] == key:
        return s["_geo"]
    N, kR, step = g.nx * g.ny, ctx["kR"], ctx["step"]
    xs, ys = map(float, g.pr.fwd(s["lat"], s["lon"]))
    zs = s["ground"] + s["mast"]
    idx = g.idx
    cx, cy = cell_center(g, idx)
    dx, dy = cx - xs, cy - ys
    Dh = np.hypot(dx, dy)
    D, AZ = np.zeros(N, np.float32), np.zeros(N, np.float32)
    D[idx] = Dh
    AZ[idx] = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    n = np.floor(Dh / step).astype(int)
    zp = np.full(len(idx), -1e9)
    tb = np.full(len(idx), 0.5)
    for q in range(1, int(n.max()) if len(n) else 0):
        m = q < n
        if not m.any():
            break
        t = q / n[m]
        h = hat(g, xs + dx[m] * t, ys + dy[m] * t)
        d1 = Dh[m] * t
        need = zs + (h + d1 * (Dh[m] - d1) / (2 * kR) - zs) / t
        up = np.isfinite(need) & (need > zp[m])
        sel = np.flatnonzero(m)[up]
        zp[sel] = need[up]
        tb[sel] = t[up]
    ZP = np.full(N, -1e9, np.float32); ZP[idx] = zp
    TS = np.full(N, 0.5, np.float32); TS[idx] = tb
    ZMIN = np.full(N, np.inf, np.float32); ZMIN[idx] = np.maximum(g.H[idx], zp)
    s["_geo"] = {"key": key, "xs": xs, "ys": ys, "zs": zs, "D": D, "AZ": AZ, "ZP": ZP, "TS": TS, "ZMIN": ZMIN}
    return s["_geo"]


def detect(s, T, kR, D, az, h, zs, zt, zp, ts):
    """Vectorized detect1: returns (pd, valid, vis). valid = inside range, elevation and sector limits."""
    D, h, zt, zp, ts = (np.asarray(a, dtype=float) for a in (D, h, zt, zp, ts))
    valid = zt >= h
    dzEff = zt - zs - D * D / (2 * kR)
    R = np.maximum(1, np.hypot(D, dzEff))
    valid &= (R >= s["rMin"]) & (R <= s["rMax"])
    el = np.degrees(np.arctan2(dzEff, D))
    valid &= (el >= s["elMin"]) & (el <= s["elMax"])
    if s["sector"] < 360:
        az0 = ((s["az"] % 360) + 360) % 360
        d = np.abs(np.asarray(az) - az0)
        d = np.minimum(d, 360 - d)
        valid &= d <= s["sector"] / 2
    vis = zt >= np.maximum(h, zp)
    needL = s["type"] in ("radar", "eo") or bool(s.get("needLOS", True))
    ref = np.minimum(s["ground"] if s.get("ground") is not None else h, h)
    ex = {"h": (zp - zt) * ts, "d1": np.maximum(1, ts * D), "d2": np.maximum(1, (1 - ts) * D), "elT": el,
          "elG": np.degrees(np.arctan2(h - zs - D * D / (2 * kR), D)), "dH": D, "hs": np.maximum(0.5, zs - ref), "ht": np.maximum(0.5, zt - ref)}
    pd = pd_sensor(s, R, vis, T, ex)
    pd = np.where(needL & ~vis, 0.0, pd)
    return pd, valid, vis


def ground_ref(g):
    return g.G if g.G is not None else g.H


def sensor_layer1(g, s, alt, alt_ref, ctx, T):
    N, G, idx = g.nx * g.ny, site_geom(g, s, ctx), g.idx
    h = g.H[idx].astype(float)
    zt = ground_ref(g)[idx] + alt if alt_ref == "AGL" else np.full(len(idx), float(alt))
    pdv, valid, vis = detect(s, T, ctx["kR"], G["D"][idx], G["AZ"][idx], h, G["zs"], zt, G["ZP"][idx], G["TS"][idx])
    pd, bit, los = np.zeros(N, np.float32), np.zeros(N, np.uint8), np.zeros(N, np.uint8)
    pd[idx] = np.where(valid, pdv, 0)
    los[idx] = valid & vis
    bit[idx] = valid & (pdv >= s["pdMin"])
    return {"s": s, "pd": pd, "bit": bit, "los": los, "geo": G}


def sensor_layer(g, s, alt, alt_ref, ctx):
    if s.get("patrol") and len(s["patrol"].get("pts") or []) >= 2:
        raise NotImplementedError("sensors on patrol routes are not part of this module")
    if ctx.get("plan"):
        o = plan_layer(g, s, alt, alt_ref, ctx)
        if ctx.get("slot_plans"):
            o["tl"] = [plan_layer(g, s, alt, alt_ref, ctx, P) for P in ctx["slot_plans"]]
        return o
    base = sensor_layer1(g, s, alt, alt_ref, ctx, ctx["T"])
    for k in ctx["classes"]:
        o = sensor_layer1(g, s, k["alt"] if k.get("alt") is not None else alt, alt_ref, ctx, k["T"])
        base["pd"] = np.minimum(base["pd"], o["pd"])
        base["bit"] &= o["bit"]
        base["los"] &= o["los"]
    return base


def is_cued(s):
    return int(float(s["p"].get("cue", 0) or 0)) == 1


def is_bearing(s):
    return not (SELF_LOC & (1 << TK.index(s["type"])))


def brg_of(s):
    v = float(s["p"].get("brg") or 0)
    return v if v > 0 else BRG_DEF.get(s["type"], 3)


def loc_j(g, o):
    """Fisher information [xx, xy, yy] of one bearing sensor where it detects; None for radar / Remote ID."""
    if not is_bearing(o["s"]):
        return None
    if o.get("_J") is not None:
        return o["_J"]
    sg = brg_of(o["s"]) * math.pi / 180
    geos = [v["_geo"] for v in o["views"] if v.get("_geo")] if o.get("views") else [o["geo"]]
    J = np.zeros((3, len(o["bit"])))
    for G in geos:
        D = np.maximum(30, G["D"].astype(float))
        a = np.radians(G["AZ"].astype(float))
        w = np.where(o["bit"] > 0, 1 / (sg * sg * D * D * len(geos)), 0.0)
        ca, sa = np.cos(a), np.sin(a)
        J += np.stack([w * ca * ca, -w * sa * ca, w * sa * sa])
    o["_J"] = J
    return o["_J"]


def loc_err(xx, xy, yy):
    det = xx * yy - xy * xy
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.sqrt((xx + yy) / det)
    return np.where(det > 1e-6 * (xx + yy) ** 2, e, np.inf)  # nearly parallel bearings: no fix (float32 vs float64 safe)


def apply_cue(g, per):
    per = [o.get("raw", o) for o in per]
    if not any(is_cued(o["s"]) for o in per):
        return per
    U = np.zeros(g.nx * g.ny, dtype=bool)
    for o in per:
        if not is_cued(o["s"]):
            U |= o["bit"] > 0
    out = []
    for o in per:
        if not is_cued(o["s"]):
            out.append(o)
            continue
        c = dict(o, raw=o, _J=None)
        c["bit"] = np.where(U, o["bit"], 0).astype(np.uint8)
        c["pd"] = np.where(U, o["pd"], 0).astype(np.float32)
        out.append(c)
    return out


def aggregate(g, per, k=2):
    per = apply_cue(g, per)
    N, idx = g.nx * g.ny, g.idx
    ti = [TK.index(o["s"]["type"]) for o in per]
    count = np.zeros(N, np.int32)
    tc = np.zeros((len(TK), N), np.int32)
    for o, t in zip(per, ti):
        b = o["bit"].astype(np.int32)
        count += b
        tc[t] += b
    tmask = sum(((tc[t] > 0).astype(np.int64) << t) for t in range(len(TK)))
    smask = sum(((tc[t] >= 2).astype(np.int64) << t) for t in range(len(TK)))
    ntypes = (tc > 0).sum(axis=0)
    flags = (tmask > 0).astype(int) | (((tmask & PASSIVE_MASK) > 0).astype(int) << 1) | ((smask > 0).astype(int) << 2) | ((ntypes >= 2).astype(int) << 3)
    lm = np.zeros((len(TK), N))
    for o, t in zip(per, ti):
        lm[t] += lm_of(o["pd"])
    jt = [None] * len(TK)
    for o, t in zip(per, ti):
        J = loc_j(g, o)
        if J is not None:
            jt[t] = J.copy() if jt[t] is None else jt[t] + J
    q = np.ones(N)
    for o in per:
        q *= 1 - o["pd"]
    por = 1 - q
    # P(≥ k sensors detect) by dynamic programming over the sensors
    dp = np.zeros((len(per) + 2, N)); dp[0] = 1
    for m, o in enumerate(per):
        p = o["pd"].astype(float)
        for r in range(m + 1, 0, -1):
            dp[r] = dp[r] * (1 - p) + dp[r - 1] * p
        dp[0] *= 1 - p
    pk = dp[k:len(per) + 1].sum(axis=0) if k <= len(per) else np.zeros(N)
    xx = sum(j[0] for j in jt if j is not None) if any(j is not None for j in jt) else np.zeros(N)
    xy = sum(j[1] for j in jt if j is not None) if any(j is not None for j in jt) else np.zeros(N)
    yy = sum(j[2] for j in jt if j is not None) if any(j is not None for j in jt) else np.zeros(N)
    lerr = np.where((tmask & SELF_LOC) > 0, 0.0, loc_err(xx, xy, yy))
    out = {"per": per, "count": count, "ntypes": ntypes, "tmask": tmask, "smask": smask, "flags": flags, "lm": lm, "jt": jt,
           "por": por, "pk": pk, "lerr": lerr, "k": k}
    if per and per[0].get("tl") is not None and getattr(g, "slots", None):
        out["slot"] = {tid: aggregate(g, [slot_layer(o.get("raw", o), j + 1) for o in per], k) for j, tid in enumerate(g.slots)}
    return out


def slot_layer(o, j):
    """Layer of one threat slot (j ≥ 1) in the shape of a sensor layer; j = 0: the layer itself."""
    if not j:
        return o
    L = o["tl"][j - 1]
    if "_o" not in L:
        L["_o"] = {"s": o["s"], "pd": L["pd"], "bit": L["bit"], "los": L["los"], "geo": o["geo"], "views": o.get("views")}
    return L["_o"]


def scope_mask(sc):
    if sc == "all":
        return ALL_MASK
    if sc == "passive":
        return PASSIVE_MASK
    if sc == "active":
        return ALL_MASK & ~PASSIVE_MASK
    return 1 << TK.index(sc) if sc in TK else ALL_MASK


def req_par(r):
    m = r["metric"]
    return float(r.get("p") or 0.9) if m == "prob" else float(r.get("e") or 100) if m == "loc" else None


def req_key(r):
    return r["metric"] + ":" + r["scope"] + (":" + repr(req_par(r)) if r["metric"] in ("prob", "loc") else "") + ("@" + r["threat"] if r.get("threat") else "")


def popcount(x):
    x = np.asarray(x, dtype=np.int64)
    n = np.zeros(x.shape, dtype=np.int64)
    for t in range(len(TK)):
        n += (x >> t) & 1
    return n


def cell_ok(agg, metric, mask, p):
    """Boolean array over all cells: requirement (metric, scope mask, parameter) met."""
    if metric == "prob":
        tau = -math.log(1 - min(0.999999, p or 0.9))
        sm = sum(agg["lm"][t] for t in range(len(TK)) if mask & (1 << t))
        return sm >= tau - 1e-9
    if metric == "loc":
        selfok = (agg["tmask"] & mask & SELF_LOC) > 0
        xx = np.zeros(len(selfok)); xy = np.zeros(len(selfok)); yy = np.zeros(len(selfok))
        for t, J in enumerate(agg["jt"]):
            if J is not None and mask & (1 << t):
                xx = xx + J[0]; xy = xy + J[1]; yy = yy + J[2]
        return selfok | (loc_err(xx, xy, yy) <= (p or 100) + 1e-9)
    if metric == "any":
        return (agg["tmask"] & mask) != 0
    if metric == "same":
        return (agg["smask"] & mask) != 0
    return popcount(agg["tmask"] & mask) >= 2


def eff_req(z, areas_by_id):
    m = {}
    flown = {e["id"] for aid in z["areaIds"] for e in (areas_by_id.get(aid, {}).get("threats") or [])}
    for aid in z["areaIds"]:
        for r in areas_by_id[aid].get("reqs", []):
            if r.get("threat") and r["threat"] not in flown:      # that threat does not fly in this zone
                continue
            k, v = req_key(r), float(r.get("min") or 0)
            if k not in m or v > m[k]["min"]:
                m[k] = {"key": k, "metric": r["metric"], "scope": r["scope"], "mask": scope_mask(r["scope"]), "min": v, "p": req_par(r),
                        "threat": r.get("threat") or None}
    return list(m.values())


def zone_name(g, z, by_id):
    names = [by_id[i]["name"] for i in z["areaIds"]]
    if len(names) > 1:
        return " ∩ ".join(names)
    only = any(o is not z and (o["mask"] & z["mask"]) == z["mask"] for o in g.zones)
    return names[0] + (" only" if only else "")


def scope_label(sc):
    from .models import TYPES
    return {"all": "all types", "passive": "passive types", "active": "active (radar)"}.get(sc) or TYPES.get(sc, {}).get("label", sc)


def req_label(e):
    sc = scope_label(e["scope"])
    if e["metric"] == "prob":
        return f"P ≥ {e['p']} · {sc}"
    if e["metric"] == "loc":
        return f"Loc ≤ {e['p']:g} m · {sc}"
    return f"{METRICS[e['metric']]} · {sc}" + (f" · {THREAT_NAMES.get(e['threat'], e['threat'])}" if e.get("threat") else "")


def zone_report(g, agg, areas):
    by_id = {a["id"]: a for a in areas}
    zones, got, total, nchk, nok = [], 0, 0, 0, 0
    cache = {}
    for zi, z in enumerate(g.zones):
        checks = []
        for e in eff_req(z, by_id):
            if e["key"] not in cache:
                cache[e["key"]] = cell_ok(agg_for(agg, e), e["metric"], e["mask"], e["p"])
            n = int(cache[e["key"]][z["cells"]].sum())
            N = len(z["cells"])
            need = math.ceil(e["min"] / 100 * N - 1e-9)
            got += min(n, need); total += need
            ok = n / N * 100 >= e["min"] - 1e-9
            checks.append(dict(e, got=n / N * 100, ok=ok, label=req_label(e)))
            nchk += 1; nok += ok
        zones.append({"zone": zone_name(g, z, by_id), "km2": len(z["cells"]) * g.sx * g.sy / 1e6, "checks": checks, "ok": all(c["ok"] for c in checks)})
    return {"zones": zones, "nChecks": nchk, "nOk": nok, "fulfil": got / total if total else 1.0}


def overall(g, agg, cost_of):
    idx, N = g.idx, max(1, g.nIn)
    f = agg["flags"][idx]
    Mc = float((f & 1).sum()) / N
    cost = sum(cost_of(o["s"]) for o in agg["per"])
    return {"Mc": Mc, "Mg": 1 - Mc, "passive": float(((f >> 1) & 1).sum()) / N, "same": float(((f >> 2) & 1).sum()) / N,
            "ovl": float(((f >> 3) & 1).sum()) / N, "kk": float((agg["count"][idx] >= agg["k"]).sum()) / N,
            "meanP": float(agg["por"][idx].sum()) / N, "cost": cost}
