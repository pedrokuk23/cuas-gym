"""Local projection, polygons and the multi-area grid of square cells."""
import math

import numpy as np

from .terrain import sample


class Proj:
    def __init__(self, lat0, lon0):
        r = lat0 * math.pi / 180
        self.lat0, self.lon0 = lat0, lon0
        self.mLat = 111132.954 - 559.822 * math.cos(2 * r) + 1.175 * math.cos(4 * r)
        self.mLon = 111412.84 * math.cos(r) - 93.5 * math.cos(3 * r)

    def fwd(self, la, lo):
        return (np.asarray(lo) - self.lon0) * self.mLon, (np.asarray(la) - self.lat0) * self.mLat

    def inv(self, x, y):
        return self.lat0 + np.asarray(y) / self.mLat, self.lon0 + np.asarray(x) / self.mLon


def in_poly(x, y, P):
    c = False
    j = len(P) - 1
    for i in range(len(P)):
        if ((P[i][1] > y) != (P[j][1] > y)) and (x < (P[j][0] - P[i][0]) * (y - P[i][1]) / (P[j][1] - P[i][1]) + P[i][0]):
            c = not c
        j = i
    return c


def in_poly_v(x, y, P):
    """Vectorized even-odd test."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    c = np.zeros(x.shape, dtype=bool)
    j = len(P) - 1
    for i in range(len(P)):
        xi, yi, xj, yj = P[i][0], P[i][1], P[j][0], P[j][1]
        cond = (yi > y) != (yj > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xx = (xj - xi) * (y - yi) / (yj - yi) + xi
        c ^= cond & (x < xx)
        j = i
    return c


def _sgn(v):
    return (v > 0) - (v < 0)


def seg_x(a, b, c, d):
    o = lambda p, q, r: _sgn((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
    return o(a, b, c) != o(a, b, d) and o(c, d, a) != o(c, d, b)


def rect_touches_poly(xa, ya, xb, yb, P):
    C = [(xa, ya), (xb, ya), (xb, yb), (xa, yb)]
    if any(in_poly(c[0], c[1], P) for c in C) or in_poly((xa + xb) / 2, (ya + yb) / 2, P):
        return True
    if any(xa <= p[0] <= xb and ya <= p[1] <= yb for p in P):
        return True
    j = len(P) - 1
    for i in range(len(P)):
        for k in range(4):
            if seg_x(P[j], P[i], C[k], C[(k + 1) % 4]):
                return True
        j = i
    return False


def area_km2(pts):
    if len(pts) < 3:
        return 0.0
    pr = Proj(pts[0][0], pts[0][1])
    P = [pr.fwd(p[0], p[1]) for p in pts]
    a, j = 0.0, len(P) - 1
    for i in range(len(P)):
        a += (P[j][0] + P[i][0]) * (P[j][1] - P[i][1])
        j = i
    return abs(a / 2) / 1e6


class Grid:
    pass


_ver = [0]


def build_grid(areas, sensors, cs, mesh, sub, tiles, src="geotiff", bare=0, obstacles=None, native=True):
    areas = [a for a in areas if len(a["pts"]) >= 3]
    if not areas:
        raise ValueError("draw at least one area first")
    cs, sub = max(5.0, float(cs)), max(1, int(sub))
    allp = [p for a in areas for p in a["pts"]] + [(s["lat"], s["lon"]) for s in sensors]
    la, lo = [p[0] for p in allp], [p[1] for p in allp]
    pr = Proj((min(la) + max(la)) / 2, (min(lo) + max(lo)) / 2)
    sx = sy = cs
    XY = [pr.fwd(p[0], p[1]) for p in allp]
    xs_, ys_ = [float(p[0]) for p in XY], [float(p[1]) for p in XY]
    x0 = math.floor((min(xs_) - sx) / sx) * sx
    y0 = math.floor((min(ys_) - sy) / sy) * sy
    nx = math.ceil((max(xs_) + sx - x0) / sx)
    ny = math.ceil((max(ys_) + sy - y0) / sy)
    N = nx * ny
    # sub-samples, cell-major
    jj, ii, bb, aa = np.meshgrid(np.arange(ny), np.arange(nx), np.arange(sub), np.arange(sub), indexing="ij")
    px = x0 + (ii + (aa + 0.5) / sub) * sx
    py = y0 + (jj + (bb + 0.5) / sub) * sy
    plat, plon = pr.inv(px.ravel(), py.ravel())
    if src == "flat":
        el = np.zeros(plat.shape)
    else:
        el = sample(tiles, plat, plon)
    miss = int(np.isnan(el).sum())
    el = np.nan_to_num(el, nan=0.0).reshape(N, sub * sub)
    H = el.mean(axis=1).astype(np.float32)
    Hs = np.sqrt(np.maximum(0, (el ** 2).mean(axis=1) - H.astype(float) ** 2)).astype(np.float32)
    Hmin = el.min(axis=1).astype(np.float32)
    P = [[tuple(map(float, pr.fwd(p[0], p[1]))) for p in a["pts"]] for a in areas]
    bbx = [(min(q[0] for q in Q), min(q[1] for q in Q), max(q[0] for q in Q), max(q[1] for q in Q)) for Q in P]
    pmask = np.zeros(N, dtype=np.int64)
    for j in range(ny):
        for i in range(nx):
            c = j * nx + i
            xa, ya = x0 + i * sx, y0 + j * sy
            xb, yb = xa + sx, ya + sy
            m = 0
            for q, Q in enumerate(P):
                B = bbx[q]
                if xb < B[0] or xa > B[2] or yb < B[1] or ya > B[3]:
                    continue
                hit = in_poly(xa + sx / 2, ya + sy / 2, Q) if mesh == "center" else rect_touches_poly(xa, ya, xb, yb, Q)
                if hit:
                    m |= 1 << q
            pmask[c] = m
    inside = pmask > 0
    idx = np.flatnonzero(inside)
    if not len(idx):
        raise ValueError("no cells fall inside the areas — use a smaller cell size")
    zm = {}
    for c in idx:
        zm.setdefault(int(pmask[c]), []).append(int(c))
    zones = [{"mask": m, "cells": np.array(cells), "areaIds": [a["id"] for q, a in enumerate(areas) if m & (1 << q)]} for m, cells in zm.items()]
    zones.sort(key=lambda z: -len(z["cells"]))
    g = Grid()
    _ver[0] += 1
    g.ver, g.pr, g.x0, g.y0, g.sx, g.sy, g.nx, g.ny = _ver[0], pr, x0, y0, sx, sy, nx, ny
    g.H, g.Hs, g.sub, g.inside, g.pmask, g.idx, g.nIn = H, Hs, sub, inside, pmask, idx, len(idx)
    g.areaIds, g.zones, g.src, g.miss, g.cs, g.mesh = [a["id"] for a in areas], zones, src, miss, cs, mesh
    g.hmin, g.hmax = float(H[idx].min()), float(H[idx].max())
    g.tiles = tiles
    g.native = native and src == "geotiff" and len(tiles) > 0
    g.bare = float(bare or 0)
    g.G = bare_ground(g, Hmin, g.bare)
    g.ob = obstacle_raster(g, obstacles or [])
    return g


def cell_center(g, c):
    c = np.asarray(c)
    return g.x0 + (c % g.nx + 0.5) * g.sx, g.y0 + (c // g.nx + 0.5) * g.sy


def cell_of(g, lat, lon):
    x, y = g.pr.fwd(lat, lon)
    i, j = math.floor((x - g.x0) / g.sx), math.floor((y - g.y0) / g.sy)
    return -1 if (i < 0 or j < 0 or i >= g.nx or j >= g.ny) else j * g.nx + i


def hat(g, x, y):
    """Terrain (surface) height at local x, y: full-resolution DEM when native, else the cell mean; obstacles on top.
    NaN outside the terrain."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    v = np.full(x.shape, np.nan)
    if g.native:
        la, lo = g.pr.inv(x, y)
        v = sample(g.tiles, la, lo)
    miss = np.isnan(v)
    if miss.any():
        i = np.floor((x - g.x0) / g.sx).astype(int)
        j = np.floor((y - g.y0) / g.sy).astype(int)
        ok = miss & (i >= 0) & (j >= 0) & (i < g.nx) & (j < g.ny)
        v[ok] = g.H[j[ok] * g.nx + i[ok]]
    o = g.ob
    if o is not None:
        i = np.floor((x - o["x0"]) / o["res"]).astype(int)
        j = np.floor((y - o["y0"]) / o["res"]).astype(int)
        ok = (i >= 0) & (j >= 0) & (i < o["nx"]) & (j < o["ny"]) & np.isfinite(v)
        t = np.full(x.shape, -1e9)
        t[ok] = o["top"][j[ok] * o["nx"] + i[ok]]
        v = np.where(ok & (t > v), t, v)
    return v


def bare_ground(g, Hmin, R):
    if not R > 0:
        return None
    r = max(0, math.floor(R / g.sx + 0.5))  # round half up
    A = Hmin.reshape(g.ny, g.nx).astype(np.float64)
    B = np.full(A.shape, np.inf)
    for d in range(-r, r + 1):
        sh = np.full(A.shape, np.inf)
        if d >= 0:
            sh[:, :g.nx - d] = A[:, d:]
        else:
            sh[:, -d:] = A[:, :g.nx + d]
        B = np.minimum(B, sh)
    C = np.full(A.shape, np.inf)
    for d in range(-r, r + 1):
        sh = np.full(A.shape, np.inf)
        if d >= 0:
            sh[:g.ny - d, :] = B[d:, :]
        else:
            sh[-d:, :] = B[:g.ny + d, :]
        C = np.minimum(C, sh)
    return C.ravel().astype(np.float32)


def obstacle_raster(g, obs):
    Ps = []
    for o in obs:
        if len(o.get("pts", [])) >= 3 and float(o.get("h") or 0) > 0:
            Q = [tuple(map(float, g.pr.fwd(p[0], p[1]))) for p in o["pts"]]
            Ps.append((float(o["h"]), Q))
    if not Ps:
        return None
    xa = min(q[0] for _, Q in Ps for q in Q); xb = max(q[0] for _, Q in Ps for q in Q)
    ya = min(q[1] for _, Q in Ps for q in Q); yb = max(q[1] for _, Q in Ps for q in Q)
    res = max(2.0, min(10.0, g.sx / 4))
    while ((xb - xa) / res) * ((yb - ya) / res) > 2e7:
        res *= 1.5
    nx, ny = max(1, math.ceil((xb - xa) / res)), max(1, math.ceil((yb - ya) / res))
    top = np.full(nx * ny, -1e9, dtype=np.float32)

    def surf(x, y):
        if g.src == "geotiff" and g.tiles:
            la, lo = g.pr.inv(x, y)
            v = sample(g.tiles, np.array([la]), np.array([lo]))[0]
            if np.isfinite(v):
                return float(v)
        i, j = math.floor((x - g.x0) / g.sx), math.floor((y - g.y0) / g.sy)
        return float(g.H[j * g.nx + i]) if 0 <= i < g.nx and 0 <= j < g.ny else 0.0

    for h, Q in Ps:
        t = min(surf(x, y) for x, y in Q) + h
        bx0, bx1 = min(q[0] for q in Q), max(q[0] for q in Q)
        by0, by1 = min(q[1] for q in Q), max(q[1] for q in Q)
        j0, j1 = max(0, math.floor((by0 - ya) / res)), min(ny - 1, math.floor((by1 - ya) / res))
        i0, i1 = max(0, math.floor((bx0 - xa) / res)), min(nx - 1, math.floor((bx1 - xa) / res))
        if j1 < j0 or i1 < i0:
            continue
        J, I = np.meshgrid(np.arange(j0, j1 + 1), np.arange(i0, i1 + 1), indexing="ij")
        X, Y = xa + (I + 0.5) * res, ya + (J + 0.5) * res
        inn = in_poly_v(X, Y, Q)
        k = (J * nx + I)[inn]
        top[k] = np.maximum(top[k], t)
    return {"x0": xa, "y0": ya, "res": res, "nx": nx, "ny": ny, "top": top, "n": len(Ps)}
