"""GeoTIFF terrain tiles (EPSG:4326) and bilinear sampling."""
import base64
import math
import glob
import io
import os
import re
import struct
import zlib

import numpy as np

try:
    import tifffile
except ImportError:          # optional: only needed for compressions other than deflate / none
    tifffile = None


class _Page:
    """Minimal TIFF reader (classic and BigTIFF): the tags and data layout of the first image."""
    def __init__(self, path):
        self.fh = open(path, "rb") if isinstance(path, str) else path
        bo = self.fh.read(2)
        self.e = "<" if bo == b"II" else ">"
        e = self.e
        ver = struct.unpack(e + "H", self.fh.read(2))[0]
        big = ver == 43
        if big:
            self.fh.read(4)
            off = struct.unpack(e + "Q", self.fh.read(8))[0]
        else:
            off = struct.unpack(e + "I", self.fh.read(4))[0]
        self.fh.seek(off)
        n = struct.unpack(e + ("Q" if big else "H"), self.fh.read(8 if big else 2))[0]
        size = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 16: 8, 17: 8, 18: 8}
        fmt = {1: "B", 2: "s", 3: "H", 4: "I", 5: "II", 6: "b", 7: "B", 8: "h", 9: "i", 10: "ii", 11: "f", 12: "d", 16: "Q", 17: "q", 18: "Q"}
        self.tags = {}
        for _ in range(n):
            ent = self.fh.read(20 if big else 12)
            tag, typ = struct.unpack(e + "HH", ent[:4])
            cnt = struct.unpack(e + ("Q" if big else "I"), ent[4:12] if big else ent[4:8])[0]
            raw = ent[12:20] if big else ent[8:12]
            nb = size.get(typ, 1) * cnt
            if nb > len(raw):
                pos = self.fh.tell()
                self.fh.seek(struct.unpack(e + ("Q" if big else "I"), raw)[0])
                raw = self.fh.read(nb)
                self.fh.seek(pos)
            if typ == 2:
                val = raw[:cnt].decode("latin-1")
            else:
                k = len(fmt[typ])
                val = struct.unpack(e + fmt[typ][0] * (cnt * k), raw[:nb])
            self.tags[tag] = val

    def get(self, tag, default=None):
        v = self.tags.get(tag, default)
        return v[0] if isinstance(v, tuple) and len(v) == 1 else v


def _decode(pg):
    t = pg.get
    W, H = t(256), t(257)
    bps = (t(258) if not isinstance(t(258), tuple) else t(258)[0]) // 8
    fmt = t(339, 1)
    fmt = fmt[0] if isinstance(fmt, tuple) else fmt
    comp, pred = t(259, 1), t(317, 1)
    if comp not in (1, 8, 32946):
        if tifffile is None:
            raise ValueError("this GeoTIFF compression needs: pip install tifffile imagecodecs")
        return None
    dt = np.dtype((("f" if fmt == 3 else "i" if fmt == 2 else "u") + str(bps)) if bps > 1 else "u1").newbyteorder(pg.e)
    tiled = 322 in pg.tags
    tw, th = (t(322), t(323)) if tiled else (W, t(278, H))
    offs = pg.tags[324 if tiled else 273]; cnts = pg.tags[325 if tiled else 279]
    offs = offs if isinstance(offs, tuple) else (offs,); cnts = cnts if isinstance(cnts, tuple) else (cnts,)
    out = np.zeros((H, W), np.float32)
    ntx = -(-W // tw)
    for k, (off, cnt) in enumerate(zip(offs, cnts)):
        pg.fh.seek(off); raw = pg.fh.read(cnt)
        buf = zlib.decompress(raw) if comp != 1 else raw
        rows = len(buf) // (tw * bps)
        a = np.frombuffer(buf, np.uint8)[:rows * tw * bps].reshape(rows, tw * bps)
        if pred == 3:   # floating point: byte-wise differences, bytes stored most significant plane first (big-endian)
            a = np.cumsum(a, axis=1, dtype=np.uint8).reshape(rows, bps, tw).transpose(0, 2, 1)
            v = np.ascontiguousarray(a).view(dt.newbyteorder(">")).reshape(rows, tw)
        else:
            v = a.copy().view(dt).reshape(rows, tw)
            if pred == 2:
                v = np.cumsum(v, axis=1, dtype=v.dtype)
        ty, tx = (k // ntx, k % ntx) if tiled else (k, 0)
        y0, x0 = ty * th, tx * tw
        hh, ww = min(rows, H - y0), min(tw, W - x0)
        if hh > 0 and ww > 0:
            out[y0:y0 + hh, x0:x0 + ww] = v[:hh, :ww]
    return out


def _js_tile(path):
    """A terrain tile stored as dem/<name>.js (the GeoTIFF in base64)."""
    txt = open(path, encoding="utf-8", errors="ignore").read()
    m = re.search(r'name:\s*"([^"]+)"', txt)
    i = txt.find('data: "')
    j = txt.find('"', i + 7) if i >= 0 else -1
    if j < 0:
        raise ValueError(f"{path}: no GeoTIFF data found")
    return (m.group(1) if m else os.path.basename(path)), io.BytesIO(base64.b64decode(txt[i + 7:j]))


class Tile:
    def __init__(self, path):
        name = str(path)
        if isinstance(path, str) and path.lower().endswith(".js"):
            name, path = _js_tile(path)
        pg = _Page(path)
        try:
            ras = _decode(pg)
            if ras is None:
                ras = tifffile.imread(path).astype(np.float32)
                ras = ras if ras.ndim == 2 else ras[..., 0]
            sc, tp = pg.tags[33550], pg.tags[33922]
            nd = pg.tags.get(42113)
            self.nd = float(nd.strip("\x00 ")) if nd else None
            gk = pg.tags.get(34735)
            if gk:
                for k in range(4, len(gk), 4):
                    if gk[k] == 3072 and gk[k + 3] != 32767:  # ProjectedCSTypeGeoKey
                        raise ValueError(f"{path}: projected (EPSG {gk[k + 3]}); reproject with gdalwarp -t_srs EPSG:4326")
        finally:
            pg.fh.close()
        self.ras = ras
        self.h, self.w = self.ras.shape
        # bounds: tie point + width × pixel size (no pixel-is-point shift)
        x0, y0 = tp[3] - tp[0] * sc[0], tp[4] + tp[1] * sc[1]
        self.minX, self.maxX = x0, x0 + self.w * sc[0]
        self.maxY, self.minY = y0, y0 - self.h * sc[1]
        self.name = name

    def covers(self, lat, lon):
        return (lon >= self.minX) & (lon <= self.maxX) & (lat >= self.minY) & (lat <= self.maxY)


def bad(z, nd):
    b = ~np.isfinite(z) | (z < -1000)
    return b | (z == nd) if nd is not None else b


def sample(tiles, lat, lon):
    """Bilinear height in the first tile containing each point; NaN outside every tile ."""
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    out = np.full(lat.shape, np.nan)
    todo = np.ones(lat.shape, dtype=bool)
    for t in tiles:
        m = todo & t.covers(lat, lon)
        if not m.any():
            continue
        la, lo = lat[m], lon[m]
        fx = (lo - t.minX) / (t.maxX - t.minX) * t.w - 0.5
        fy = (t.maxY - la) / (t.maxY - t.minY) * t.h - 0.5
        x0 = np.clip(np.floor(fx), 0, t.w - 1).astype(int)
        y0 = np.clip(np.floor(fy), 0, t.h - 1).astype(int)
        x1, y1 = np.minimum(t.w - 1, x0 + 1), np.minimum(t.h - 1, y0 + 1)
        ax, ay = np.clip(fx - x0, 0, 1), np.clip(fy - y0, 0, 1)
        v = [t.ras[y0, x0].astype(float), t.ras[y0, x1].astype(float), t.ras[y1, x0].astype(float), t.ras[y1, x1].astype(float)]
        B = [bad(z, t.nd) for z in v]
        anyb = B[0] | B[1] | B[2] | B[3]
        bil = (v[0] * (1 - ax) + v[1] * ax) * (1 - ay) + (v[2] * (1 - ax) + v[3] * ax) * ay
        ng = sum((~b).astype(int) for b in B)
        sg = sum(np.where(b, 0.0, z) for z, b in zip(v, B))
        with np.errstate(invalid="ignore", divide="ignore"):
            val = np.where(anyb, np.where(ng > 0, sg / np.maximum(ng, 1), np.nan), bil)
        done = np.isfinite(val)          # points with only bad pixels try the next tile
        idx = np.flatnonzero(m)
        out[idx[done]] = val[done]
        todo[idx[done]] = False
    return out


def pixel_m(tiles, pr):
    t = tiles[0]
    return min((t.maxX - t.minX) / t.w * pr.mLon, (t.maxY - t.minY) / t.h * pr.mLat)


def find_tiles(points, folders):
    """Terrain tiles in `folders` (.tif/.tiff, or base64 .js tiles) that cover any of the points (lat, lon).
    A .tif and a .js of the same tile are read once (the .tif)."""
    files = []
    for d in folders:
        if d and os.path.isdir(d):
            files += sorted(glob.glob(os.path.join(d, "*.tif")) + glob.glob(os.path.join(d, "*.tiff")))
            files += sorted(f for f in glob.glob(os.path.join(d, "*.js")) if os.path.basename(f) != "index.js")
    seen, out = set(), []
    for f in files:
        stem = re.sub(r"\.(tif|tiff|js)$", "", os.path.basename(f), flags=re.I)
        if stem in seen:
            continue
        b = _bounds_from_name(stem)
        if b and not any(b[0] <= la <= b[1] and b[2] <= lo <= b[3] for la, lo in points):
            continue        # Copernicus name says it is elsewhere: skip without reading
        try:
            t = Tile(f)
        except Exception:
            continue
        if any(t.minY <= la <= t.maxY and t.minX <= lo <= t.maxX for la, lo in points):
            out.append(t); seen.add(stem)
    return out


def _bounds_from_name(stem):
    m = re.search(r"_([NS])(\d\d)_00_([EW])(\d\d\d)_00", stem)
    if not m:
        return None
    lat = int(m.group(2)) * (1 if m.group(1) == "N" else -1)
    lon = int(m.group(4)) * (1 if m.group(3) == "E" else -1)
    return lat, lat + 1, lon, lon + 1


def copernicus_name(lat, lon):
    la, lo = math.floor(lat), math.floor(lon)
    return f"Copernicus_DSM_COG_10_{'N' if la >= 0 else 'S'}{abs(la):02d}_00_{'E' if lo >= 0 else 'W'}{abs(lo):03d}_00_DEM"


def needed_tiles(points):
    return sorted({copernicus_name(la, lo) for la, lo in points})


# Required attribution for Copernicus DEM GLO-30 data (and for any product made with it, e.g. reports and figures).
COPERNICUS_ATTRIBUTION = ("© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS "
                          "by the European Union and ESA; all rights reserved")


def download_copernicus(name, folder, log=print):
    """Copernicus GLO-30 tile from the public AWS bucket (free, no account), saved as <folder>/<name>.tif."""
    import urllib.request
    os.makedirs(folder, exist_ok=True)
    url = f"https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"
    out = os.path.join(folder, name + ".tif")
    log(f"Downloading terrain {name} (about 30 MB) … Copernicus DEM GLO-30: {COPERNICUS_ATTRIBUTION}")
    tmp = out + ".part"
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    os.replace(tmp, out)
    return out
