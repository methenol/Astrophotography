"""Frame discovery, raw FITS reading, CFA helpers and cosmetic correction."""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, asdict
from datetime import datetime

import cv2
import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter

CHANNEL_INDEX = {"R": 0, "G": 1, "B": 2}


@dataclass
class FrameInfo:
    path: str
    name: str
    object: str
    filter: str
    exptime: float
    gain: float
    temp: float
    date_obs: str
    timestamp: float
    bayer: str
    bias: float
    width: int
    height: int
    ra: float | None
    dec: float | None
    focallen: float
    pixsize: float

    def to_dict(self):
        return asdict(self)


def _parse_time(s: str) -> float:
    try:
        return datetime.fromisoformat(s.replace("Z", "")).timestamp()
    except Exception:
        return 0.0


def read_info(path: str) -> FrameInfo:
    h = fits.getheader(path)
    return FrameInfo(
        path=path,
        name=os.path.basename(path),
        object=str(h.get("OBJECT", "")).strip(),
        filter=str(h.get("FILTER", "")).strip(),
        exptime=float(h.get("EXPTIME", h.get("EXPOSURE", 0.0)) or 0.0),
        gain=float(h.get("GAIN", 0) or 0),
        temp=float(h.get("CCD-TEMP", 0) or 0),
        date_obs=str(h.get("DATE-OBS", "")),
        timestamp=_parse_time(str(h.get("DATE-OBS", ""))),
        bayer=str(h.get("BAYERPAT", "GRBG")).strip() or "GRBG",
        bias=float(h.get("BIAS", 0) or 0),
        width=int(h["NAXIS1"]),
        height=int(h["NAXIS2"]),
        ra=float(h["RA"]) if "RA" in h else None,
        dec=float(h["DEC"]) if "DEC" in h else None,
        focallen=float(h.get("FOCALLEN", 250) or 250),
        pixsize=float(h.get("XPIXSZ", 2.9) or 2.9),
    )


def discover(folder: str) -> list[FrameInfo]:
    """Find light frames in ``folder``, keep the dominant (filter, exposure, size) group."""
    paths = sorted(
        p for ext in ("*.fit", "*.fits", "*.fts", "*.FIT", "*.FITS")
        for p in glob.glob(os.path.join(folder, ext))
    )
    paths = sorted(set(paths))
    infos = []
    for p in paths:
        try:
            h = fits.getheader(p)
            if str(h.get("IMAGETYP", "Light")).strip().lower() not in ("light", "light frame", ""):
                continue
            if int(h.get("NAXIS", 0)) != 2:
                continue
            infos.append(read_info(p))
        except Exception:
            continue
    if not infos:
        return []
    groups: dict[tuple, list[FrameInfo]] = {}
    for fi in infos:
        groups.setdefault((fi.filter, fi.width, fi.height, fi.bayer), []).append(fi)
    best = max(groups.values(), key=lambda g: sum(f.exptime for f in g))
    return sorted(best, key=lambda f: f.timestamp or 0)


def read_raw(path: str, bias: float) -> np.ndarray:
    """Read a raw CFA frame as float32 with the black level removed."""
    data = fits.getdata(path).astype(np.float32)
    data -= bias
    np.maximum(data, 0, out=data)
    return data


# --------------------------------------------------------------------------- CFA

def cfa_channel_map(pattern: str, shape: tuple[int, int]) -> np.ndarray:
    """Return an int8 array with the colour index (0=R,1=G,2=B) of every CFA site."""
    pattern = pattern.upper()
    cmap = np.empty(shape, np.int8)
    for k, (dy, dx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        cmap[dy::2, dx::2] = CHANNEL_INDEX[pattern[k]]
    return cmap


def cfa_masks(pattern: str, shape: tuple[int, int]) -> np.ndarray:
    cmap = cfa_channel_map(pattern, shape)
    return np.stack([(cmap == c).astype(np.float32) for c in range(3)], axis=-1)


def superpixel(raw: np.ndarray, pattern: str) -> np.ndarray:
    """2x2 super-pixel debayer -> half resolution RGB (no interpolation)."""
    pattern = pattern.upper()
    h, w = raw.shape
    h2, w2 = h // 2 * 2, w // 2 * 2
    sites = {(0, 0): pattern[0], (0, 1): pattern[1], (1, 0): pattern[2], (1, 1): pattern[3]}
    out = np.zeros((h2 // 2, w2 // 2, 3), np.float32)
    cnt = [0, 0, 0]
    for (dy, dx), c in sites.items():
        ci = CHANNEL_INDEX[c]
        out[..., ci] += raw[dy:h2:2, dx:w2:2]
        cnt[ci] += 1
    for ci in range(3):
        out[..., ci] /= max(cnt[ci], 1)
    return out


_CV_CODE_CACHE: dict[str, int] = {}


def _opencv_bayer_code(pattern: str) -> int:
    """Empirically find the OpenCV edge-aware demosaic code for ``pattern``.

    OpenCV's Bayer naming is notoriously offset, so we test each code on a
    synthetic mosaic rather than trusting names.
    """
    if pattern in _CV_CODE_CACHE:
        return _CV_CODE_CACHE[pattern]
    cmap = cfa_channel_map(pattern, (16, 16))
    truth = np.array([1000, 5000, 20000], np.float32)
    mosaic = truth[cmap].astype(np.uint16)
    for code in (cv2.COLOR_BayerBG2RGB_EA, cv2.COLOR_BayerGB2RGB_EA,
                 cv2.COLOR_BayerRG2RGB_EA, cv2.COLOR_BayerGR2RGB_EA):
        rgb = cv2.cvtColor(mosaic, code)[4:12, 4:12].reshape(-1, 3).mean(0)
        if np.allclose(rgb, truth, rtol=0.02):
            _CV_CODE_CACHE[pattern] = code
            return code
    raise ValueError(f"Unsupported Bayer pattern {pattern}")


def demosaic(raw: np.ndarray, pattern: str) -> np.ndarray:
    """Edge-aware demosaic of a (bias-subtracted, float) CFA frame -> RGB float32."""
    code = _opencv_bayer_code(pattern.upper())
    u16 = np.clip(raw, 0, 65535).astype(np.uint16)
    return cv2.cvtColor(u16, code).astype(np.float32)


# ------------------------------------------------------------------- cosmetic

def build_defect_map(infos: list[FrameInfo], n_sample: int = 24, k_sigma: float = 7.0) -> np.ndarray:
    """Detect hot/cold pixels from the median of a sample of *unregistered* frames.

    Real sky features wander between frames (tracking drift and alt-az field
    rotation) while sensor defects stay put, so the temporal median keeps the
    defects.  A pixel is flagged when it deviates strongly from same-colour
    neighbours **and** is isolated (its immediate neighbours are not elevated),
    which protects the cores of stars that barely moved.
    """
    idx = np.linspace(0, len(infos) - 1, min(n_sample, len(infos))).round().astype(int)
    stack = np.stack([read_raw(infos[i].path, infos[i].bias) for i in np.unique(idx)])
    med = np.median(stack, axis=0)
    del stack
    h, w = med.shape
    defects = np.zeros((h, w), bool)
    for dy in (0, 1):
        for dx in (0, 1):
            plane = med[dy::2, dx::2]
            resid = plane - median_filter(plane, size=5, mode="reflect")
            sigma = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-6
            defects[dy::2, dx::2] = np.abs(resid) > k_sigma * sigma
    # isolation test on the full-resolution mosaic (8-neighbourhood, any colour)
    neigh = median_filter(med, footprint=np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], bool), mode="reflect")
    local_bg = median_filter(med, size=9, mode="reflect")
    elevated_neigh = (neigh - local_bg) > 0.25 * np.abs(med - local_bg)
    defects &= ~elevated_neigh
    return defects


def fix_defects(raw: np.ndarray, defects: np.ndarray | None) -> np.ndarray:
    """Replace defective CFA pixels with the median of same-colour neighbours."""
    if defects is None or not defects.any():
        return raw
    ys, xs = np.nonzero(defects)
    h, w = raw.shape
    vals = []
    for oy, ox in ((-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (-2, 2), (2, -2), (2, 2)):
        yy = np.clip(ys + oy, 0, h - 1)
        xx = np.clip(xs + ox, 0, w - 1)
        v = raw[yy, xx].copy()
        v[defects[yy, xx]] = np.nan
        vals.append(v)
    rep = np.nanmedian(np.stack(vals), axis=0)
    rep = np.where(np.isfinite(rep), rep, np.median(raw))
    raw[ys, xs] = rep
    return raw
