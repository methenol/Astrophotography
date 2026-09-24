"""Per-frame quality analysis, registration and rejection.

For every sub we measure star statistics (count, FWHM, elongation), sky
background and noise.  Frames are registered to the best frame by asterism
(triangle) matching followed by a RANSAC similarity refinement using every
matched star (handles alt-az field rotation).

Obstructions (trees, roofs, dew, passing clouds) are found photometrically:
each reference star that *should* be visible in a frame is looked up; the
flux ratio frame/reference is aggregated on a coarse tile grid.  Tiles where
stars dim or vanish become per-frame masks, so a frame that is 30% blocked by
a tree still contributes its clean 70%.  Whole frames are rejected by robust
(median/MAD) outlier tests plus a multivariate IsolationForest anomaly model.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import astroalign
import cv2
import numpy as np
import sep
from scipy.spatial import cKDTree
from sklearn.ensemble import IsolationForest

from .frames import FrameInfo, fix_defects, read_raw, superpixel

sep.set_extract_pixstack(3_000_000)
sep.set_sub_object_limit(4096)

MAX_STARS = 2500
_DEFECTS = None


def _init_worker(defects):
    global _DEFECTS
    _DEFECTS = defects
    cv2.setNumThreads(1)


def measure_stars(lum: np.ndarray, thresh: float = 5.0):
    """Detect and measure stars on a (half-res) luminance image with SEP.

    Returns (stars[N, 6] = x, y, flux, fwhm, elongation, peak ; bkg ; rms).
    """
    lum = np.ascontiguousarray(lum, dtype=np.float32)
    bkg = sep.Background(lum, bw=64, bh=64, fw=3, fh=3)
    sub = lum - bkg.back()
    rms = float(bkg.globalrms)
    try:
        objs = sep.extract(sub, thresh, err=rms, minarea=4, deblend_cont=0.005)
    except Exception:
        objs = sep.extract(sub, thresh * 2, err=rms, minarea=6, deblend_cont=0.01)
    if len(objs) == 0:
        return np.zeros((0, 6), np.float32), bkg, rms
    good = (objs["flag"] & ~0x01) == 0  # allow merged (crowded) objects, drop truncated etc.
    good &= objs["a"] > 0.3
    objs = objs[good]
    if len(objs) == 0:
        return np.zeros((0, 6), np.float32), bkg, rms
    r50, _ = sep.flux_radius(sub, objs["x"], objs["y"], 6.0 * objs["a"], 0.5,
                             normflux=objs["flux"], subpix=5)
    fwhm = 2.0 * r50  # exact for a Gaussian profile
    elong = objs["a"] / np.maximum(objs["b"], 1e-3)
    stars = np.stack([objs["x"], objs["y"], objs["flux"], fwhm, elong, objs["peak"]], axis=1)
    ok = np.isfinite(stars).all(1) & (fwhm > 0.5) & (fwhm < 30)
    stars = stars[ok]
    stars = stars[np.argsort(-stars[:, 2])][:MAX_STARS]
    return stars.astype(np.float32), bkg, rms


def analyse_frame(info: FrameInfo) -> dict:
    raw = read_raw(info.path, info.bias)
    raw = fix_defects(raw, _DEFECTS)
    sp = superpixel(raw, info.bayer)
    lum = sp.mean(axis=2)
    stars, bkg, rms = measure_stars(lum)
    # convert half-res coordinates to full-resolution CFA pixel coordinates
    if len(stars):
        stars[:, 0] = stars[:, 0] * 2 + 0.5
        stars[:, 1] = stars[:, 1] * 2 + 0.5
        stars[:, 3] *= 2
    sat_level = 0.9 * (65535 - info.bias)
    # use unsaturated, round-ish, well-measured stars for shape statistics
    shape_sel = stars[(stars[:, 5] < sat_level * 0.8) & (stars[:, 2] > 0)] if len(stars) else stars
    shape_sel = shape_sel[: max(10, len(shape_sel) // 2)]
    back = bkg.back()
    h, w = back.shape
    return {
        "name": info.name,
        "stars": stars,
        "n_stars": int(len(stars)),
        "fwhm": float(np.median(shape_sel[:, 3])) if len(shape_sel) >= 3 else float("nan"),
        "elongation": float(np.median(shape_sel[:, 4])) if len(shape_sel) >= 3 else float("nan"),
        "background": [float(np.median(sp[..., c])) for c in range(3)],
        "bg_gradient": float(np.percentile(back, 95) - np.percentile(back, 5)),
        "noise": rms,
        "saturated_frac": float((raw > sat_level).mean()),
    }


# -------------------------------------------------------------- registration

def _match_transform(src: np.ndarray, ref: np.ndarray, ref_tree: cKDTree):
    """Asterism match then refine with all stars. Returns (M 2x3, n_matched, rms)."""
    if len(src) < 6 or len(ref) < 6:
        return None, 0, float("nan")
    try:
        tf, _ = astroalign.find_transform(src[:80, :2], ref[:80, :2], max_control_points=60)
        M = tf.params[:2].astype(np.float64)
    except Exception:
        return None, 0, float("nan")
    for radius in (4.0, 2.0):
        proj = src[:, :2] @ M[:, :2].T + M[:, 2]
        d, j = ref_tree.query(proj, distance_upper_bound=radius)
        ok = np.isfinite(d)
        if ok.sum() < 6:
            return None, int(ok.sum()), float("nan")
        M2, inl = cv2.estimateAffinePartial2D(
            src[ok, :2].astype(np.float32), ref[j[ok], :2].astype(np.float32),
            method=cv2.RANSAC, ransacReprojThreshold=1.5, maxIters=4000, confidence=0.999)
        if M2 is None:
            return None, 0, float("nan")
        M = M2
    proj = src[:, :2] @ M[:, :2].T + M[:, 2]
    d, j = ref_tree.query(proj, distance_upper_bound=2.0)
    ok = np.isfinite(d)
    rms = float(np.sqrt(np.mean(d[ok] ** 2))) if ok.any() else float("nan")
    return M, int(ok.sum()), rms


def poly_terms(x, y, deg=3):
    return np.stack([x ** i * y ** j for i in range(deg + 1) for j in range(deg + 1 - i)], axis=-1)


def fit_distortion(src, ref_stars, M, width, height, deg=3):
    """Fit a polynomial *inverse* mapping reference -> frame pixel coordinates.

    Corrects residual optical distortion that a similarity transform cannot
    model (it grows with alt-az field rotation because stars cross different
    parts of the optics).  Returns (coefs[2, T], rms_before, rms_after) or None.
    """
    if M is None or len(src) < 40:
        return None
    proj = src[:, :2] @ M[:, :2].T + M[:, 2]
    tree = cKDTree(ref_stars[:, :2])
    d, j = tree.query(proj, distance_upper_bound=2.0)
    ok = np.isfinite(d)
    if ok.sum() < 40:
        return None
    ref_xy = ref_stars[j[ok], :2].astype(np.float64)
    frm_xy = src[ok, :2].astype(np.float64)
    nx = ref_xy[:, 0] / width * 2 - 1
    ny = ref_xy[:, 1] / height * 2 - 1
    X = poly_terms(nx, ny, deg)
    # predicted frame coordinates from the similarity model (for comparison)
    A = np.vstack([M, [0, 0, 1]])
    Ai = np.linalg.inv(A)[:2]
    sim_pred = ref_xy @ Ai[:, :2].T + Ai[:, 2]
    rms_before = float(np.sqrt(np.mean(np.sum((sim_pred - frm_xy) ** 2, 1))))
    keep = np.ones(len(X), bool)
    for _ in range(3):
        cx, *_ = np.linalg.lstsq(X[keep], frm_xy[keep, 0], rcond=None)
        cy, *_ = np.linalg.lstsq(X[keep], frm_xy[keep, 1], rcond=None)
        res = np.hypot(X @ cx - frm_xy[:, 0], X @ cy - frm_xy[:, 1])
        s = 1.4826 * np.median(res[keep]) + 1e-6
        keep = res < max(3 * s, 0.5)
    rms_after = float(np.sqrt(np.mean(res[keep] ** 2)))
    if rms_after >= rms_before * 0.97:
        return None
    return {"coefs": np.stack([cx, cy]), "deg": deg, "rms_before": rms_before, "rms_after": rms_after}


def _tile_grid(width: int, height: int, target: int = 360):
    nx = max(2, int(round(width / target)))
    ny = max(2, int(round(height / target)))
    return nx, ny


def _photometry(frame_stars, ref_stars, M, width, height, nx, ny):
    """Per-tile flux ratio frame/reference for reference stars expected in frame."""
    tile = np.full((ny, nx), np.nan, np.float32)
    if M is None or len(frame_stars) == 0:
        return float("nan"), tile
    # which reference stars fall inside this frame's footprint?
    A = np.vstack([M, [0, 0, 1]])
    Ainv = np.linalg.inv(A)[:2]
    ref_in_frame = ref_stars[:, :2] @ Ainv[:, :2].T + Ainv[:, 2]
    margin = 20
    inside = ((ref_in_frame[:, 0] > margin) & (ref_in_frame[:, 0] < width - margin) &
              (ref_in_frame[:, 1] > margin) & (ref_in_frame[:, 1] < height - margin))
    # only reference stars bright enough that they'd be detected in any decent frame
    bright = ref_stars[:, 2] >= np.percentile(ref_stars[:, 2], 50)
    sel = inside & bright
    if sel.sum() < 5:
        return float("nan"), tile
    proj = frame_stars[:, :2] @ M[:, :2].T + M[:, 2]
    tree = cKDTree(proj)
    d, j = tree.query(ref_stars[sel, :2], distance_upper_bound=3.0)
    ratio = np.zeros(sel.sum(), np.float32)
    ok = np.isfinite(d)
    ratio[ok] = frame_stars[j[ok], 2] / np.maximum(ref_stars[sel][ok, 2], 1e-6)
    ratio = np.clip(ratio, 0, 5)
    xs, ys = ref_stars[sel, 0], ref_stars[sel, 1]
    ix = np.clip((xs / width * nx).astype(int), 0, nx - 1)
    iy = np.clip((ys / height * ny).astype(int), 0, ny - 1)
    for ty in range(ny):
        for tx in range(nx):
            m = (ix == tx) & (iy == ty)
            if m.sum() >= 4:
                tile[ty, tx] = np.median(ratio[m])
    return float(np.median(ratio[ratio > 0])) if (ratio > 0).any() else 0.0, tile


def _robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = 1.4826 * np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad < 1e-9:
        mad = max(abs(med) * 0.02, 1e-9)
    return (x - med) / mad


def analyse(infos: list[FrameInfo], defects: np.ndarray | None, progress=None,
            workers: int | None = None, sensitivity: float = 1.0) -> dict:
    """Run the full per-frame analysis. ``sensitivity`` >1 rejects more aggressively."""
    workers = workers or max(1, min(8, (os.cpu_count() or 4) - 2))
    n = len(infos)
    results: list[dict] = [None] * n  # type: ignore
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(defects,)) as ex:
        for i, r in enumerate(ex.map(analyse_frame, infos, chunksize=2)):
            results[i] = r
            if progress:
                progress(i + 1, n, f"Measuring stars {i + 1}/{n}")

    width, height = infos[0].width, infos[0].height
    fwhm = np.array([r["fwhm"] for r in results])
    elong = np.array([r["elongation"] for r in results])
    nst = np.array([r["n_stars"] for r in results], float)

    # reference: many sharp, round stars
    score = nst / np.maximum(np.nan_to_num(fwhm, nan=99), 0.5) ** 2 / np.maximum(np.nan_to_num(elong, nan=9), 1)
    ref_idx = int(np.nanargmax(score))
    ref_stars = results[ref_idx]["stars"]
    ref_tree = cKDTree(ref_stars[:, :2])
    nx, ny = _tile_grid(width, height)

    for i, r in enumerate(results):
        if i == ref_idx:
            M, nm, rms = np.array([[1, 0, 0], [0, 1, 0]], float), len(ref_stars), 0.0
        else:
            M, nm, rms = _match_transform(r["stars"], ref_stars, ref_tree)
        r["transform"] = M
        r["n_matched"] = nm
        r["reg_rms"] = rms
        r["distortion"] = fit_distortion(r["stars"], ref_stars, M, width, height) if i != ref_idx else None
        if r["distortion"] is not None:
            r["reg_rms"] = r["distortion"]["rms_after"]
        t, tiles = _photometry(r["stars"], ref_stars, M, width, height, nx, ny)
        r["transparency_raw"] = t
        r["tiles"] = tiles
        if progress and (i % 10 == 0 or i == n - 1):
            progress(i + 1, n, f"Registering {i + 1}/{n}")

    return finalize_selection(results, ref_idx, (nx, ny), sensitivity)


def finalize_selection(results, ref_idx, grid, sensitivity=1.0) -> dict:
    """Decide rejections/weights/obstruction masks from measured features."""
    n = len(results)
    k = 1.0 / max(sensitivity, 0.1)
    t_raw = np.array([r["transparency_raw"] for r in results], float)
    t_med = np.nanmedian(t_raw[np.isfinite(t_raw) & (t_raw > 0)]) if np.any(t_raw > 0) else 1.0
    for r in results:
        tiles = r["tiles"] / t_med
        valid = np.isfinite(tiles)
        mask = np.ones_like(tiles, np.float32)
        if valid.sum() >= 3:
            hi = np.nanpercentile(tiles, 80)
            # a tile is obstructed if its stars are much dimmer than the frame's clear parts
            # or much dimmer than a typical frame
            obstructed = valid & ((tiles < 0.45 * hi) | (tiles < 0.35))
            mask[obstructed] = 0.0
            clear = valid & ~obstructed
            r["transparency"] = float(np.nanmedian(tiles[clear])) if clear.any() else 0.0
            r["obstructed_frac"] = float(obstructed.sum() / valid.sum())
        else:
            r["transparency"] = float(r["transparency_raw"] / t_med) if np.isfinite(r["transparency_raw"]) else float("nan")
            r["obstructed_frac"] = 0.0
        r["tile_mask"] = mask

    feats = {
        "fwhm": np.array([r["fwhm"] for r in results], float),
        "elongation": np.array([r["elongation"] for r in results], float),
        "n_stars": np.array([r["n_stars"] for r in results], float),
        "background": np.array([np.mean(r["background"]) for r in results], float),
        "noise": np.array([r["noise"] for r in results], float),
        "transparency": np.array([r["transparency"] for r in results], float),
        "obstructed": np.array([r["obstructed_frac"] for r in results], float),
        "reg_rms": np.array([r["reg_rms"] for r in results], float),
    }
    z = {key: _robust_z(v) for key, v in feats.items()}
    med = {key: np.nanmedian(v) for key, v in feats.items()}

    # multivariate anomaly model (unsupervised ML)
    X = np.column_stack([
        np.nan_to_num(z["fwhm"]), np.nan_to_num(z["elongation"]),
        np.nan_to_num(_robust_z(np.log1p(feats["n_stars"]))), np.nan_to_num(z["background"]),
        np.nan_to_num(z["noise"]), np.nan_to_num(z["transparency"]), feats["obstructed"] * 5,
    ])
    X = np.clip(X, -20, 20)
    anomaly = np.zeros(n)
    if n >= 20:
        iso = IsolationForest(n_estimators=300, random_state=0, contamination="auto").fit(X)
        s = -iso.score_samples(X)
        anomaly = _robust_z(s)

    for i, r in enumerate(results):
        reasons = []
        if r["transform"] is None:
            reasons.append("registration failed")
        if r["n_matched"] < 8 and r["transform"] is not None:
            reasons.append("too few matched stars")
        if feats["fwhm"][i] > med["fwhm"] * (1 + 0.35 * k) and z["fwhm"][i] > 3.5 * k:
            reasons.append(f"blurred (FWHM {feats['fwhm'][i]:.2f}px vs {med['fwhm']:.2f})")
        if feats["elongation"][i] > max(1.35, med["elongation"] * (1 + 0.2 * k)) and z["elongation"][i] > 3.5 * k:
            reasons.append(f"trailed stars (elongation {feats['elongation'][i]:.2f})")
        if feats["n_stars"][i] < med["n_stars"] * max(0.2, 1 - 0.5 * k):
            reasons.append(f"few stars ({int(feats['n_stars'][i])} vs {int(med['n_stars'])})")
        if np.isfinite(feats["transparency"][i]) and feats["transparency"][i] < 0.55 * min(1.0, 1 / k):
            reasons.append(f"clouds/haze (transparency {feats['transparency'][i]:.2f})")
        if feats["obstructed"][i] > 0.5 * k:
            reasons.append(f"obstructed ({feats['obstructed'][i] * 100:.0f}% of field)")
        if z["background"][i] > 6 * k and feats["background"][i] > med["background"] * 1.15:
            reasons.append("bright sky (moon/dawn/cloud glow)")
        if anomaly[i] > 6 * k:
            reasons.append("ML anomaly (IsolationForest)")
        r["reject_reasons"] = reasons
        r["accepted"] = not reasons
        r["anomaly"] = float(anomaly[i])

    # weights: inverse-variance of normalised signal x sharpness
    for r in results:
        if not r["accepted"]:
            r["weight"] = 0.0
            continue
        t = r["transparency"] if np.isfinite(r["transparency"]) and r["transparency"] > 0 else 1.0
        noise_rel = r["noise"] / med["noise"]
        sharp = (med["fwhm"] / r["fwhm"]) if np.isfinite(r["fwhm"]) and r["fwhm"] > 0 else 1.0
        r["weight"] = float(t ** 2 / noise_rel ** 2 * sharp)
    wmax = max((r["weight"] for r in results), default=1.0) or 1.0
    for r in results:
        r["weight"] = r["weight"] / wmax

    return {"frames": results, "ref_idx": ref_idx, "grid": grid,
            "medians": {k2: float(v) for k2, v in med.items()}}
