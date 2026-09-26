"""Exposures for ImageMM: the data products of Sec. 2 of arXiv:2501.03002.

ImageMM needs, for every exposure t: the coregistered, background-subtracted
image y(t), per-pixel variances v(t), a binary mask m(t) and the PSF f(t)
measured from the exposure's stars.  This module derives them from the raw
Seestar subs.

Each step is linear in the sky signal, so the paper's model y = f * x + noise
holds for the prepared exposure, with f the PSF measured on it:

* Demosaic: bilinear interpolation (a fixed linear filter per colour).  An
  edge-aware demosaic is non-linear and would break the convolution model.
* Registration: the analysis stage's similarity transform and 3rd-order
  distortion polynomial (reference -> frame coordinates), evaluated exactly per
  pixel, then refined: windowed centroids (SExtractor WINPOS, Bertin & Arnouts
  1996) of reference-catalogue stars measured on the registered exposure give
  residual offsets, a robust 3rd-order polynomial of those offsets is composed
  into the mapping.  Resampling: Lanczos-4 (linear).
* Photometric scale: per colour (extinction is colour dependent), the median
  ratio of aperture fluxes of isolated, unsaturated stars in the exposure and in
  the reference coadd, with local annulus backgrounds and apertures of 3 FWHM so
  the ratio is independent of the seeing.
* Background: the scaled exposure minus the reference leaves the exposure's own
  smooth sky deviation, fitted with a robust 2nd-order surface; the reference's
  sky model is then subtracted too, so the latent sky is 0 as ImageMM assumes.
* Variances: photon-transfer curve (Janesick 2007) of the registered data,
  var = c0 + c1 * level in ADU per channel (read + sky noise, shot noise), fitted
  on differences of consecutive exposures in star-free pixels, binned by level,
  with robust (MAD) variances.  It is measured after demosaicing and resampling,
  so their effect on the per-pixel variance is included.
* Masks: outside the footprint of the Lanczos support, saturated and defective
  raw pixels (spread by the demosaic and resampling supports), and obstructed
  parts of the frame.
* PSFs: per exposure and colour, an empirical PSF: cut-outs of isolated,
  unsaturated reference stars around their *reference* positions (so the PSF
  also carries the exposure's residual registration), shifted onto the pixel
  grid with an exact Fourier shift, normalised by aperture flux, combined with a
  per-pixel sigma-clipped mean, negative wings set to 0 and normalised.
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import sep

from .analysis import poly_terms
from .frames import cfa_masks, fix_defects, read_raw


# ----------------------------------------------------------------- demosaic
_K_RB = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 4
_K_G = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], np.float32) / 4


def bilinear_demosaic(raw: np.ndarray, pattern: str, origin: tuple[int, int] = (0, 0)) -> np.ndarray:
    """Bilinear demosaic of a CFA frame (or of a window starting at ``origin`` of it)."""
    oy, ox = origin
    masks = cfa_masks(pattern, (raw.shape[0] + 2, raw.shape[1] + 2))[oy % 2: oy % 2 + raw.shape[0],
                                                                       ox % 2: ox % 2 + raw.shape[1]]
    out = np.empty(raw.shape + (3,), np.float32)
    for c, k in ((0, _K_RB), (1, _K_G), (2, _K_RB)):
        out[..., c] = cv2.filter2D(raw * masks[..., c], -1, k, borderType=cv2.BORDER_CONSTANT)
        # at the frame border the neighbour sum is incomplete: normalise by the weights present
        norm = cv2.filter2D(masks[..., c], -1, k, borderType=cv2.BORDER_CONSTANT)
        out[..., c] /= np.maximum(norm, 1e-6)
    return out


# ----------------------------------------------------------------- geometry
def source_coords(fr: dict, W0: int, H0: int, xs: np.ndarray, ys: np.ndarray,
                  refine: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Frame (source) pixel coordinates of reference-grid points (xs, ys).

    ``refine``: residual offsets d(p) (reference coordinates) from ``refine_registration``;
    the frame content seen at reference p + d(p) belongs at p, so the mapping is
    evaluated at p + d(p)."""
    xs = np.asarray(xs, np.float64)
    ys = np.asarray(ys, np.float64)
    if refine is not None:
        T = poly_terms(xs / W0 * 2 - 1, ys / H0 * 2 - 1, refine["deg"])
        xs, ys = xs + T @ refine["coefs"][0], ys + T @ refine["coefs"][1]
    dist = fr.get("distortion")
    if dist is not None:
        T = poly_terms(xs / W0 * 2 - 1, ys / H0 * 2 - 1, dist["deg"])
        return T @ dist["coefs"][0], T @ dist["coefs"][1]
    A = np.vstack([np.asarray(fr["transform"], np.float64), [0, 0, 1]])
    Ai = np.linalg.inv(A)[:2]
    return Ai[0, 0] * xs + Ai[0, 1] * ys + Ai[0, 2], Ai[1, 0] * xs + Ai[1, 1] * ys + Ai[1, 2]


def window_maps(fr, W0, H0, y0, y1, x0, x1, refine=None, rows: int = 128):
    """Float32 remap maps for the reference-grid window [y0, y1) x [x0, x1), evaluated
    exactly at every pixel (in blocks of ``rows`` rows to bound memory)."""
    mx = np.empty((y1 - y0, x1 - x0), np.float32)
    my = np.empty_like(mx)
    xs = np.arange(x0, x1, dtype=np.float64)
    for a in range(y0, y1, rows):
        b = min(a + rows, y1)
        xx, yy = np.meshgrid(xs, np.arange(a, b, dtype=np.float64))
        sx, sy = source_coords(fr, W0, H0, xx.ravel(), yy.ravel(), refine)
        mx[a - y0:b - y0] = sx.reshape(xx.shape)
        my[a - y0:b - y0] = sy.reshape(xx.shape)
    return mx, my


LANCZOS_R = 4          # half-width of the Lanczos-4 kernel (8 taps)


def warp_window(raw: np.ndarray, sat: np.ndarray, rep: np.ndarray, pattern: str, mx: np.ndarray, my: np.ndarray,
                rep_tol: float = 0.5):
    """Demosaic the part of the raw frame the window needs and resample it.

    ``sat``: saturated raw pixels (their values are wrong by an unknown amount: every
    output pixel whose demosaic + Lanczos support touches one is masked); ``rep``:
    defective (hot) raw pixels already repaired from the median of their 8 same-colour
    neighbours, a good estimate on smooth sky (error ~0.45 sigma): an output pixel is
    masked only where they carry more than ``rep_tol`` of its interpolation weight (it is
    then mostly an interpolation, not a measurement); the weights are the bilinear
    demosaic weights of the defects carried through the resampling.
    Returns (rgb, valid, hard) on the window: ``hard`` = full Lanczos support inside the
    frame and no saturated pixel in the support (for star measurements, where repaired
    pixels are acceptable estimates); ``valid`` = hard and not dominated by repaired pixels
    (the ImageMM mask m(t))."""
    H, W = raw.shape
    m = 3 + LANCZOS_R
    sx0 = int(max(0, math.floor(np.nanmin(mx)) - m)) // 2 * 2
    sy0 = int(max(0, math.floor(np.nanmin(my)) - m)) // 2 * 2
    sx1 = int(min(W, math.ceil(np.nanmax(mx)) + m + 1))
    sy1 = int(min(H, math.ceil(np.nanmax(my)) + m + 1))
    if sx1 <= sx0 + 2 * m or sy1 <= sy0 + 2 * m:
        return None, None, None
    rgb = bilinear_demosaic(raw[sy0:sy1, sx0:sx1], pattern, (sy0, sx0))
    lx, ly = mx - sx0, my - sy0
    out = cv2.remap(rgb, lx, ly, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    inside = ((mx >= 1 + LANCZOS_R) & (mx <= W - 2 - LANCZOS_R) &
              (my >= 1 + LANCZOS_R) & (my <= H - 2 - LANCZOS_R))
    valid = inside
    s_ = sat[sy0:sy1, sx0:sx1]
    if s_.any():
        b = cv2.dilate(s_.astype(np.uint8), np.ones((3, 3), np.uint8))                    # demosaic support
        b = cv2.dilate(b, np.ones((2 * LANCZOS_R + 2,) * 2, np.uint8))                     # Lanczos support
        valid = valid & (cv2.remap(b, lx, ly, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=1) == 0)
    hard = valid.copy()
    r_ = rep[sy0:sy1, sx0:sx1]
    if r_.any():
        wdem = bilinear_demosaic(r_.astype(np.float32), pattern, (sy0, sx0))                # demosaic weight of defects
        wout = cv2.remap(wdem, lx, ly, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid = valid & (wout.max(-1) <= rep_tol)
    return out, valid, hard


# ----------------------------------------------------------------- stars
def star_catalog(img: np.ndarray, sat: float, fwhm: float, thresh: float = 10.0):
    """Sources of a background-subtracted RGB coadd (luminance): WINPOS centroids, flux,
    peak, flags, and for each the distance to its nearest detected neighbour (3-sigma
    detections) and the flux of the brightest neighbour within 8 FWHM relative to it, so
    that every measurement can apply its own isolation criterion (``select``).  All
    3-sigma detections are kept for star masks."""
    from scipy.spatial import cKDTree
    L = np.ascontiguousarray(img.mean(-1), np.float32)
    bkg = sep.Background(L, bw=64, bh=64)
    sub = L - bkg.back()
    rms = bkg.globalrms
    allobj = sep.extract(sub, 3.0, err=rms, minarea=5)
    objs = sep.extract(sub, thresh, err=rms, minarea=5)
    wx, wy, wflag = sep.winpos(sub, objs["x"], objs["y"], np.full(len(objs), fwhm / 2.3548))
    ix = np.clip(np.round(wx).astype(int), 0, img.shape[1] - 1)
    iy = np.clip(np.round(wy).astype(int), 0, img.shape[0] - 1)
    peak_rgb = img[iy, ix].max(-1)
    tree = cKDTree(np.stack([allobj["x"], allobj["y"]], 1))
    d, j = tree.query(np.stack([objs["x"], objs["y"]], 1), k=2)
    nn = d[:, 1]                                        # [:, 0] is the source itself
    near = tree.query_ball_point(np.stack([objs["x"], objs["y"]], 1), 8 * fwhm)
    nflux = np.array([max([allobj["flux"][q] for q in lst if np.hypot(allobj["x"][q] - x0, allobj["y"][q] - y0) > 1.0],
                          default=0.0) for lst, x0, y0 in zip(near, objs["x"], objs["y"])])
    good = (objs["flag"] == 0) & (wflag == 0) & (peak_rgb < 0.5 * sat)
    return {"x": wx[good], "y": wy[good], "flux": objs["flux"][good], "peak": objs["peak"][good],
            "nn": nn[good], "nflux": nflux[good] / np.maximum(objs["flux"][good], 1e-12),
            "all_x": allobj["x"], "all_y": allobj["y"], "all_a": allobj["a"], "all_flux": allobj["flux"],
            "rms": rms}


def select(cat: dict, min_dist: float, max_nflux: float = np.inf) -> dict:
    """Catalogue subset whose nearest neighbour is farther than ``min_dist`` pixels, or whose
    neighbours within 8 FWHM are all fainter than ``max_nflux`` x the star."""
    ok = (cat["nn"] > min_dist) | (cat["nflux"] < max_nflux)
    return {k: (v[ok] if isinstance(v, np.ndarray) and len(v) == len(ok) and not k.startswith("all_") else v)
            for k, v in cat.items()}


def star_mask(shape, cat, fwhm: float, grow: float = 3.0) -> np.ndarray:
    """Pixels within ``grow`` FWHM (and 3 isophotal semi-axes) of any detected source."""
    m = np.zeros(shape, np.uint8)
    for x, y, a in zip(cat["all_x"], cat["all_y"], cat["all_a"]):
        cv2.circle(m, (int(round(x)), int(round(y))), int(math.ceil(max(grow * fwhm, 3 * a))), 1, -1)
    return m.astype(bool)


def refine_registration(L: np.ndarray, hard: np.ndarray, cat: dict, fwhm: float, W0: int, H0: int,
                        max_deg: int = 3, min_snr: float = 20.0) -> dict | None:
    """Residual registration offsets of a registered exposure (luminance L): WINPOS
    centroids of catalogue stars minus their reference positions, for stars with an
    exposure SNR >= ``min_snr`` and no neighbour within 3 FWHM.  Offsets are fitted by
    weighted least squares (weights SNR^2: a centroid's variance scales as 1/SNR^2),
    3-sigma clipped, with a polynomial whose degree (0 .. max_deg) minimises the AIC."""
    bkg = sep.Background(np.ascontiguousarray(L, np.float32), bw=64, bh=64)
    sub = np.ascontiguousarray(L - bkg.back(), np.float32)
    c = select(cat, 3 * fwhm)
    x, y = c["x"], c["y"]
    ix, iy = np.round(x).astype(int), np.round(y).astype(int)
    r = int(math.ceil(3 * fwhm))
    ok = (ix > r) & (iy > r) & (ix < L.shape[1] - r - 1) & (iy < L.shape[0] - r - 1)
    k = max(1, int(round(fwhm)))
    ok[ok] &= np.array([hard[j - k:j + k + 1, i - k:i + k + 1].all() for i, j in zip(ix[ok], iy[ok])], bool)
    if ok.sum() < 10:
        return None
    x, y = x[ok], y[ok]
    fl, fe, _ = sep.sum_circle(sub, x, y, 1.5 * fwhm, err=bkg.globalrms, mask=~hard)
    snr = fl / np.maximum(fe, 1e-12)
    s = snr >= min_snr
    if s.sum() < 10:
        return None
    x, y, snr = x[s], y[s], snr[s]
    wx, wy, flg = sep.winpos(sub, x, y, np.full(len(x), fwhm / 2.3548), mask=~hard)
    g = flg == 0
    x, y, snr, dx, dy = x[g], y[g], snr[g], (wx - x)[g], (wy - y)[g]
    w = snr ** 2
    u, v = x / W0 * 2 - 1, y / H0 * 2 - 1
    best = None
    for deg in range(0, max_deg + 1):
        Tm = poly_terms(u, v, deg)
        p = Tm.shape[1]
        if len(x) < 3 * p:
            break
        keep = np.ones(len(x), bool)
        for _ in range(10):
            sw = np.sqrt(w[keep])[:, None]
            cx = np.linalg.lstsq(Tm[keep] * sw, dx[keep] * sw[:, 0], rcond=None)[0]
            cy = np.linalg.lstsq(Tm[keep] * sw, dy[keep] * sw[:, 0], rcond=None)[0]
            res2 = ((Tm @ cx - dx) ** 2 + (Tm @ cy - dy) ** 2) * w
            s2 = np.sum(res2[keep]) / max(2 * (keep.sum() - p), 1)
            new = res2 < 9 * 2 * s2
            if (new == keep).all():
                break
            keep = new
        best = best or []
        best.append((deg, cx, cy, keep, s2, p))
    if not best:
        return None
    # AIC with the noise scale of the most flexible model (sigma^2 per unit weight)
    s2_ref, kp_ref = best[-1][4], best[-1][3]          # common inlier set and noise scale
    aic = [np.sum((((poly_terms(u, v, d) @ cx - dx) ** 2 + (poly_terms(u, v, d) @ cy - dy) ** 2) * w)[kp_ref]) / s2_ref
           + 2 * 2 * p for d, cx, cy, _, _, p in best]
    deg, cx, cy, keep, s2, p = best[int(np.argmin(aic))]
    Tm = poly_terms(u, v, deg)
    res = np.hypot(Tm @ cx - dx, Tm @ cy - dy)
    ww = w[keep] / w[keep].sum()
    return {"coefs": np.stack([cx, cy]), "deg": deg, "n": int(keep.sum()),
            "rms_before": float(np.sqrt(np.sum(ww * (dx[keep] ** 2 + dy[keep] ** 2)))),
            "rms_after": float(np.sqrt(np.sum(ww * res[keep] ** 2))),
            "noise_rms": float(np.sqrt(2 * s2 / np.mean(w[keep])))}


def compose_refine(a: dict | None, b: dict | None, W0: int, H0: int) -> dict | None:
    """Offsets of two successive refinements (a applied first): d(p) = a(p) + b(p) to first
    order in these sub-pixel offsets; polynomials of different degree are summed term by
    term (poly_terms orders the terms x^i y^j by i, then j)."""
    if a is None:
        return b
    if b is None:
        return a
    deg = max(a["deg"], b["deg"])
    full = [(i, j) for i in range(deg + 1) for j in range(deg + 1 - i)]

    def expand(r):
        own = [(i, j) for i in range(r["deg"] + 1) for j in range(r["deg"] + 1 - i)]
        out = np.zeros((2, len(full)))
        for q, t in enumerate(own):
            out[:, full.index(t)] = r["coefs"][:, q]
        return out
    return {"coefs": expand(a) + expand(b), "deg": deg, "n": b["n"], "rms_before": a["rms_before"],
            "rms_after": b["rms_after"], "noise_rms": b.get("noise_rms")}


def aperture_ratio(img: np.ndarray, ref: np.ndarray, hard: np.ndarray, cat: dict, radius: float,
                   min_snr: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel photometric scale img / ref: aperture fluxes (radius ~3 FWHM, so the
    ratio does not depend on the seeing) with local annulus backgrounds, of stars with no
    neighbour brighter than 1 % of them inside aperture + annulus and an exposure SNR >=
    ``min_snr``; inverse-variance weighted mean of the ratios, 3-sigma clipped.
    Returns (scale, standard error) per channel."""
    c = select(cat, radius + 8, max_nflux=0.01)
    x, y = c["x"], c["y"]
    ix, iy = np.round(x).astype(int), np.round(y).astype(int)
    R = int(math.ceil(radius + 8))
    ok = (ix > R) & (iy > R) & (ix < img.shape[1] - R - 1) & (iy < img.shape[0] - R - 1)
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    rr = np.hypot(yy, xx)
    ap, ann = rr <= radius, (rr >= radius + 3) & (rr <= radius + 8)
    for j in np.nonzero(ok)[0]:
        w = hard[iy[j] - R:iy[j] + R + 1, ix[j] - R:ix[j] + R + 1]
        ok[j] = w[ap].all() and w[ann].mean() >= 0.5
    out, err = np.full(img.shape[2], np.nan), np.full(img.shape[2], np.nan)
    if ok.sum() < 5:
        return out, err
    x, y = x[ok], y[ok]
    for ch in range(img.shape[2]):
        a = np.ascontiguousarray(img[..., ch], np.float32)
        b = np.ascontiguousarray(ref[..., ch], np.float32)
        ea = sep.Background(a, bw=64, bh=64, mask=~hard).globalrms
        eb = sep.Background(b, bw=64, bh=64).globalrms
        fa, sa, _ = sep.sum_circle(a, x, y, radius, err=ea, bkgann=(radius + 3, radius + 8), mask=~hard, gain=None)
        fb, sb, _ = sep.sum_circle(b, x, y, radius, err=eb, bkgann=(radius + 3, radius + 8))
        q = fa / np.where(fb > 0, fb, np.nan)
        sq = np.abs(q) * np.sqrt((sa / np.maximum(np.abs(fa), 1e-12)) ** 2 + (sb / np.maximum(np.abs(fb), 1e-12)) ** 2)
        good = np.isfinite(q) & (fa / np.maximum(sa, 1e-12) >= min_snr)
        q, sq = q[good], sq[good]
        if len(q) < 3:
            continue
        keep = np.ones(len(q), bool)
        for _ in range(10):
            wq = 1 / sq[keep] ** 2
            mu = np.sum(wq * q[keep]) / np.sum(wq)
            new = np.abs(q - mu) <= 3 * np.maximum(sq, 1e-12) * max(1.0, np.sqrt(np.sum(wq * (q[keep] - mu) ** 2) / max(keep.sum() - 1, 1)))
            if (new == keep).all():
                break
            keep = new
        wq = 1 / sq[keep] ** 2
        out[ch] = np.sum(wq * q[keep]) / np.sum(wq)
        chi2 = np.sum(wq * (q[keep] - out[ch]) ** 2) / max(keep.sum() - 1, 1)
        err[ch] = np.sqrt(max(chi2, 1.0) / np.sum(wq))
    return out, err


def fourier_shift(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Exact (band-limited) sub-pixel shift of a 2-D array by (dy, dx)."""
    h, w = img.shape
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.rfftfreq(w)[None, :]
    return np.fft.irfft2(np.fft.rfft2(img) * np.exp(-2j * np.pi * (ky * dy + kx * dx)), s=(h, w))


def empirical_psf(img: np.ndarray, valid: np.ndarray, cat: dict, half: int, clip: float = 3.0,
                  min_stars: int = 10, return_error: bool = False, nsig: float = 2.0):
    """Empirical PSF of one background-subtracted channel.

    Cut-outs of isolated, unsaturated catalogue stars around their reference positions,
    local residual sky removed (annulus median), shifted onto the pixel grid by an exact
    Fourier shift and normalised by aperture flux, are combined by a per-pixel sigma-
    clipped weighted mean.  Weights are flux^2: a flux-normalised cut-out of a sky-limited
    star has variance sigma_sky^2 / flux^2.

    The mean is noisy in the far wings, and clipping its negative pixels to zero before
    normalising would add a positive pedestal there (on M 27 subs it put 20-40 % of the
    flux beyond 2 FWHM).  So the kernel is cut at the support radius where the azimuthally
    averaged profile is no longer significant (< ``nsig`` times its standard error), at
    least 2 FWHM; only then are the few remaining negative pixels set to 0 and the kernel
    normalised.  ``return_error``: also return (unclipped mean, per-pixel standard error,
    support radius) for model fitting.
    Returns (PSF of (2 half + 1)^2 pixels with unit sum, number of stars used[, extras])."""
    pad = half + 8
    cuts, wts = [], []
    c = select(cat, 2 * pad, max_nflux=1e-3)
    ap = np.hypot(*np.mgrid[-half:half + 1, -half:half + 1]) <= half
    for x, y in zip(c["x"], c["y"]):
        ix, iy = int(round(x)), int(round(y))
        if ix - pad < 0 or iy - pad < 0 or ix + pad + 1 > img.shape[1] or iy + pad + 1 > img.shape[0]:
            continue
        if not valid[iy - pad:iy + pad + 1, ix - pad:ix + pad + 1].all():
            continue
        cc = img[iy - pad:iy + pad + 1, ix - pad:ix + pad + 1].astype(np.float64)
        yy, xx = np.mgrid[-pad:pad + 1, -pad:pad + 1]
        rr = np.hypot(yy - (y - iy), xx - (x - ix))
        cc = cc - np.median(cc[(rr > half + 2) & (rr <= half + 8)])          # local residual sky
        cc = fourier_shift(cc, -(y - iy), -(x - ix))                        # star centre -> pixel centre
        cc = cc[8:-8, 8:-8]
        flux = cc[ap].sum()
        if flux <= 0:
            continue
        cuts.append(cc / flux)
        wts.append(flux ** 2)
    if len(cuts) < min_stars:
        return (None, len(cuts), None) if return_error else (None, len(cuts))
    S = np.stack(cuts)
    w = np.array(wts)[:, None, None] * np.ones_like(S)
    keep = np.ones_like(S, bool)
    for _ in range(10):
        sw = (w * keep).sum(0)
        mu = (S * w * keep).sum(0) / np.maximum(sw, 1e-300)
        # weighted scatter -> standard error of the weighted mean
        var = (w * keep * (S - mu) ** 2).sum(0) / np.maximum(sw, 1e-300)
        neff = sw ** 2 / np.maximum((w ** 2 * keep).sum(0), 1e-300)
        sd = np.sqrt(var * neff / np.maximum(neff - 1, 1))
        new = np.abs(S - mu) <= clip * np.maximum(sd, 1e-12)
        if (new == keep).all():
            break
        keep = new
    se = sd / np.sqrt(np.maximum(neff, 1))
    # support radius: first 1-px annulus (beyond 2 FWHM) whose mean is not significant
    n = mu.shape[0]
    rr = np.hypot(*np.mgrid[:n, :n] - n // 2)
    fw = 2 * np.sqrt(max((mu * rr ** 2).sum() / max(mu.sum(), 1e-12), 0) / 2) * 1.1774   # 2nd-moment FWHM
    rsup = float(half)
    for k in range(1, half + 1):
        ring = (rr >= k - 0.5) & (rr < k + 0.5)
        m_ = mu[ring].mean()
        e_ = np.sqrt((se[ring] ** 2).sum()) / ring.sum()
        if k >= 2 * fw and m_ < nsig * e_:
            rsup = k - 0.5
            break
    psf = np.where(rr <= rsup, mu, 0.0)
    psf = np.clip(psf, 0, None)
    psf = (psf / psf.sum()).astype(np.float32)
    if return_error:
        return psf, len(cuts), {"mean": mu.astype(np.float32), "se": se.astype(np.float32), "support": rsup}
    return psf, len(cuts)


# ----------------------------------------------------------------- Moffat model
def moffat_image(p, size: int, sub: int = 9) -> np.ndarray:
    """Elliptical Moffat profile (Moffat 1969; elliptical form as in Trujillo et al. 2001)
    I = A [1 + (x'/a1)^2 + (y'/a2)^2]^-beta, (x', y') rotated by theta about (x0, y0)
    relative to the centre pixel, integrated over each pixel (sub x sub samples)."""
    A, x0, y0, a1, a2, theta, beta = p
    c = size // 2
    o = (np.arange(size * sub) + 0.5) / sub - 0.5 - c
    yy, xx = np.meshgrid(o - y0, o - x0, indexing="ij")
    ct, st = np.cos(theta), np.sin(theta)
    u, v = ct * xx + st * yy, -st * xx + ct * yy
    img = A * (1 + (u / a1) ** 2 + (v / a2) ** 2) ** (-beta)
    return img.reshape(size, sub, size, sub).mean((1, 3))


def fit_moffat(psf: np.ndarray, err: np.ndarray, ee: float = 0.995, max_half: int | None = None) -> tuple[np.ndarray, dict]:
    """Weighted least-squares fit of the pixel-integrated elliptical Moffat to an empirical
    PSF (the unclipped mean) with per-pixel standard errors ``err``.

    The model kernel extends to the radius holding ``ee`` of its analytic flux
    (enclosed energy within the ellipse of scaled radius R: 1 - (1 + R^2)^(1 - beta)), capped
    at ``max_half`` (default: twice the fitted cut-out's half-size); it is normalised to unit
    sum over that extent and the enclosed fraction is reported (``ee_kernel``)."""
    from scipy.optimize import least_squares
    n = psf.shape[0]
    c = n // 2
    max_half = max_half or 2 * c
    yy, xx = np.mgrid[:n, :n] - c
    m0 = max(psf.sum(), 1e-12)
    sx = np.sqrt(max((psf * xx ** 2).sum() / m0, 0.25))
    fw = 2.3548 * sx
    p0 = [psf.max(), 0.0, 0.0, fw, fw, 0.0, 2.5]
    w = 1 / np.maximum(err, 1e-6 * psf.max())

    def resid(p):
        return ((moffat_image(p, n) - psf) * w).ravel()

    lo = [0, -2, -2, 0.1, 0.1, -np.pi, 1.01]
    hi = [np.inf, 2, 2, 10 * n, 10 * n, np.pi, 30]
    sol = least_squares(resid, p0, bounds=(lo, hi), x_scale="jac")
    A, x0, y0, a1, a2, th, beta = sol.x
    R = np.sqrt(np.exp(np.log(1 - ee) / (1 - beta)) - 1) if beta > 1.0001 else np.inf
    half = int(min(max_half, np.ceil(R * max(a1, a2) + max(abs(x0), abs(y0)))))
    model = moffat_image(sol.x, 2 * half + 1)
    total = np.pi * a1 * a2 * A / (beta - 1)
    info = {"A": A, "x0": x0, "y0": y0, "alpha1": a1, "alpha2": a2, "theta": th, "beta": beta,
            "fwhm1": 2 * a1 * np.sqrt(2 ** (1 / beta) - 1), "fwhm2": 2 * a2 * np.sqrt(2 ** (1 / beta) - 1),
            "chi2_red": float(np.sum(sol.fun ** 2) / max(sol.fun.size - 7, 1)), "half": half,
            "ee_kernel": float(model.sum() / total)}
    return (model / model.sum()).astype(np.float32), info


# ----------------------------------------------------------------- the exposure set
class ExposureSet:
    """The prepared exposures of one session (see module docstring).

    ``prepare()`` makes one pass over the subs (registration refinement, photometric
    scale, background, PSFs, photon-transfer statistics); ``window()`` then yields
    y(t), v(t), m(t) of every exposure for any window of the reference grid."""

    def __init__(self, infos, analysis, defects, ref: np.ndarray, sat: float, workers: int | None = None):
        frames = analysis["frames"]
        self.items = [(info, fr) for info, fr in zip(infos, frames) if fr["accepted"] and fr["weight"] > 0]
        self.info0 = infos[0]
        self.W0, self.H0 = infos[0].width, infos[0].height
        self.pattern = infos[0].bayer
        self.defects = defects if defects is not None else np.zeros((self.H0, self.W0), bool)
        self.sat = float(sat)                     # saturation in bias-subtracted ADU
        self.ref = ref.astype(np.float32)         # reference coadd on the reference (1x) grid
        self.workers = workers or max(1, min(6, (os.cpu_count() or 4) // 2))
        fw = np.array([fr["fwhm"] for _, fr in self.items], float)
        self.fwhm_max = float(np.nanmax(fw))
        self.fwhm_med = float(np.nanmedian(fw))
        self.params: list[dict] = []
        self.ptc: dict | None = None

    # ------------------------------------------------------------- per frame
    def _raw(self, info):
        """Bias-subtracted raw frame with defects repaired (same-colour neighbour median),
        its saturated pixels, and the repaired ones."""
        raw = read_raw(info.path, info.bias)
        sat = raw >= 0.95 * self.sat
        raw = fix_defects(raw, self.defects)
        return raw, sat, self.defects

    def _obstruction(self, fr, y0, y1, x0, x1):
        tm = fr.get("tile_mask")
        if tm is None or (tm >= 0.5).all():
            return None
        full = cv2.resize(tm.astype(np.float32), (self.W0, self.H0), interpolation=cv2.INTER_LINEAR)
        return full[y0:y1, x0:x1] >= 0.5

    def register(self, k: int, refine=None):
        info, fr = self.items[k]
        raw, sat, rep = self._raw(info)
        mx, my = window_maps(fr, self.W0, self.H0, 0, self.H0, 0, self.W0, refine)
        y, valid, hard = warp_window(raw, sat, rep, self.pattern, mx, my)
        obs = self._obstruction(fr, 0, self.H0, 0, self.W0)
        if obs is not None:
            valid &= obs
            hard &= obs
        return y, valid, hard

    def prepare_one(self, k: int) -> dict:
        from .stacking import eval_surface, fit_smooth_surface
        info, fr = self.items[k]
        fwhm = float(fr["fwhm"]) if np.isfinite(fr["fwhm"]) else self.fwhm_max
        refine = None
        y, valid, hard = self.register(k)
        for _ in range(3):                                   # re-measure after each correction
            d = refine_registration(y.mean(-1), hard, self.cat, fwhm, self.W0, self.H0)
            if d is None:
                break
            refine = compose_refine(refine, d, self.W0, self.H0)
            y, valid, hard = self.register(k, refine)
            if d["rms_before"] < 1.5 * d["noise_rms"]:      # the offsets are already at the noise level
                break
        final = refine_registration(y.mean(-1), hard, self.cat, fwhm, self.W0, self.H0)
        T, T_err = aperture_ratio(y, self.ref, hard, self.cat, radius=3.0 * max(fwhm, self.fwhm_ref))
        if not np.all(np.isfinite(T)):
            # no photometric scale in some channel (too few isolated stars at SNR >= 50): the
            # exposure cannot be placed in the model; usable() leaves it out
            return {"refine": refine, "residual": final, "T": T, "T_err": T_err, "surf": None, "psf": [None] * 3,
                    "psf_err": [None] * 3, "psf_stars": [0, 0, 0], "fwhm": fwhm, "valid_frac": float(valid.mean()),
                    "_y": None}
        e = y / T[None, None, :] - self.ref
        coefs = fit_smooth_surface(e, valid & ~self.smask, deg=2)
        surf = eval_surface(coefs, self.H0, self.W0)
        ybs = y / T[None, None, :] - surf - self.sky_ref
        half = int(math.ceil(3.5 * max(fwhm, self.fwhm_ref)))
        psfs, nst, errs = [], [], []
        for c in range(3):
            p, ns, pe = empirical_psf(ybs[..., c], hard, self.cat, half, return_error=True)
            psfs.append(p)
            nst.append(ns)
            errs.append(pe)
        return {"refine": refine, "residual": final, "T": T, "T_err": T_err, "surf": coefs, "psf": psfs, "psf_err": errs,
                "psf_stars": nst, "fwhm": fwhm,
                "valid_frac": float(valid.mean()), "_y": y, "_valid": valid, "_surf": surf}

    # ------------------------------------------------------------- the pass
    def prepare(self, progress=None, cancel=None):
        from .postprocess import background_model
        self.sky_ref, self.sky_info = background_model(self.ref, "poly", 2)
        ref_bs = self.ref - self.sky_ref
        self.fwhm_ref = self.fwhm_med
        self.cat = star_catalog(ref_bs, self.sat, self.fwhm_ref)
        self.smask = star_mask(self.ref.shape[:2], self.cat, self.fwhm_ref)
        n = len(self.items)
        self.params = [None] * n
        acc = {"X": [], "v": [], "n": []}        # photon-transfer bins
        prev = None
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futs = [ex.submit(self.prepare_one, k) for k in range(min(n, self.workers))]
            nxt = len(futs)
            for k in range(n):
                if cancel and cancel():
                    raise RuntimeError("cancelled")
                p = futs[k].result()
                futs[k] = None
                if nxt < n:
                    futs.append(ex.submit(self.prepare_one, nxt))
                    nxt += 1
                if p["_y"] is None:                  # not usable: no photometric scale
                    prev = None
                else:
                    if prev is not None:
                        self._ptc_pair(prev, p, acc)
                    prev = {kk: p[kk] for kk in ("_y", "_valid", "_surf", "T")}
                self.params[k] = {kk: v for kk, v in p.items() if not kk.startswith("_")}
                if progress:
                    progress(k + 1, n, f"ImageMM: preparing exposures {k + 1}/{n}")
        self.ptc = self._ptc_fit(acc)
        return self

    # ------------------------------------------------------------- photon transfer
    def _ptc_pair(self, a: dict, b: dict, acc: dict, nbins: int = 32):
        """Variance of the difference of two consecutive exposures (both scaled to the
        reference and background-matched) in star-free pixels, binned by level.

        var(y/T) = (c0 + c1 L)/T^2 with L = T * l the expected exposure level (ADU), l the
        reference level plus the exposure's sky deviation, so
        var(diff) = c0 (1/Ta^2 + 1/Tb^2) + c1 (la/Ta + lb/Tb)."""
        ok = a["_valid"] & b["_valid"] & ~self.smask
        for c in range(3):
            Ta, Tb = a["T"][c], b["T"][c]
            la = self.ref[..., c] + a["_surf"][..., c]
            lb = self.ref[..., c] + b["_surf"][..., c]
            d = (a["_y"][..., c] / Ta - a["_surf"][..., c]) - (b["_y"][..., c] / Tb - b["_surf"][..., c])
            lev = 0.5 * (la + lb)
            dv, lv = d[ok], lev[ok]
            edges = np.unique(np.quantile(lv, np.linspace(0, 1, nbins + 1)))
            idx = np.clip(np.searchsorted(edges, lv, side="right") - 1, 0, len(edges) - 2)
            for j in range(len(edges) - 1):
                s = idx == j
                if s.sum() < 500:
                    continue
                dd = dv[s]
                var = (1.4826 * np.median(np.abs(dd - np.median(dd)))) ** 2
                l_a, l_b = np.median(la[ok][s]), np.median(lb[ok][s])
                acc["X"].append((c, 1 / Ta ** 2 + 1 / Tb ** 2, l_a / Ta + l_b / Tb))
                acc["v"].append(var)
                acc["n"].append(int(s.sum()))

    @staticmethod
    def _ptc_fit(acc: dict) -> dict:
        from scipy.optimize import nnls
        if not acc["X"]:
            raise RuntimeError("photon-transfer calibration impossible: no two consecutive usable exposures "
                               "(each needs a photometric scale from >= 3 isolated stars at SNR >= 50 in every "
                               "channel) with enough star-free pixels")
        X = np.array(acc["X"], float)
        v = np.array(acc["v"], float)
        n = np.array(acc["n"], float)
        c0, c1, red = np.zeros(3), np.zeros(3), np.zeros(3)
        for c in range(3):
            s = X[:, 0] == c
            if s.sum() < 2:
                raise RuntimeError(f"photon-transfer calibration impossible in channel {c}: too few level bins")
            A = X[s, 1:]
            y = v[s]
            w = np.sqrt(n[s] / 2) / np.maximum(y, 1e-12)        # sd of a variance estimate ~ var sqrt(2/n)
            coef = nnls(A * w[:, None], y * w)[0]              # c0 (read noise^2), c1 (1/gain) >= 0
            for _ in range(3):                                 # reweight with the model variance
                mdl = A @ coef
                w = np.sqrt(n[s] / 2) / np.maximum(mdl, 1e-12)
                coef = nnls(A * w[:, None], y * w)[0]
            c0[c], c1[c] = coef
            mdl = A @ coef
            red[c] = float(np.mean(((y - mdl) * w) ** 2))
        return {"c0": c0, "c1": c1, "chi2_red": red, "bins": int(len(v))}

    # ------------------------------------------------------------- persistence
    _STATE = ("params", "ptc", "sky_ref", "sky_info", "fwhm_ref", "cat", "smask")

    def save(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"names": [info.name for info, _ in self.items],
                         **{k: getattr(self, k) for k in self._STATE}}, f)

    def load(self, path: str) -> "ExposureSet":
        import pickle
        with open(path, "rb") as f:
            st = pickle.load(f)
        if st["names"] != [info.name for info, _ in self.items]:
            raise RuntimeError("the prepared exposures belong to a different frame selection")
        for k in self._STATE:
            setattr(self, k, st[k])
        return self

    def usable(self) -> list[int]:
        """Exposures with a PSF in every channel (the model needs f(t))."""
        return [k for k, p in enumerate(self.params) if p is not None and all(q is not None for q in p["psf"])
                and np.all(np.isfinite(p["T"]))]

    def moffat_psfs(self):
        """Fit the Moffat model to every exposure's empirical PSFs (``fit_moffat``)."""
        for p in self.params:
            if p is None or "psf_moffat" in p:
                continue
            ms, infos = [], []
            for q, e in zip(p["psf"], p["psf_err"]):
                if q is None or e is None:
                    ms.append(None)
                    infos.append(None)
                    continue
                mdl, inf = fit_moffat(e["mean"], e["se"])
                ms.append(mdl)
                infos.append(inf)
            p["psf_moffat"], p["moffat"] = ms, infos

    def kernels(self, idx: list[int], model: str = "empirical") -> np.ndarray:
        """(n, 3, ks, ks) PSFs of the chosen exposures, zero-padded to a common odd size
        (zero padding does not change a convolution).  model: "empirical" (measured, the
        paper's input) or "moffat" (the fitted models, see ``moffat_psfs``)."""
        key = "psf" if model == "empirical" else "psf_moffat"
        if key == "psf_moffat":
            self.moffat_psfs()
        ks = max(q.shape[0] for k in idx for q in self.params[k][key])
        out = np.zeros((len(idx), 3, ks, ks), np.float32)
        for a, k in enumerate(idx):
            for c, q in enumerate(self.params[k][key]):
                o = (ks - q.shape[0]) // 2
                out[a, c, o:o + q.shape[0], o:o + q.shape[0]] = q
        return out

    # ------------------------------------------------------------- data for a window
    def group_coadds(self, idx: list[int], n_groups: int, model: str = "empirical", progress=None,
                     cancel=None) -> dict:
        """Full-field seeing-group coadds of the exposures ``idx`` (see imagemm.coadd_groups):
        exposures sorted by PSF width into ``n_groups`` equal-count groups, each reduced to
            y_g = sum w y / sum w,  v_g = 1 / sum w,  m_g = [sum w > 0],  w = m / v,
        with PSF f_g = sum_t W_t f_t / sum_t W_t (W_t: field-mean weight per channel).
        One pass over the exposures.  Returns {"y", "v", "m": (G, H, W, 3), "kernels":
        (G, 3, ks, ks), "groups": member lists, "fwhm": per group}."""
        from .imagemm import _fwhm, seeing_groups
        K = self.kernels(idx, model)
        groups = seeing_groups(K, n_groups)
        G = len(groups)
        of = {int(t): g for g, members in enumerate(groups) for t in members}
        H, W = self.H0, self.W0
        S = np.zeros((G, H, W, 3), np.float32)
        Wsum = np.zeros((G, H, W, 3), np.float32)
        Wbar = np.zeros((len(idx), 3))
        with ThreadPoolExecutor(max_workers=max(1, self.workers // 2)) as ex:
            futs = {ex.submit(self.window, k, 0, H, 0, W): a for a, k in enumerate(idx[:self.workers])}
            nxt = min(len(idx), self.workers)
            done = 0
            while futs:
                fut = next(iter(futs))
                a = futs.pop(fut)
                y, v, m = fut.result()
                if nxt < len(idx):
                    futs[ex.submit(self.window, idx[nxt], 0, H, 0, W)] = nxt
                    nxt += 1
                w = np.where(m > 0, 1 / np.maximum(v, 1e-30), 0).astype(np.float32)
                g = of[a]
                S[g] += w * y
                Wsum[g] += w
                Wbar[a] = w.reshape(-1, 3).mean(0)
                done += 1
                if cancel and cancel():
                    raise RuntimeError("cancelled")
                if progress:
                    progress(done, len(idx), f"Seeing-group coadds {done}/{len(idx)}")
        Kg = np.stack([(K[g] * Wbar[g][:, :, None, None]).sum(0) / np.maximum(Wbar[g].sum(0), 1e-30)[:, None, None]
                       for g in groups]).astype(np.float32)
        ok = Wsum > 0
        Y = np.where(ok, S / np.maximum(Wsum, 1e-30), 0).astype(np.float32)
        V = np.where(ok, 1 / np.maximum(Wsum, 1e-30), 1).astype(np.float32)
        return {"y": Y, "v": V, "m": ok.astype(np.float32), "kernels": Kg,
                "groups": [[idx[int(t)] for t in g] for g in groups],
                "fwhm": [float(_fwhm(k.mean(0))) for k in Kg]}

    def windows(self, idx: list[int], y0: int, y1: int, x0: int, x1: int):
        """Stacked y, v, m of the chosen exposures on a window: (n, 3, h, w) float32 each."""
        n, h, w = len(idx), y1 - y0, x1 - x0
        Y = np.empty((n, 3, h, w), np.float32)
        V = np.empty_like(Y)
        Mk = np.empty_like(Y)
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            for a, (yy, vv, mm) in enumerate(ex.map(lambda k: self.window(k, y0, y1, x0, x1), idx)):
                Y[a], V[a], Mk[a] = (np.moveaxis(z, -1, 0) for z in (yy, vv, mm))
        return Y, V, Mk

    def window(self, k: int, y0: int, y1: int, x0: int, x1: int):
        """y(t), v(t), m(t) of exposure k on the reference-grid window (float32, HxWx3)."""
        from .stacking import eval_surface
        info, fr = self.items[k]
        p = self.params[k]
        raw, sat, rep = self._raw(info)
        mx, my = window_maps(fr, self.W0, self.H0, y0, y1, x0, x1, p["refine"])
        y, valid, _ = warp_window(raw, sat, rep, self.pattern, mx, my)
        h, w = y1 - y0, x1 - x0
        if y is None:
            z = np.zeros((h, w, 3), np.float32)
            return z, np.ones_like(z), z
        obs = self._obstruction(fr, y0, y1, x0, x1)
        if obs is not None:
            valid &= obs
        surf = eval_surface(p["surf"], self.H0, self.W0)[y0:y1, x0:x1]
        T = p["T"][None, None, :]
        ybs = y / T - surf - self.sky_ref[y0:y1, x0:x1]
        level = np.maximum(T * (self.ref[y0:y1, x0:x1] + surf), 0)            # expected exposure level, ADU
        var = (self.ptc["c0"][None, None, :] + self.ptc["c1"][None, None, :] * level) / T ** 2
        m = np.repeat(valid[..., None], 3, -1).astype(np.float32)
        return ybs.astype(np.float32), var.astype(np.float32), m
