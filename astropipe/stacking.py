"""Streaming, memory-bounded integration.

Pass 1  weighted mean of globally normalised frames (reference for pass 2)
Pass 2  *local normalisation*: each frame's smooth sky difference against the
        pass-1 mean is modelled with a robust 2-D polynomial and removed, which
        equalises gradients that rotate with the alt-az field.  Weighted mean and
        variance of the normalised frames are accumulated.
Pass 3  weighted sigma-clipped integration against pass-2 statistics.  Frames
        are alternately accumulated into two independent half-stacks (A/B) whose
        noise is independent – the training pair for Noise2Noise denoising.

Two resampling modes:
  * ``demosaic``   edge-aware debayer + Lanczos warp
  * ``drizzle``    CFA/Bayer drizzle: each colour's sparse samples and its sampling
                   mask are warped separately and combined as a normalised
                   convolution – no demosaic interpolation, better colour
                   resolution, needs a reasonable number of dithered frames.
Optional output ``scale`` (e.g. 1.5, 2) produces an up-sampled (drizzle-like)
integration that exploits the sub-pixel dithering/rotation between frames.
"""
from __future__ import annotations

import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .frames import FrameInfo, cfa_masks, demosaic, fix_defects, read_raw


def _bounded_map(fn, items, workers):
    """Like executor.map but keeps at most ``2*workers`` results in flight (bounded memory)."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        q = deque()
        it = iter(items)
        for item in it:
            q.append(ex.submit(fn, item))
            if len(q) >= workers + 1:
                break
        while q:
            yield q.popleft().result()
            nxt = next(it, None)
            if nxt is not None:
                q.append(ex.submit(fn, nxt))


def _scaled_transform(M: np.ndarray, s: float) -> np.ndarray:
    """Map frame pixel coords -> output grid scaled by ``s`` (pixel-centre convention)."""
    A = np.asarray(M, np.float64).copy()
    out = A * s
    out[:, 2] += 0.5 * s - 0.5
    return out


def _poly_terms(x, y, deg):
    return [x ** i * y ** j for i in range(deg + 1) for j in range(deg + 1 - i)]


def fit_smooth_surface(img: np.ndarray, valid: np.ndarray, deg: int = 2, sub: int = 4,
                       block: int = 16) -> np.ndarray:
    """Robust low-order polynomial fit of a (H, W, C) residual image.

    Returns coefficient array (C, n_terms) in normalised [-1, 1] coordinates.
    """
    h, w, c = img.shape
    small = img[::sub, ::sub]
    vs = valid[::sub, ::sub]
    hb, wb = small.shape[0] // block, small.shape[1] // block
    small = small[: hb * block, : wb * block]
    vs = vs[: hb * block, : wb * block]
    arr = np.where(vs[..., None], small, np.nan).reshape(hb, block, wb, block, c)
    arr = arr.transpose(0, 2, 4, 1, 3).reshape(hb, wb, c, block * block)
    frac = np.isfinite(arr[..., 0, :]).mean(-1)
    with np.errstate(all="ignore"):
        med = np.nanmedian(arr, axis=-1)  # (hb, wb, c)
    yy, xx = np.mgrid[0:hb, 0:wb]
    cy = ((yy + 0.5) * block * sub) / h * 2 - 1
    cx = ((xx + 0.5) * block * sub) / w * 2 - 1
    ok = frac > 0.5
    coefs = np.zeros((c, len(_poly_terms(0, 0, deg))), np.float64)
    if ok.sum() < 3 * len(coefs[0]):
        for ch in range(c):
            coefs[ch, 0] = np.nanmedian(med[..., ch]) if ok.any() else 0.0
        return coefs
    X = np.stack(_poly_terms(cx[ok], cy[ok], deg), axis=1)
    for ch in range(c):
        z = med[..., ch][ok]
        keep = np.isfinite(z)
        for _ in range(4):
            beta, *_ = np.linalg.lstsq(X[keep], z[keep], rcond=None)
            r = z - X @ beta
            s = 1.4826 * np.median(np.abs(r[keep])) + 1e-9
            keep = np.abs(r) < 3 * s
        coefs[ch] = beta
    return coefs


def eval_surface(coefs: np.ndarray, h: int, w: int, deg: int = 2) -> np.ndarray:
    """Evaluate polynomial coefficients on a full (h, w) grid (via low-res + resize)."""
    lh, lw = max(8, h // 16), max(8, w // 16)
    y = (np.arange(lh) + 0.5) / lh * 2 - 1
    x = (np.arange(lw) + 0.5) / lw * 2 - 1
    xx, yy = np.meshgrid(x, y)
    terms = np.stack(_poly_terms(xx, yy, deg), axis=-1)  # lh, lw, T
    low = terms @ coefs.T  # lh, lw, C
    return cv2.resize(low.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC).reshape(h, w, -1)


class Integrator:
    def __init__(self, infos: list[FrameInfo], analysis: dict, defects: np.ndarray | None,
                 mode: str = "auto", scale: float = 1.0, sigma_low: float = 4.0,
                 sigma_high: float = 3.0, local_norm: bool = True, workers: int | None = None,
                 progress=None, cancel=None):
        frames = analysis["frames"]
        self.items = [(info, fr) for info, fr in zip(infos, frames) if fr["accepted"] and fr["weight"] > 0]
        if not self.items:
            raise RuntimeError("No frames accepted for stacking")
        self.defects = defects
        self.pattern = infos[0].bayer
        self.bias = infos[0].bias
        self.W0, self.H0 = infos[0].width, infos[0].height
        self.scale = float(scale)
        self.W, self.H = int(round(self.W0 * scale)), int(round(self.H0 * scale))
        if mode == "auto":
            mode = "drizzle" if (len(self.items) >= 15 and scale > 1.01) else "demosaic"
        self.mode = mode
        self.sigma_low, self.sigma_high = sigma_low, sigma_high
        self.local_norm = local_norm and len(self.items) >= 3
        # each in-flight frame holds values+weights at output resolution; keep RAM bounded
        # (values + weights + warp temporaries ~ 2.5x a frame), bounded by a RAM budget
        frame_gb = self.W * self.H * 3 * 4 * 2 * 2.5 / 2**30
        budget_gb = float(os.environ.get("ASTROPIPE_STACK_RAM_GB", 3.0))
        self.workers = workers or int(max(1, min(6, (os.cpu_count() or 4) // 2, budget_gb // max(frame_gb, 0.05))))
        self.progress = progress or (lambda *a: None)
        self.cancel = cancel or (lambda: False)
        self.grid = analysis["grid"]
        self._masks = cfa_masks(self.pattern, (self.H0, self.W0)) if mode == "drizzle" else None
        self.offsets: dict[int, np.ndarray] = {}
        self.gradients: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------ frames
    def _warp(self, idx: int):
        info, fr = self.items[idx]
        raw = read_raw(info.path, info.bias)
        raw = fix_defects(raw, self.defects)
        size = (self.W, self.H)
        maps = self._distortion_maps(fr)
        if maps is not None:
            m1, m2 = maps

            def warp(img, interp):
                return cv2.remap(img, m1, m2, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            M = _scaled_transform(fr["transform"], self.scale)

            def warp(img, interp):
                return cv2.warpAffine(img, M, size, flags=interp, borderValue=0)
        ones = np.ones((self.H0, self.W0), np.float32)
        if self.mode == "drizzle":
            # all three colours in two 3-channel warps: sparse samples and their masks
            num = warp(raw[..., None] * self._masks, cv2.INTER_LINEAR)
            den = warp(self._masks, cv2.INTER_LINEAR)
            # the masks sum to one, so their warped sum is the frame's footprint;
            # a pixel only counts where the frame fully covers it (no ragged edges)
            cov = den.sum(-1) > 0.999
            ok = (den > 1e-3) & cov[..., None]
            # in place: values where the weight is zero are never used (weight 0)
            vals = num
            np.divide(num, np.maximum(den, np.float32(1e-3)), out=vals)
            wts = den
            wts *= ok
        else:
            rgb = demosaic(raw, self.pattern)
            vals = warp(rgb, cv2.INTER_LANCZOS4)
            np.maximum(vals, 0, out=vals)
            cov = warp(ones, cv2.INTER_LINEAR) > 0.999
            wts = np.repeat(cov[..., None].astype(np.float32), 3, axis=2)
        # obstruction mask (tile grid in reference coordinates) -> smooth full-res weight
        tm = fr.get("tile_mask")
        if tm is not None and (tm < 1).any():
            m = cv2.resize(tm.astype(np.float32), (self.W, self.H), interpolation=cv2.INTER_LINEAR)
            wts *= np.clip((m - 0.25) / 0.5, 0, 1)[..., None]
        t = fr.get("transparency", 1.0)
        if not np.isfinite(t) or t <= 0.05:
            t = 1.0
        vals /= t
        return vals, wts, float(fr["weight"])

    def _distortion_maps(self, fr):
        """Inverse maps (output pixel -> source pixel): exact similarity + interpolated distortion.

        The polynomial model is split into the (exact, per-pixel) inverse similarity
        transform plus a small, smooth residual that is evaluated on a coarse grid
        and bilinearly up-sampled (error << 0.01 px).
        """
        dist = fr.get("distortion")
        if dist is None:
            return None
        from .analysis import poly_terms
        A = np.vstack([np.asarray(fr["transform"], np.float64), [0, 0, 1]])
        Ai = np.linalg.inv(A)[:2]
        # reference-pixel coordinates of every output pixel centre (separable, float32,
        # built with in-place broadcasting – no full-size float64 temporaries)
        rx = ((np.arange(self.W, dtype=np.float64) + 0.5) / self.scale - 0.5)
        ry = ((np.arange(self.H, dtype=np.float64) + 0.5) / self.scale - 0.5)
        mx = np.empty((self.H, self.W), np.float32)
        my = np.empty((self.H, self.W), np.float32)
        mx[:] = (Ai[0, 0] * rx + Ai[0, 2]).astype(np.float32)[None, :]
        mx += (Ai[0, 1] * ry).astype(np.float32)[:, None]
        my[:] = (Ai[1, 0] * rx + Ai[1, 2]).astype(np.float32)[None, :]
        my += (Ai[1, 1] * ry).astype(np.float32)[:, None]
        # residual on a coarse grid whose cells line up with cv2.resize's pixel-centre convention
        gh, gw = max(4, self.H // 32), max(4, self.W // 32)
        cy = (np.arange(gh) + 0.5) * self.H / gh - 0.5
        cx = (np.arange(gw) + 0.5) * self.W / gw - 0.5
        gx, gy = np.meshgrid((cx + 0.5) / self.scale - 0.5, (cy + 0.5) / self.scale - 0.5)
        T = poly_terms(gx / self.W0 * 2 - 1, gy / self.H0 * 2 - 1, dist["deg"])
        px, py = T @ dist["coefs"][0], T @ dist["coefs"][1]
        sx = Ai[0, 0] * gx + Ai[0, 1] * gy + Ai[0, 2]
        sy = Ai[1, 0] * gx + Ai[1, 1] * gy + Ai[1, 2]
        mx += cv2.resize((px - sx).astype(np.float32), (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        my += cv2.resize((py - sy).astype(np.float32), (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        # fixed-point maps make cv2.remap considerably faster
        return cv2.convertMaps(mx, my, cv2.CV_16SC2)

    def _normalise(self, idx, vals, wts, use_gradient: bool):
        valid = wts[..., 1] > 0
        if idx not in self.offsets:
            sub = vals[::8, ::8][valid[::8, ::8]]
            self.offsets[idx] = np.median(sub, axis=0) if len(sub) else np.zeros(3, np.float32)
        vals = vals - (self.offsets[idx] - self.ref_level)[None, None, :]
        if use_gradient and idx in self.gradients:
            vals -= eval_surface(self.gradients[idx], self.H, self.W)
        return vals, valid

    # ------------------------------------------------------------------ passes
    def run(self) -> dict:
        n = len(self.items)
        H, W = self.H, self.W
        # reference background level: the best-weighted frame
        best = int(np.argmax([fr["weight"] for _, fr in self.items]))
        v, w_, _ = self._warp(best)
        valid = w_[..., 1] > 0
        self.ref_level = np.median(v[::8, ::8][valid[::8, ::8]], axis=0).astype(np.float32)
        del v, w_

        # ---- pass 1: weighted mean
        S = np.zeros((H, W, 3), np.float32)
        Wt = np.zeros((H, W, 3), np.float32)
        for k, (vals, wts, fw) in enumerate(_bounded_map(self._warp, range(n), self.workers)):
            if self.cancel():
                raise RuntimeError("cancelled")
            vals, _ = self._normalise(k, vals, wts, False)
            ww = wts * fw
            S += vals * ww
            Wt += ww
            self.progress(k + 1, n, f"Integration pass 1/3 ({k + 1}/{n})")
        mu1 = S / np.maximum(Wt, 1e-6)
        del S

        # ---- pass 2: local normalisation + mean/variance
        S1 = np.zeros((H, W, 3), np.float32)
        S2 = np.zeros((H, W, 3), np.float32)
        Wt[:] = 0
        for k, (vals, wts, fw) in enumerate(_bounded_map(self._warp, range(n), self.workers)):
            if self.cancel():
                raise RuntimeError("cancelled")
            vals, valid = self._normalise(k, vals, wts, False)
            if self.local_norm:
                self.gradients[k] = fit_smooth_surface(vals - mu1, valid, deg=2)
                vals -= eval_surface(self.gradients[k], H, W)
            d = vals - mu1
            ww = wts * fw
            S1 += d * ww
            S2 += d * d * ww
            Wt += ww
            self.progress(k + 1, n, f"Integration pass 2/3 ({k + 1}/{n})")
        Wsafe = np.maximum(Wt, 1e-6)
        m = S1 / Wsafe
        mu2 = mu1 + m
        sd = np.sqrt(np.maximum(S2 / Wsafe - m * m, 0))
        del S1, S2, m, mu1
        # robust floor for the clipping scale (few-frame pixels at the edges)
        sd_floor = np.median(sd[sd > 0]) * 0.25 if (sd > 0).any() else 1.0
        sd = np.maximum(sd, sd_floor)

        # ---- pass 3: sigma-clipped integration into two half stacks
        SA = np.zeros((H, W, 3), np.float32)
        WA = np.zeros((H, W, 3), np.float32)
        SB = np.zeros((H, W, 3), np.float32)
        WB = np.zeros((H, W, 3), np.float32)
        rejected = np.zeros((H, W), np.float32)
        clip = n >= 5
        for k, (vals, wts, fw) in enumerate(_bounded_map(self._warp, range(n), self.workers)):
            if self.cancel():
                raise RuntimeError("cancelled")
            vals, valid = self._normalise(k, vals, wts, True)
            ww = wts * fw
            if clip:
                r = (vals - mu2) / sd
                bad = (r > self.sigma_high) | (r < -self.sigma_low)
                ww = np.where(bad, 0, ww)
                rejected += bad.any(-1) & valid
            if k % 2 == 0:
                SA += vals * ww
                WA += ww
            else:
                SB += vals * ww
                WB += ww
            self.progress(k + 1, n, f"Integration pass 3/3 ({k + 1}/{n})")
        Wtot = WA + WB
        full = np.where(Wtot > 0, (SA + SB) / np.maximum(Wtot, 1e-6), mu2)
        half_a = np.where(WA > 0, SA / np.maximum(WA, 1e-6), full)
        half_b = np.where(WB > 0, SB / np.maximum(WB, 1e-6), full)
        coverage = Wt[..., 1] / max(Wt[..., 1].max(), 1e-6)
        return {
            "stack": full.astype(np.float32),
            "half_a": half_a.astype(np.float32),
            "half_b": half_b.astype(np.float32),
            "coverage": coverage.astype(np.float32),
            "rejected_frac": (rejected / n).astype(np.float32),
            "mode": self.mode,
            "scale": self.scale,
            "n_frames": n,
            "total_exposure": float(sum(info.exptime for info, _ in self.items)),
        }
