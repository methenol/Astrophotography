"""Shared benchmark harness for denoising / deconvolution experiments.

Ground truth never exists for real astro data, but the two half-stacks give an
unbiased referee: a method sees ONLY half A (and is trained only on the
training bands), then its estimate x̂ is compared with half B on held-out bands.
Because B's noise is independent of everything the method saw,

    E |x̂ - B|² = E |x̂ - x|² + σ_B²

so subtracting B's (measured) noise variance gives the true error of x̂.
For deconvolution the same holds after re-blurring: E|k*x̂ - B|² - σ_B².
Scores are reported relative to the noise of a single half-stack
("error / σ²"; 1.0 = no better than the raw half, lower is better) and as dB.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from astropy.io import fits  # noqa: E402

DATASETS = {
    "IC5070": "output/IC_5070_sub-a9642d",
    "M27": "output/M_27_sub-2c99a3",
    "M31": "output/M_31_sub-44e3ac",
    # re-stacked with psf_groups=3 for the ImageMM experiments (exp_imagemm.py)
    "IC5070g": "experiments/cache/stacks/IC_5070_sub-a9642d",
    "M27g": "experiments/cache/stacks/M_27_sub-2c99a3",
}
BAND = 256          # width of train / test bands
CROP = 1536         # benchmark crop (per side) at 1x; scaled for drizzled stacks


def _fits(path):
    d = fits.getdata(path).astype(np.float32)
    return np.ascontiguousarray(np.moveaxis(d, 0, -1)) if d.ndim == 3 and d.shape[0] == 3 else d


def load(name: str, crop: int | None = None):
    """Half stacks, full stack and coverage cropped to the deepest square region."""
    d = os.path.join(ROOT, DATASETS[name])
    meta = json.load(open(os.path.join(d, "stack_meta.json")))
    cov = _fits(os.path.join(d, "coverage.fits"))
    s = int((crop or CROP) * meta.get("scale", 1.0) ** 0.5)   # 2x stacks: bigger crop, same sky fraction-ish
    s -= s % BAND
    import cv2
    c = cv2.blur(cov, (s // 4, s // 4))
    c[: s // 2], c[-s // 2:], c[:, : s // 2], c[:, -s // 2:] = 0, 0, 0, 0
    cy, cx = np.unravel_index(np.argmax(c), c.shape)
    y0, x0 = cy - s // 2, cx - s // 2
    sl = (slice(y0, y0 + s), slice(x0, x0 + s))
    a = _fits(os.path.join(d, "half_a.fits"))[sl].copy()
    b = _fits(os.path.join(d, "half_b.fits"))[sl].copy()
    full = _fits(os.path.join(d, "stack.fits"))[sl].copy()
    return dict(name=name, dir=d, sl=sl, a=a, b=b, full=full, cov=cov[sl].copy(), sat=meta["saturation"],
                scale=meta.get("scale", 1.0), filter=meta.get("filter", ""))


def split_masks(shape):
    """Vertical bands: every third band is held out for testing."""
    h, w = shape[:2]
    band = (np.arange(w) // BAND) % 3 == 1
    test = np.broadcast_to(band[None, :], (h, w)).copy()
    return ~test, test


class NoiseModel:
    """Local noise variance of ONE half stack, measured from the half-stack difference.

    var(p) = Gaussian-window mean of (A-B)^2 / 2.  A ~20 px window averages
    hundreds of pixels, so its correlation with any single pixel of B is negligible.
    """

    def __init__(self, a, b, signal=None, sigma=10.0):
        import cv2
        d2 = 0.5 * (a - b) ** 2
        v = cv2.GaussianBlur(d2, (0, 0), sigma)
        self.v = np.maximum(v, np.percentile(v, 1, axis=(0, 1)) * 0.5).astype(np.float32)

    def var(self, signal=None):
        return self.v


def score(xhat, bench, blur=None, key="test"):
    """Unbiased error of x̂ on the held-out bands, in units of one half-stack's noise variance.

    ``blur``: optional callable applied to x̂ first (the forward model k*x̂ for deconvolution).
    Returns dict with linear-domain (inverse-variance weighted) and stretched-domain scores.
    """
    a, b = bench["a"], bench["b"]
    test = bench[key]
    pred = blur(xhat) if blur is not None else xhat
    var = bench["nm"].var(bench["ref"])
    ok = test & bench["valid"]
    e_lin = ((pred - b) ** 2 / var)[ok].mean() - 1.0
    # stretched domain (what the eye sees after an asinh-type stretch): faint signal matters
    st = bench["stab"]
    ps, bs, as_ = st.fwd(pred), st.fwd(b), st.fwd(a)
    noise_s = 0.5 * ((as_ - bs) ** 2)[ok].mean()
    e_str = ((ps - bs) ** 2)[ok].mean() - noise_s
    return {"lin": float(e_lin), "lin_db": float(-10 * np.log10(max(e_lin, 1e-6))),
            "str": float(e_str / noise_s), "str_db": float(-10 * np.log10(max(e_str / noise_s, 1e-6)))}


def prepare(name: str, crop: int | None = None):
    from astrophoto.denoise import Stabiliser
    from scipy.ndimage import binary_erosion
    bench = load(name, crop)
    train, test = split_masks(bench["a"].shape)
    good = bench["cov"] >= 0.6 * np.percentile(bench["cov"], 90)
    unsat = bench["full"].max(-1) < 0.5 * bench["sat"]
    unsat = binary_erosion(unsat, iterations=6)
    bench["train"], bench["test"] = train & good, test
    bench["valid"] = good & unsat
    bench["stab"] = Stabiliser(bench["a"], bench["b"])
    # smooth signal estimate for the variance model (full stack, lightly blurred)
    import cv2
    bench["ref"] = cv2.GaussianBlur(bench["full"], (0, 0), 1.5)
    bench["nm"] = NoiseModel(bench["a"], bench["b"], bench["ref"])
    return bench


class Timer:
    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        self.dt = time.time() - self.t
