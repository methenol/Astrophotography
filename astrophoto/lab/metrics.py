"""Quality metrics for the experiment lab.

Real data has no ground truth, so every real-data score is a held-out one:

* ``heldout_chi2`` (ImageMM-style restorations of a window): the subs are split into A
  (even) and B (odd); the method sees only A.  Every B sub y is predicted through its own
  forward model D H x̂ (Eq. 1 / 10 of arXiv:2501.03002).  y is independent of x̂, so
  E[(y - D H x̂)^2 / v] - 1 = E[(D H (x̂ - x))^2] / v: the mean normalised squared residual
  minus one is an unbiased measure of the restoration error seen through each held-out
  sub's own PSF (0 = perfect), reported on sky and on source pixels.
* ``halfstack_score`` (stack-level denoising): the stacker's two half-stacks are the same
  sky with independent noise.  A method sees half A only (and trains on two of every three
  bands); on the held-out bands E|x̂ - B|^2 = E|x̂ - x|^2 + sigma_B^2, so subtracting B's
  noise variance gives the true error, in units of one half-stack's noise variance.

The paper's own quality metrics (Sec. 5.2, 5.3): sharpness S_F (Krotkov 1988), sigma_sky
(sep background rms), and aperture photometry against the coadd.

Synthetic data has an exact truth (``synthetic.truth_image``); ``truth_metrics`` compares a
result with it at a common resolution.
"""
from __future__ import annotations

import numpy as np
import sep
import torch

BAND = 256


# ----------------------------------------------------------------------------- held-out, subs
def heldout_chi2(es, idx_b, x, r, kernels_b, window, smask, device):
    """Mean of (y - D H x)^2 / v - 1 over held-out exposures ``idx_b``, per channel, on sky
    (``~smask``) and source (``smask``) pixels of ``window`` = (y0, y1, x0, x1)."""
    from .. import imagemm as M
    y0, y1, x0, x1 = window
    Y, V, Mk = es.windows(idx_b, y0, y1, x0, x1)
    ks = kernels_b.shape[-1]
    X, Yl = M.latent_shape(y1 - y0, x1 - x0, ks, r)
    o = int(round(M.exposure_origin(ks, r)))
    # embed the estimate (defined on the window) into the padded latent; outside the window
    # continue it by edge replication (only the outer PSF radius is affected, excluded below)
    xt = torch.from_numpy(np.ascontiguousarray(np.moveaxis(x, -1, 0)))[None].to(device)
    xl = torch.nn.functional.pad(xt, (o, Yl - xt.shape[-1] - o, o, X - xt.shape[-2] - o), mode="replicate")
    ops = M.Operators(torch.from_numpy(kernels_b).to(device), r)
    res = {"sky": [], "src": []}
    edge = (ks // r) + 2
    inner = np.zeros((y1 - y0, x1 - x0), bool)
    inner[edge:-edge, edge:-edge] = True
    for a, b in ops.ranges():
        fx = ops.forward(xl, a, b).cpu().numpy()
        z = (Y[a:b] - fx) ** 2 / V[a:b]
        m = (Mk[a:b] > 0) & inner[None, None]
        for key, sel in (("sky", ~smask), ("src", smask)):
            mm = m & sel[None, None]
            res[key].append((np.where(mm, z, 0).sum((0, 2, 3)), mm.sum((0, 2, 3))))
    out = {}
    for key, lst in res.items():
        s = sum(q[0] for q in lst)
        n = sum(q[1] for q in lst)
        # no pixels of this kind in the window: undefined, not -1
        out[key] = [round(float(a_ / b_ - 1), 5) if b_ > 0 else None for a_, b_ in zip(s, n)]
    return out


def sharpness(img):
    """S_F (Krotkov 1988): log magnitude of the 2-D Fourier transform, averaged over frequencies."""
    return float(np.mean(np.log(np.abs(np.fft.fft2(img)) + 1e-12)))


def sky_sigma(img):
    return float(sep.Background(np.ascontiguousarray(img, np.float32)).globalrms)


def photometry(coadd, img, thresh):
    """Sources of ``coadd`` (sep, absolute threshold); fluxes in the same elliptical apertures
    (2.5 Kron radii) on the coadd and on ``img``.  Returns (coadd magnitudes, img - coadd)."""
    L = np.ascontiguousarray(coadd, np.float32)
    objs = sep.extract(L, thresh)
    kr, _ = sep.kron_radius(L, objs["x"], objs["y"], objs["a"], objs["b"], objs["theta"], 6.0)
    r = 2.5 * np.maximum(kr, 1.0)
    fc, _, _ = sep.sum_ellipse(L, objs["x"], objs["y"], objs["a"], objs["b"], objs["theta"], r)
    fi, _, _ = sep.sum_ellipse(np.ascontiguousarray(img, np.float32), objs["x"], objs["y"], objs["a"], objs["b"],
                               objs["theta"], r)
    ok = (fc > 0) & (fi > 0)
    mc = -2.5 * np.log10(fc[ok])
    dm = -2.5 * np.log10(fi[ok]) - mc
    return mc, dm


def paper_metrics(x, coadd):
    """S_F, sigma_sky, PSNR / SSIM against the coadd, and photometry (Sec. 5.2, 5.3)."""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    rng = float(coadd.max() - coadd.min())
    out = {"S_F": float(np.mean([sharpness(x[..., c]) for c in range(3)])),
           "sigma_sky": float(np.mean([sky_sigma(x[..., c]) for c in range(3)])),
           "psnr_vs_coadd": float(peak_signal_noise_ratio(coadd, x, data_range=rng)),
           "ssim_vs_coadd": float(structural_similarity(coadd, x, data_range=rng, channel_axis=-1))}
    try:
        mc, dm = photometry(coadd.mean(-1), x.mean(-1), thresh=5 * sky_sigma(coadd.mean(-1)))
        if len(mc) >= 4:
            bright = mc <= np.percentile(mc, 50)
            out["dm_bright"] = float(np.median(dm[bright]))
            out["dm_faint"] = float(np.median(dm[~bright]))
    except Exception:
        pass
    return out


# ----------------------------------------------------------------------------- held-out, half stacks
def split_masks(shape):
    """Vertical bands: every third band is held out for testing."""
    h, w = shape[:2]
    band = (np.arange(w) // BAND) % 3 == 1
    test = np.broadcast_to(band[None, :], (h, w)).copy()
    return ~test, test


class NoiseModel:
    """Local noise variance of ONE half stack from the half-stack difference:
    var(p) = Gaussian-window mean of (A - B)^2 / 2.  A ~20 px window averages hundreds of
    pixels, so its correlation with any single pixel of B is negligible."""

    def __init__(self, a, b, sigma=10.0):
        import cv2
        v = cv2.GaussianBlur(0.5 * (a - b) ** 2, (0, 0), sigma)
        self.v = np.maximum(v, np.percentile(v, 1, axis=(0, 1)) * 0.5).astype(np.float32)


def halfstack_score(xhat, a, b, test, valid, stab, var, blur=None):
    """Unbiased error of x̂ (an estimate made from half A) against half B on the held-out
    pixels, in units of one half-stack's noise variance: linear (inverse-variance weighted)
    and in the variance-stabilised (asinh) domain.  ``blur``: the forward model k * x̂ first."""
    pred = blur(xhat) if blur is not None else xhat
    ok = test & valid
    e_lin = ((pred - b) ** 2 / var)[ok].mean() - 1.0
    ps, bs, as_ = stab.fwd(pred), stab.fwd(b), stab.fwd(a)
    noise_s = 0.5 * ((as_ - bs) ** 2)[ok].mean()
    e_str = ((ps - bs) ** 2)[ok].mean() - noise_s
    return {"heldout_lin": float(e_lin), "heldout_str": float(e_str / noise_s)}


# ----------------------------------------------------------------------------- stack proxies
def star_fwhm(img, sat, thresh=15.0):
    """Median FWHM (2 x half-light radius) of isolated, unsaturated stars (luminance)."""
    from scipy.spatial import cKDTree
    L = np.ascontiguousarray(img.mean(-1) if img.ndim == 3 else img, np.float32)
    bk = sep.Background(L, bw=64, bh=64)
    sub = L - bk.back()
    o = sep.extract(sub, thresh, err=bk.globalrms, minarea=5)
    if len(o) < 5:
        return float("nan")
    d, _ = cKDTree(np.stack([o["x"], o["y"]], 1)).query(np.stack([o["x"], o["y"]], 1), k=2)
    ok = (o["peak"] < 0.5 * sat) & (o["flag"] == 0) & (d[:, 1] > 12 * np.sqrt(o["a"] * o["b"]))
    if ok.sum() < 5:
        return float("nan")
    r, _ = sep.flux_radius(sub, o["x"][ok], o["y"][ok], 6 * o["a"][ok], 0.5, normflux=o["flux"][ok], subpix=5)
    return float(np.median(2 * r))


def background_noise(img):
    """Robust per-channel noise of the background (sep global rms), channel mean."""
    return float(np.mean([sky_sigma(img[..., c]) for c in range(img.shape[-1])]))


# ----------------------------------------------------------------------------- ground truth
def _plane_fit(diff, mask):
    """Least-squares plane (offset + x, y gradient) of ``diff`` (h, w) over ``mask``."""
    h, w = diff.shape
    yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
    A = np.stack([np.ones(mask.sum()), xx[mask], yy[mask]], 1)
    coef, *_ = np.linalg.lstsq(A, diff[mask], rcond=None)
    return coef[0] + coef[1] * xx + coef[2] * yy


def gaussian_blur(img, sigma):
    import cv2
    if not sigma:
        return img
    return np.stack([cv2.GaussianBlur(img[..., c], (0, 0), sigma) for c in range(img.shape[-1])], -1)


def truth_metrics(est, truth, valid, star_truth=None, stars=None, sigma_eval: float = 1.0,
                  fit_plane: bool = True, ap_radius: float = 3.0):
    """Comparison of ``est`` with the exact ``truth`` (both (h, w, 3), same grid and units) on
    ``valid`` pixels, after smoothing both with a Gaussian of ``sigma_eval`` pixels (the
    resolution at which they are compared) and, if ``fit_plane``, removing a per-channel
    plane from the difference (the sky background and its gradient are not part of the
    truth).

    * nrmse: rms error / rms of the truth's structure (per channel, averaged); lower is better
    * psnr: 20 log10(truth peak / rms error)
    * ssim: SSIM (Wang et al. 2004) of the asinh-stretched images
    * faint_nrmse: nrmse where the truth's stars are negligible (extended emission and sky)
    * star_dmag_median / star_dmag_mad: aperture photometry of isolated true stars
      (``stars``: dict with x, y, flux (per channel) on this grid) vs their true fluxes
    """
    from skimage.metrics import structural_similarity
    e, t = gaussian_blur(est, sigma_eval), gaussian_blur(truth, sigma_eval)
    out = {}
    nr, ps, fr = [], [], []
    starless = None
    if star_truth is not None:
        st = gaussian_blur(star_truth, sigma_eval).max(-1)
        starless = valid & (st < 0.02 * np.maximum(t.max(-1), 1e-6) + 1e-3 * np.percentile(t, 99.9))
    for c in range(3):
        d = e[..., c] - t[..., c]
        if fit_plane:
            d = d - _plane_fit(d, valid)
        rms = float(np.sqrt(np.mean(d[valid] ** 2)))
        spread = float(np.std(t[..., c][valid])) or 1.0
        nr.append(rms / spread)
        ps.append(20 * np.log10(max(float(t[..., c][valid].max()), 1e-9) / max(rms, 1e-12)))
        if starless is not None and starless.sum() > 100:
            fr.append(float(np.sqrt(np.mean(d[starless] ** 2))) / (float(np.std(t[..., c][starless])) or 1.0))
        if c == 0:
            dplane = [d]
        else:
            dplane.append(d)
    out["nrmse"] = float(np.mean(nr))
    out["psnr"] = float(np.mean(ps))
    if fr:
        out["faint_nrmse"] = float(np.mean(fr))
    # SSIM in a common asinh stretch set by the truth
    k = max(float(np.percentile(t[valid], 50)) * 0.2, 1e-3 * float(np.percentile(t[valid], 99.9)), 1e-6)
    top = float(np.arcsinh(np.percentile(t[valid], 99.9) / k))
    ec = np.stack([t[..., c] + dplane[c] for c in range(3)], -1)        # plane-corrected estimate
    sa, sb = np.arcsinh(np.maximum(ec, 0) / k) / top, np.arcsinh(np.maximum(t, 0) / k) / top
    ys, xs = np.nonzero(valid)
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    out["ssim"] = float(structural_similarity(np.clip(sb[sl], 0, 1.5), np.clip(sa[sl], 0, 1.5), data_range=1.5,
                                              channel_axis=-1))
    if stars is not None and star_truth is not None and len(stars["x"]):
        # photometry of the stars alone: the true extended emission is removed from the
        # estimate, and the estimate is compared with the true stars in the same aperture
        sb_ = gaussian_blur(star_truth, sigma_eval)
        ecl = np.ascontiguousarray((ec - (t - sb_)).mean(-1), np.float32)
        stl = np.ascontiguousarray(sb_.mean(-1), np.float32)
        x, y, f = stars["x"], stars["y"], stars["flux"].mean(-1)
        h, w = valid.shape
        ok = (x > ap_radius + 2) & (x < w - ap_radius - 3) & (y > ap_radius + 2) & (y < h - ap_radius - 3)
        ok &= valid[np.clip(y.astype(int), 0, h - 1), np.clip(x.astype(int), 0, w - 1)]
        if ok.sum() >= 5:
            from scipy.spatial import cKDTree
            # isolated: no other star with >= 10 % of its flux within 4 aperture radii
            tree = cKDTree(np.stack([x, y], 1))
            iso = np.array([all(f[j] < 0.1 * f[i] for j in tree.query_ball_point([x[i], y[i]], 4 * ap_radius) if j != i)
                            if ok[i] else False for i in range(len(x))])
            if iso.sum() >= 5:
                fe, _, _ = sep.sum_circle(ecl, x[iso], y[iso], ap_radius + 2 * sigma_eval, subpix=5)
                ft, _, _ = sep.sum_circle(stl, x[iso], y[iso], ap_radius + 2 * sigma_eval, subpix=5)
                pos = (fe > 0) & (ft > 0)
                if pos.sum() >= 5:
                    dm = -2.5 * np.log10(fe[pos] / ft[pos])
                    out["star_dmag_median"] = float(np.median(dm))
                    out["star_dmag_mad"] = float(1.4826 * np.median(np.abs(dm - np.median(dm))))
                    out["n_stars_phot"] = int(pos.sum())
    return out
