"""Verification of astrophoto/imagemm.py against the paper's own model (arXiv:2501.03002).

Synthetic data follow Eq. 1 / Eq. 10 exactly (background-subtracted latent, per-exposure
PSFs, Gaussian noise with known per-pixel variances, masks), so every result can be
checked against the truth.

    python experiments/test_imagemm.py
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from astrophoto import imagemm as M  # noqa: E402
from astrophoto.denoise import pick_device  # noqa: E402

DEV = pick_device()
torch.manual_seed(0)
RNG = np.random.default_rng(0)
FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


def gauss_int(fwhm, size, e=0.0, theta=0.0, dx=0.0, dy=0.0):
    """Pixel-integrated elliptical Gaussian (5x5 sub-sampling), unit sum."""
    s = fwhm / 2.3548
    sub = 5
    c = size // 2
    o = (np.arange(size * sub) + 0.5) / sub - 0.5 - c
    yy, xx = np.meshgrid(o - dy, o - dx, indexing="ij")
    ct, st = np.cos(theta), np.sin(theta)
    u, v = ct * xx + st * yy, -st * xx + ct * yy
    k = np.exp(-(u ** 2 / (2 * s ** 2 * (1 + e) ** 2) + v ** 2 / (2 * s ** 2)))
    k = k.reshape(size, sub, size, sub).mean((1, 3))
    return (k / k.sum()).astype(np.float32)


# ----------------------------------------------------------------- 1. operators
def test_operators():
    for r in (1, 2):
        n, C, d, ks = 5, 3, 40, 9 * r
        k = torch.rand(n, C, ks, ks, device=DEV)
        k /= k.sum((-2, -1), keepdim=True)
        ops = M.Operators(k, r, chunk=2)
        X, Y = M.latent_shape(d, d, ks, r)
        x = torch.rand(1, C, X, Y, device=DEV)
        z = torch.rand(n, C, d, d, device=DEV)
        fx = torch.cat([ops.forward(x, a, b) for a, b in ops.ranges()])
        check(f"r={r} forward shape", tuple(fx.shape) == (n, C, d, d), str(tuple(fx.shape)))
        lhs = float((fx * z).sum())
        rhs = float(sum((x * ops.adjoint_sum(z[a:b], a, b)).sum() for a, b in ops.ranges())) / r ** 2
        check(f"r={r} <DHx,z> = <x,H^T D^T z>/r^2", abs(lhs - rhs) / abs(lhs) < 1e-5, f"rel {abs(lhs - rhs) / abs(lhs):.1e}")


# ----------------------------------------------------------------- 2. geometry
def test_operator_vs_conv2d():
    """Operators (im2col matrix products, in row bands) against a plain grouped conv2d /
    conv_transpose2d, with bands forced to a few rows."""
    for r in (1, 2):
        n, C, d, ks = 7, 3, 37, 11 * r
        k = torch.rand(n, C, ks, ks, device=DEV)
        X, Y = M.latent_shape(d, d, ks, r)
        x = torch.rand(1, C, X, Y, device=DEV)
        z = torch.rand(n, C, d, d, device=DEV)
        w = k.flip(-2, -1).reshape(n * C, 1, ks, ks)
        ref = M.pool(F.conv2d(x.expand(n, -1, -1, -1).reshape(1, n * C, X, Y), w, groups=n * C), r).reshape(n, C, d, d)
        refa = F.conv_transpose2d(M.unpool(z, r).reshape(1, n * C, d * r, d * r), w, groups=n * C).reshape(n, C, X, Y).sum(0, keepdim=True)
        for bb in (512e6, 4 * (ks * ks + n) * Y * 3):          # one band / bands of ~3 rows
            ops = M.Operators(k, r, band_bytes=bb)
            fx = ops.forward(x, 0, n)
            ad = ops.adjoint_sum(z, 0, n)
            e1 = float((fx - ref).abs().max() / ref.abs().max())
            e2 = float((ad - refa).abs().max() / refa.abs().max())
            check(f"r={r} operators = conv2d / conv_transpose2d (band {int(bb)} B)", e1 < 1e-5 and e2 < 1e-5,
                  f"forward {e1:.1e}, adjoint {e2:.1e}")


def test_stack_forward():
    """stack_forward: a point source at stack pixel u (drizzle grid, scale s) must predict an
    exposure-grid image centred on reference coordinate (u + 1/2)/s - 1/2; for s = 1 the
    prediction must equal the verified ImageMM operator."""
    for s_ in (1, 2):
        P = 96
        k = 25 if s_ == 1 else 2 * 15
        o = np.arange(k) - (k - 1) / 2
        g = np.exp(-(o[:, None] ** 2 + (o[None] * 1.3) ** 2) / (2 * (1.6 * s_) ** 2))
        K = torch.from_numpy((g / g.sum()).astype(np.float32))[None, None].expand(1, 3, k, k).contiguous().to(DEV)
        errs = []
        for u in (40.0, 41.0, 47.0, 48.0):
            x = torch.zeros(1, 3, P, P, device=DEV)
            x[..., int(u), int(u)] = 1.0
            pred, e0 = M.stack_forward(x, K, s_)
            img = pred[0, 0, 0].cpu().numpy()
            yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
            cy = (img * yy).sum() / img.sum() + e0
            cx = (img * xx).sum() / img.sum() + e0
            want = (u + 0.5) / s_ - 0.5
            errs.append(max(abs(cy - want), abs(cx - want)))
        check(f"stack_forward s={s_}: point source lands at (u + 1/2)/s - 1/2", max(errs) < 2e-3, f"max centroid error {max(errs):.1e} px")
        if s_ == 1:
            x = torch.rand(1, 3, P, P, device=DEV)
            pred, e0 = M.stack_forward(x, K, 1)
            ref = M.Operators(K, 1).forward(x, 0, 1)
            check("stack_forward s=1 equals the ImageMM operator", float((pred[0, 0] - ref[0]).abs().max()) < 1e-5 and e0 == (k - 1) / 2)


def test_true_convolution():
    """conv2d is a correlation: the operator must flip kernels.  A kernel whose only mass
    sits (ks-1)/2 pixels right of its centre must move a source right by that much."""
    d, ks = 40, 9
    ka = torch.zeros(1, 1, ks, ks, device=DEV)
    ka[0, 0, ks // 2, ks - 1] = 1.0
    op1 = M.Operators(ka, 1)
    x1 = torch.zeros(1, 1, *M.latent_shape(d, d, ks, 1), device=DEV)
    o = int(M.exposure_origin(ks, 1))
    x1[0, 0, 20 + o, 20 + o] = 1.0                   # source at exposure pixel (20, 20)
    out = op1.forward(x1, 0, 1)[0, 0]
    iy, ix = np.unravel_index(int(out.argmax()), out.shape)
    check("true convolution (kernel mass at +x moves the source +x)", (iy, ix) == (20, 20 + ks // 2), f"peak at {(iy, ix)}")


def test_geometry():
    for r in (1, 2):
        d, dp = 31, 11
        ks = r * dp
        k = torch.from_numpy(gauss_int(2.0 * r, ks if ks % 2 else ks + 1)[: ks, : ks]).to(DEV)
        # use a kernel whose centre is at (ks-1)/2 exactly: symmetric even/odd grid Gaussian
        o = np.arange(ks) - (ks - 1) / 2
        g = np.exp(-(o[:, None] ** 2 + o[None] ** 2) / (2 * (0.85 * r) ** 2))
        k = torch.from_numpy((g / g.sum()).astype(np.float32)).to(DEV)[None, None]
        ops = M.Operators(k, r)
        X, Y = M.latent_shape(d, d, ks, r)
        x = torch.zeros(1, 1, X, Y, device=DEV)
        i = 13
        lc = r * i + M.exposure_origin(ks, r)
        check(f"r={r} exposure origin integer", abs(lc - round(lc)) < 1e-9, f"{lc}")
        x[0, 0, int(round(lc)), int(round(lc))] = 1.0
        out = ops.forward(x, 0, 1)[0, 0].cpu().numpy()
        yy, xx = np.mgrid[:d, :d]
        cy, cx = (out * yy).sum() / out.sum(), (out * xx).sum() / out.sum()
        check(f"r={r} latent point at r*i+origin lands on exposure pixel i", abs(cy - i) < 1e-4 and abs(cx - i) < 1e-4,
              f"centroid {cy:.4f},{cx:.4f}")
        # initial_guess: D H x0 for a delta-like kernel reproduces a smooth median exactly
        med = torch.from_numpy(np.fromfunction(lambda a, b: 5 + np.sin(a / 5) + np.cos(b / 7), (d, d))
                               .astype(np.float32)).to(DEV)[None, None]
        x0 = M.initial_guess(med.expand(3, 1, d, d).contiguous(), torch.ones(3, 1, d, d, device=DEV), ks, r)
        yl = torch.arange(X, device=DEV, dtype=torch.float32)
        e = (yl - M.exposure_origin(ks, r)) / r                      # exposure coord of each latent row
        inside = (e >= 0) & (e <= d - 1)
        ref = 5 + torch.sin(e[inside] / 5)[:, None] + torch.cos(e[inside] / 7)[None, :]
        # exact at exposure pixel centres; between them bilinear interpolation, whose error is
        # bounded by h^2/8 max|f''| per axis (h = 1): 1/8 (1/25 + 1/49)
        on = inside & ((e - e.round()).abs() < 1e-6)
        err_c = float((x0[0, 0][on][:, on] - (5 + torch.sin(e[on] / 5)[:, None] + torch.cos(e[on] / 7)[None, :])).abs().max())
        err = float((x0[0, 0][inside][:, inside] - ref).abs().max())
        bound = (1 / 25 + 1 / 49) / 8
        check(f"r={r} initial guess: exact at exposure pixel centres", err_c < 1e-5, f"max err {err_c:.1e}")
        check(f"r={r} initial guess: within the bilinear bound elsewhere", err <= bound * 1.01, f"max err {err:.2e} <= {bound:.2e}")


# ----------------------------------------------------------------- 3. g_sigma, Eq. 11
def test_psf_solver():
    from scipy.special import erf
    sig, size = 1.1, 21
    g = M.gaussian_psf_mc(sig, size, n_samples=100_000_000)
    e = (np.arange(size + 1) - size // 2 - 0.5) / (sig * np.sqrt(2))
    p = 0.5 * np.diff(erf(e))
    exact = np.outer(p, p)
    exact /= exact.sum()
    err = np.abs(g - exact).max()
    check("Monte Carlo g_sigma matches the exact pixel integral", err < 2e-4, f"max abs err {err:.1e}")
    for r, sg in ((2, 1.1), (1, 1.0), (4, 1.1)):
        f = gauss_int(3.6, 25, e=0.15, theta=0.4)
        t = time.time()
        h, loss = M.refine_psf(f, r, sg, device=DEV)
        check(f"Eq.11 r={r} sigma={sg}: D(h*g) = f", loss < 1e-8 * float((f ** 2).mean()) and h.shape == (r * 25, r * 25),
              f"mse {loss:.2e} (paper: 3.94e-8), {time.time() - t:.0f}s")


# ----------------------------------------------------------------- 4. restoration
def scene(size, r=1, n_stars=60, gal=True):
    """Background-subtracted latent sky on the latent grid (r x oversampled)."""
    N = size * r
    x = np.zeros((N, N), np.float64)
    yy, xx = np.mgrid[:N, :N] / r
    if gal:                                            # extended source (exponential disc + spiral-ish arm)
        rr = np.hypot(yy - size * 0.45, (xx - size * 0.55) * 1.4)
        x += 40 * np.exp(-rr / (size * 0.06))
        x += 8 * np.exp(-((rr - size * 0.12) ** 2) / (2 * (size * 0.015) ** 2)) * (1 + np.cos(np.arctan2(yy - size * .45, xx - size * .55) * 2))
    pos = RNG.uniform(size * 0.08, size * 0.92, (n_stars, 2))
    flux = 10 ** RNG.uniform(2.3, 4.5, n_stars)
    for (py, px), fl in zip(pos, flux):
        iy, ix = int(py * r), int(px * r)
        x[iy, ix] += fl * r * r                        # point source on the latent grid (surface brightness)
    return x.astype(np.float32), pos, flux


def make_exposures(x_lat, r, n, size, fwhms, sky_sigma, gain, satellite=True, bad_frac=1e-3):
    """Exposures per Eq. 10: y = D(h_t * x) + noise, noise ~ N(0, sky^2 + max(signal,0)/gain)."""
    ks_lat = r * 25
    ys, vs, ms, ks = [], [], [], []
    xt = torch.from_numpy(x_lat).to(DEV)[None, None]
    # the padded latent: put the scene in the middle of the latent grid of latent_shape
    X, _ = M.latent_shape(size, size, ks_lat, r)
    o = int(round(M.exposure_origin(ks_lat, r) - (r - 1) / 2))
    xl = torch.zeros(1, 1, X, X, device=DEV)
    xl[..., o:o + size * r, o:o + size * r] = xt
    for t in range(n):
        f = gauss_int(fwhms[t] * r, ks_lat + (0 if ks_lat % 2 else 1), e=RNG.uniform(0, .2), theta=RNG.uniform(0, 3))
        f = f[:ks_lat, :ks_lat]
        f /= f.sum()
        kt = torch.from_numpy(f).to(DEV)[None, None]
        ops = M.Operators(kt, r)
        clean = ops.forward(xl, 0, 1)[0, 0].cpu().numpy()
        v = sky_sigma[t] ** 2 + np.maximum(clean, 0) / gain
        y = clean + RNG.normal(size=clean.shape) * np.sqrt(v)
        m = (RNG.random(clean.shape) > bad_frac).astype(np.float32)
        y[m == 0] = RNG.uniform(-1e4, 1e4, int((m == 0).sum()))        # garbage where masked
        if satellite and t == n // 2:                                    # unmasked outlier: satellite trail
            ii = np.arange(size)
            y[ii, np.clip((ii * 0.8 + 5).astype(int), 0, size - 1)] += 3000
        ys.append(y), vs.append(v), ms.append(m), ks.append(f)
    T = lambda a: torch.from_numpy(np.stack(a).astype(np.float32))[:, None].to(DEV)
    return T(ys), T(vs), T(ms), T(ks), xl


def test_restoration():
    size, n = 160, 20
    x_true, pos, flux = scene(size)
    fwhms = RNG.uniform(3.0, 6.0, n)
    y, v, m, k, xl = make_exposures(x_true, 1, n, size, fwhms, RNG.uniform(3, 6, n), gain=2.0)
    ks = k.shape[-1]
    x0 = M.initial_guess(y, m, ks)
    # ---- Algorithm 1: MM must not increase the loss (Eq. 3-4 guarantee; clipping only shortens steps)
    ops = M.Operators(k, 1)
    W = m / v

    def loss(x):
        return float(sum((W[a:b] * (y[a:b] - ops.forward(x, a, b)) ** 2).sum() for a, b in ops.ranges()))

    losses = []
    x1, info1 = M.mm_restore(y, v, m, k, x0, robust=False, max_iters=300, epsilon=0,
                             log=None)
    xx = x0.clone()
    for it in range(30):
        xx, _ = M.mm_restore(y, v, m, k, xx, robust=False, max_iters=2, epsilon=0)
        losses.append(loss(xx))
    mono = all(b <= a * (1 + 1e-6) for a, b in zip(losses, losses[1:]))
    check("Algorithm 1: L2 loss non-increasing", mono, f"{losses[0]:.4g} -> {losses[-1]:.4g}")
    # ---- restoration quality vs truth, on the field (latent crop = exposure grid)
    o = ks // 2
    crop = lambda z: z[0, 0, o:o + size, o:o + size].cpu().numpy()
    mean = (torch.where(m > 0, y, 0).sum(0) / m.sum(0).clamp_min(1))[0].cpu().numpy()
    xr, info = M.mm_restore(y, v, m, k, x0, robust=True, max_iters=2000, epsilon=1e-5)
    xr_c = crop(xr)
    print(f"      Algorithm 3: {info['iterations']} iterations, converged={info['converged']}")
    check("Eq. C15 stopping criterion reached", info["converged"])
    # the converged fit must explain the data down to the noise (reduced chi^2 ~ 1), apart from
    # the satellite trail, which the Huber loss is meant to leave unexplained
    ii = np.arange(size)
    trail_mask = torch.ones_like(m)
    trail_mask[n // 2, :, ii, np.clip((ii * 0.8 + 5).astype(int), 0, size - 1)] = 0
    chi = float(sum((m[a:b] * trail_mask[a:b] * (y[a:b] - ops.forward(xr, a, b)) ** 2 / v[a:b]).sum()
                    for a, b in ops.ranges()) / (m * trail_mask).sum())
    check("Algorithm 3: reduced chi^2 of the converged fit ~ 1", 0.8 < chi < 1.3, f"chi2/N {chi:.3f}")
    # background noise: pixels far from every source
    blur = F.conv2d(torch.from_numpy(x_true)[None, None], torch.ones(1, 1, 31, 31), padding=15)[0, 0].numpy()
    bgm = blur < 0.05                                  # truth below 1 % of the sky noise
    nb_mean, nb_mm = mean[bgm].std(), xr_c[bgm].std()
    check("sky background noise removed (paper Sec. 5.2)", nb_mm < 0.05 * nb_mean, f"std {nb_mean:.3f} -> {nb_mm:.4f}")
    # photometry: apertures of radius 6 around bright isolated stars
    ap = []
    yyg, xxg = np.mgrid[:size, :size]
    for (py, px), fl in zip(pos, flux):
        if fl < 1e3 or not (12 < py < size - 12 and 12 < px < size - 12):
            continue
        a = np.hypot(yyg - int(py), xxg - int(px)) <= 6
        ap.append((xr_c[a].sum() - 0) / (x_true[a].sum()))
    ap = np.array(ap)
    check("bright-star flux preserved (paper Sec. 5.3)", np.abs(np.median(ap) - 1) < 0.02, f"median ratio {np.median(ap):.4f}")
    # sharpness: restoration closer to the truth convolved with a narrow kernel than the mean is
    g1 = torch.from_numpy(gauss_int(1.5, 9))[None, None]
    tb = F.conv2d(torch.from_numpy(x_true)[None, None], g1, padding=4)[0, 0].numpy()
    xb = F.conv2d(torch.from_numpy(xr_c)[None, None], g1, padding=4)[0, 0].numpy()
    e_mm = np.sqrt(((xb - tb) ** 2).mean())
    e_mean = np.sqrt(((mean - tb) ** 2).mean())
    check("restoration closer to the truth than the coadd", e_mm < 0.5 * e_mean, f"rmse {e_mean:.2f} -> {e_mm:.2f}")
    # robustness: the satellite trail (unmasked outlier in one exposure)
    x_l2, _ = M.mm_restore(y, v, m, k, x0, robust=False, max_iters=2000, epsilon=1e-5)
    ii = np.arange(size)
    trail = (ii, np.clip((ii * 0.8 + 5).astype(int), 0, size - 1))
    off = blur[trail] < 0.05
    t_l2 = crop(x_l2)[trail][off].mean()
    t_rb = xr_c[trail][off].mean()
    check("Algorithm 3 (Huber) removes the satellite trail", t_rb < 0.1 * t_l2, f"trail level L2 {t_l2:.2f} vs Huber {t_rb:.3f}")
    # acceleration (optional, Biggs & Andrews): at least as close to the fixed point (a 5000-
    # iteration plain run) as the plain run with the same tolerance, in fewer iterations
    ref, _ = M.mm_restore(y, v, m, k, x0, robust=True, max_iters=5000, epsilon=0)
    xa, ia = M.mm_restore(y, v, m, k, x0, robust=True, max_iters=5000, epsilon=1e-5, accelerate=True)
    # inside the field only: the padding (flux from outside the field) is weakly constrained
    # and can grow without bound; its values must not normalise the errors
    ref_c = crop(ref)
    pk = float(ref_c.max())
    e_plain = float(np.abs(xr_c - ref_c).max()) / pk
    e_acc = float(np.abs(crop(xa) - ref_c).max()) / pk
    check("Biggs-Andrews acceleration: as close to the fixed point in fewer iterations",
          ia["iterations"] < info["iterations"] and e_acc <= e_plain * 1.05,
          f"{info['iterations']} -> {ia['iterations']} iterations; max |x - x*|/peak plain {e_plain:.1e}, accelerated {e_acc:.1e}")
    return xr_c, mean, x_true


def test_superresolution():
    size, n, r = 80, 16, 2
    x_true, _, _ = scene(size, r=r, n_stars=25)
    fwhms = RNG.uniform(3.0, 5.0, n)
    # exposures generated from the r-times finer truth with fine PSFs of FWHM r*fwhm
    y, v, m, h_true, _ = make_exposures(x_true, r, n, size, fwhms, RNG.uniform(3, 5, n), gain=2.0, satellite=False)
    # "measured" native PSFs f(t) = D(h_true) on the exposure grid, then Eq. 11 back to r d'
    f_nat = F.avg_pool2d(h_true, r)
    f_nat = f_nat / f_nat.sum((-2, -1), keepdim=True)
    g = M.gaussian_psf_mc(1.1, 2 * r * f_nat.shape[-1] - 1, n_samples=50_000_000)
    hs = []
    for t in range(n):
        h, loss = M.refine_psf(f_nat[t, 0].cpu().numpy(), r, 1.1, g=g, device=DEV)
        hs.append(h)
    h = torch.from_numpy(np.stack(hs))[:, None].to(DEV)
    x0 = M.initial_guess(y, m, h.shape[-1], r)
    xs, info = M.mm_restore(y, v, m, h, x0, r=r, robust=True, max_iters=3000, epsilon=1e-5)
    # the restored latent, re-imaged through D(h_t *) must reproduce the exposures to the noise
    ops = M.Operators(h, r)
    chi = float(sum(((m[a:b] * (y[a:b] - ops.forward(xs, a, b)) ** 2 / v[a:b])).sum() for a, b in ops.ranges())
                / m.sum())
    check("Algorithm 2: reduced chi^2 of the super-resolved fit ~ 1", 0.8 < chi < 1.3, f"chi2/N {chi:.3f}, {info['iterations']} it")
    return xs



def test_batched_psf_solver():
    """refine_psfs (batched) must give each kernel what refine_psf gives it alone."""
    fs = np.stack([gauss_int(w, 15, e=0.1 * i, theta=0.3 * i) for i, w in enumerate((2.8, 3.4, 4.1))])
    g = M.gaussian_psf_mc(1.1, 2 * 2 * 15 - 1, n_samples=20_000_000)
    hb, lb = M.refine_psfs(fs, 2, 1.1, g=g, device=DEV)
    for i in range(3):
        h1, l1 = M.refine_psf(fs[i], 2, 1.1, g=g, device=DEV)
        rel = np.abs(hb[i] - h1).max() / np.abs(h1).max()
        tol = 1e-8 * float((fs[i] ** 2).mean())
        check(f"batched Eq.11 kernel {i} = single solve", lb[i] < tol and rel < 1e-4,
              f"mse {lb[i]:.1e} vs {l1:.1e} (target {tol:.1e}), max rel diff {rel:.1e}")


def test_moffat():
    from astrophoto.exposures import fit_moffat, moffat_image
    true = [1.0, 0.2, -0.15, 3.1, 2.4, 0.6, 2.8]
    img = moffat_image(true, 31)
    img /= img.sum()
    err = np.full_like(img, 2e-5) + 0.01 * img
    noisy = img + RNG.normal(size=img.shape) * err
    mdl, p = fit_moffat(noisy, err)
    got = [p["x0"], p["y0"], p["alpha1"], p["alpha2"], p["beta"]]
    want = true[1:5] + [true[6]]
    rel = [abs(a - b) / max(abs(b), 0.1) for a, b in zip(got, want)]
    check("Moffat fit recovers the parameters", max(rel) < 0.05 and 0.7 < p["chi2_red"] < 1.4,
          f"x0 y0 a1 a2 beta {np.round(got, 3)} vs {want}, chi2 {p['chi2_red']:.2f}")
    pp = [p["A"], p["x0"], p["y0"], p["alpha1"], p["alpha2"], p["theta"], p["beta"]]
    inside = moffat_image(pp, 2 * p["half"] + 1).sum()
    total = np.pi * p["alpha1"] * p["alpha2"] * p["A"] / (p["beta"] - 1)
    check("Moffat kernel holds >= 99.5% of the analytic flux", inside / total >= 0.995,
          f"{inside / total:.4f} within {2 * p['half'] + 1} px")


def test_seeing_groups():
    """With identical PSFs inside each group, the grouped L2 objective equals the full one up
    to a constant, so Algorithm 1 iterates must agree."""
    size, n = 96, 12
    x_true, _, _ = scene(size, n_stars=20)
    fwhms = np.repeat([3.0, 4.5, 6.0], 4)
    y, v, m, k, _ = make_exposures(x_true, 1, n, size, fwhms, RNG.uniform(3, 6, n), gain=2.0, satellite=False)
    # identical PSF inside each group of four
    for g in range(3):
        k[g * 4:(g + 1) * 4] = k[g * 4]
    ks = k.shape[-1]
    x0 = M.initial_guess(y, m, ks)
    xa, _ = M.mm_restore(y, v, m, k, x0, robust=False, max_iters=50, epsilon=0)
    groups = [np.arange(g * 4, (g + 1) * 4) for g in range(3)]
    Yg, Vg, Mg, Kg = M.coadd_groups(*(z.cpu().numpy() for z in (y, v, m, k)), groups)
    T = lambda z: torch.from_numpy(z).to(DEV)
    xb, _ = M.mm_restore(T(Yg), T(Vg), T(Mg), T(Kg), x0, robust=False, max_iters=50, epsilon=0)
    rel = float((xa - xb).abs().max() / xa.abs().max())
    check("seeing groups: identical iterates when members share a PSF", rel < 1e-4, f"max rel diff {rel:.1e}")


def test_mf_data_term():
    """The network's multi-frame data term with the true sky as x must equal the noise
    level (2 rho ~ chi2 ~ 1 per pixel): targets, windows and masks line up."""
    from astrophoto.denoise import mf_data_term
    for s_ in (1, 2):
        H0 = 120                                   # exposure grid
        P = 64                                     # stack patch
        size = H0 * s_
        # truth on the stack grid (drizzle convention), smooth enough for the Lanczos step
        yy, xx = np.mgrid[:size, :size]
        truth = np.zeros((size, size), np.float64)
        for _ in range(40):
            cy, cx = RNG.uniform(10, size - 10, 2)
            truth += RNG.uniform(50, 500) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (1.2 * s_) ** 2))
        truth = np.repeat(truth[None], 3, 0).astype(np.float32)
        k = 25 if s_ == 1 else 30
        G = 3
        Ks = []
        for gi in range(G):
            o = np.arange(k) - (k - 1) / 2
            g = np.exp(-(o[:, None] ** 2 + o[None] ** 2) / (2 * ((1.2 + 0.4 * gi) * s_) ** 2))
            Ks.append(np.repeat((g / g.sum())[None], 3, 0))
        K = torch.from_numpy(np.stack(Ks).astype(np.float32)).to(DEV)
        xt = torch.from_numpy(truth)[None].to(DEV)
        # exposure-grid targets = the forward model of the whole truth + noise
        pred, e0 = M.stack_forward(xt, K, s_)
        n = pred.shape[-1]
        lo = int(round(e0))
        Y = np.zeros((G, H0, H0, 3), np.float32)
        V = np.full((G, H0, H0, 3), 4.0, np.float32)
        Mk = np.zeros((G, H0, H0, 3), np.float32)
        clean = np.moveaxis(pred[0].cpu().numpy(), 1, -1)
        Y[:, lo:lo + n, lo:lo + n] = clean + RNG.normal(size=clean.shape) * 2.0
        Mk[:, lo:lo + n, lo:lo + n] = 1
        mf = {"sets": [{"y": Y, "v": V, "m": Mk, "kernels": K}], "s": s_, "delta": 2.0}
        tw = torch.ones(3, size, size)
        vals = []
        for (py, px) in ((0, 0), (s_ * 10, s_ * 14), (size - P, size - P)):
            py -= py % s_
            px -= px % s_
            d, c = mf_data_term(xt[:, :, py:py + P, px:px + P], [(0, py, px)], mf, tw, P, DEV)
            vals.append(float(d / c))
        check(f"multi-frame data term s={s_}: true sky gives the noise level", all(0.9 < v < 1.1 for v in vals),
              f"2 rho per pixel {np.round(vals, 3)}")



class _FakeExposureSet:
    """Minimal stand-in for exposures.ExposureSet over synthetic full-field exposures."""

    def __init__(self, y, v, m, k):
        self.Y, self.V, self.M, self.K = y, v, m, k
        self.H0, self.W0 = y.shape[-2:]

    def usable(self):
        return list(range(self.Y.shape[0]))

    def kernels(self, idx, model="empirical"):
        return self.K[idx]

    def windows(self, idx, y0, y1, x0, x1):
        sl = (idx, slice(None), slice(y0, y1), slice(x0, x1))
        return self.Y[sl].copy(), self.V[sl].copy(), self.M[sl].copy()


def test_tiling():
    """The tiled restoration (overlapping cutouts, each with its own padded latent) must match
    one restoration of the whole field inside the field: the weakly constrained padding of each
    cutout must not reach the blended result."""
    size, n = 256, 10
    x_true, _, _ = scene(size, n_stars=60)
    fwhms = RNG.uniform(3.0, 5.0, n)
    y, v, m, k, _ = make_exposures(x_true, 1, n, size, fwhms, RNG.uniform(3, 5, n), gain=2.0, satellite=False)
    Y, V, Mk, K = (z.cpu().numpy()[:, 0][:, None].repeat(3, 1) for z in (y, v, m, k))
    es = _FakeExposureSet(Y, V, Mk, K)
    whole, _ = M.restore_cutout(es, 0, size, 0, size, device=DEV, max_iters=300, epsilon=0)
    tiled, info = M.restore(es, tile=160, device=DEV, max_iters=300, epsilon=0)
    pk = float(whole.max())
    e = np.abs(tiled - whole)
    inner = e[16:-16, 16:-16]
    check("tiled restoration = whole-field restoration inside the field", float(inner.max()) / pk < 1e-3,
          f"max diff / peak {float(inner.max()) / pk:.1e}, rms {float(np.sqrt((inner ** 2).mean())) / pk:.1e}, "
          f"{len(info['tiles'])} cutouts")



def test_empirical_psf():
    """empirical_psf on a synthetic field (known elliptical Moffat PSF, 10 s-sub-like sky
    noise): the kernel's wing flux and shape must match the truth (no positive pedestal from
    clipped noise)."""
    from astrophoto.exposures import empirical_psf, moffat_image, fit_moffat
    H = W = 1400
    true_p = [1.0, 0.0, 0.0, 3.4, 3.0, 0.5, 2.8]          # FWHM ~3.7 px, as the Seestar subs
    half = 16
    img = RNG.normal(size=(H, W)) * 30.0                       # sky noise
    xs, ys, fl = [], [], []
    for _ in range(140):
        x, y = RNG.uniform(40, W - 40, 2)
        f = 10 ** RNG.uniform(4.0, 5.3)
        ix, iy = int(round(x)), int(round(y))
        stamp = moffat_image([1.0, x - ix, y - iy] + true_p[3:], 2 * 30 + 1)
        img[iy - 30:iy + 31, ix - 30:ix + 31] += f * stamp / moffat_image([1.0, 0, 0] + true_p[3:], 401).sum()
        xs.append(x), ys.append(y), fl.append(f)
    cat = {"x": np.array(xs), "y": np.array(ys), "flux": np.array(fl), "nn": np.full(len(xs), 1e9),
           "nflux": np.zeros(len(xs)), "peak": np.array(fl)}
    psf, n, extra = empirical_psf(img, np.ones((H, W), bool), cat, half, return_error=True)
    truth = moffat_image(true_p, 2 * half + 1)
    rr = np.hypot(*np.mgrid[:2 * half + 1, :2 * half + 1] - half)
    # compare shapes within the support radius the estimator chose, both normalised there
    sup = rr <= extra["support"]
    t = np.where(sup, truth, 0)
    t /= t.sum()
    wing_est = psf[rr > 2 * 3.7].sum()
    wing_true = t[rr > 2 * 3.7].sum()
    err = np.abs(psf - t).max() / t.max()
    check("empirical PSF: no wing pedestal, shape matches the truth", abs(wing_est - wing_true) < 0.01 and err < 0.03,
          f"{n} stars, support {extra['support']:.1f} px, wing flux beyond 2 FWHM {wing_est:.3f} vs true {wing_true:.3f}, "
          f"max |diff|/peak {err:.3f}")
    mdl, info = fit_moffat(extra["mean"], extra["se"])
    check("Moffat fit to the empirical PSF recovers beta and the axes",
          # the axes are only defined up to naming (theta + 90 deg swaps them): compare sorted
          abs(info["beta"] - 2.8) < 0.3 and np.allclose(sorted([info["alpha1"], info["alpha2"]]), [3.0, 3.4], atol=0.25),
          f"beta {info['beta']:.2f}, alpha {info['alpha1']:.2f}/{info['alpha2']:.2f}, kernel {mdl.shape[0]} px, EE {info['ee_kernel']:.3f}")

if __name__ == "__main__":
    t0 = time.time()
    ALL = [test_operators, test_operator_vs_conv2d, test_stack_forward, test_true_convolution, test_geometry, test_psf_solver, test_restoration,
           test_superresolution, test_batched_psf_solver, test_moffat, test_seeing_groups, test_mf_data_term, test_tiling, test_empirical_psf]
    chosen = [f for f in ALL if not sys.argv[1:] or f.__name__ in sys.argv[1:]]
    for f in chosen:
        f()
    print(f"\n{len(FAILS)} failed: {FAILS}  ({time.time() - t0:.0f}s)")
