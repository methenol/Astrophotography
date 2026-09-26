"""Benchmark of ImageMM restorations on real data.

Held-out fidelity.  The subs are split into A (even) and B (odd).  A method sees only A
and returns a latent estimate x^ of the sky.  Every sub t of B is then predicted through
its own forward model (Eq. 1 / 10): D H(t) x^.  Because y(t) is independent of x^,

    E[(y - D H x^)^2 / v] - 1 = E[(D H (x^ - x))^2] / v,

so the mean normalised squared residual minus one is an unbiased measure of the
restoration error as seen through each held-out exposure's own PSF (0 = perfect).
It is reported per channel, on sky pixels and on source pixels.

The paper's own quality metrics (Sec. 5.2, 5.3):
* sharpness S_F: mean over the frequency domain of log |FFT(image)| (Krotkov 1988);
* sigma_sky: standard deviation of the background (sep, as in the paper);
* PSNR and SSIM against the coadd;
* aperture photometry: sources detected on the coadd (sep, absolute threshold), fluxes in
  the same elliptical apertures (2.5 Kron radii) on the coadd and on the restoration,
  magnitude differences vs coadd magnitude.

    python experiments/bench_imagemm.py M27 --size 512 --methods imagemm,imagemm_accel,imagemm_moffat,imagemm_g8
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import sep
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from test_exposures import DATA  # noqa: E402
from astrophoto import imagemm as M  # noqa: E402
from astrophoto.pipeline import Session  # noqa: E402

METHODS = {
    # name: restore_cutout options
    "imagemm": {},                                   # Algorithm 3, measured PSFs (the paper)
    "imagemm_l2": {"robust": False},                 # Algorithm 1
    "imagemm_accel": {"accelerate": True},
    "imagemm_moffat": {"psf_model": "moffat"},
    "imagemm_g8": {"n_groups": 8},
    "imagemm_g16": {"n_groups": 16},
    "imagemm_elementwise": {"stop": "elementwise"},
    # Algorithm 2, the paper's sigma for r = 2.  Eq. C15 plateaus at ~9e-5 at r = 2 (4x the unknowns
    # per exposure pixel), so this is scored after a fixed 200 accelerated iterations
    "imagemm_r2": {"r": 2, "sigma": 1.1, "accelerate": True, "max_iters": 200},
    "imagemm_n2n": {"n2n": True},                    # ImageMM on two disjoint halves of A + Noise2Noise
    "network": {"network": 0},                       # N2N deconvolution network (window held out)
    "network_mf": {"network": 8},                    # ... with the ImageMM multi-frame loss, 8 seeing groups
}


def run_network(sess, es, window, groups, dev):
    """The pipeline's N2N denoiser + deconvolution network, trained with the benchmark window
    (and a margin) excluded from every training patch, predicting the window from half A only.
    Returns the prediction averaged onto the 1x reference grid (drizzle 2x2 blocks)."""
    import copy
    import cv2
    from astrophoto.denoise import (Stabiliser, _batch_and_tile, _sky_map, channel_psfs, infer, train_n2n,
                                    train_n2n_deconv)
    from astrophoto.pipeline import _load_fits
    s = int(round(float(sess.meta.get("scale", 1.0))))
    a, b = _load_fits(sess._p("half_a.fits")), _load_fits(sess._p("half_b.fits"))
    full = _load_fits(sess._p("stack.fits"))
    cov = _load_fits(sess._p("coverage.fits"))
    y0, y1, x0, x1 = window
    marg = 64
    mask = cov >= 0.5 * np.percentile(cov[cov > 0], 90)
    mask[max(0, (y0 - marg) * s):(y1 + marg) * s, max(0, (x0 - marg) * s):(x1 + marg) * s] = False
    stab = Stabiliser(a, b)            # a fixed transform (not learned): as the pipeline builds it
    ga, gb = stab.fwd(a), stab.fwd(b)
    batch, tile = _batch_and_tile(dev)
    net = train_n2n(ga, gb, iters=2000, batch=batch, device=dev, sample_mask=mask)
    da, db = infer(net, ga, tile=tile, tta=8), infer(net, gb, tile=tile, tta=8)
    den = stab.inv(0.5 * (da + db))
    sat = sess.meta.get("saturation", 63471.0)
    psfs = channel_psfs(den, sat)
    var = cv2.GaussianBlur(0.5 * (a - b) ** 2, (0, 0), 10 * s)
    var = np.maximum(var, np.percentile(var[::4, ::4], 1, axis=(0, 1)) * 0.5).astype(np.float32)
    unsat = (full.max(-1) < 0.5 * sat).astype(np.uint8)
    weight = cv2.erode(unsat, np.ones((9, 9), np.uint8)).astype(np.float32)
    sky = _sky_map(full, int(64 * s))
    mf = None
    if groups:
        mf = sess.multiframe_targets(groups, 1.1, "empirical")
        for T_ in mf["sets"]:
            T_["kernels"] = torch.as_tensor(T_["kernels"], dtype=torch.float32, device=dev)
    log = lambda i, n, msg: print(msg, flush=True) if i % 100 == 1 or i == n else None
    dnet = train_n2n_deconv(copy.deepcopy(net), da, db, a, b, stab, psfs, var, weight, sky, iters=2000,
                            batch=max(4, batch * 3 // 4), device=dev, sample_mask=mask, mf=mf, progress=log)
    # predict the window (with context) from half A only
    ctx = 64 * s
    Y0, Y1, X0, X1 = y0 * s - ctx, y1 * s + ctx, x0 * s - ctx, x1 * s + ctx
    sharp_a = stab.inv(infer(dnet, da[Y0:Y1, X0:X1], tile=tile, tta=8))[ctx:-ctx, ctx:-ctx]
    print(f"network prediction: finite {np.isfinite(sharp_a).mean():.4f}, range {np.nanmin(sharp_a):.4g} .. "
          f"{np.nanmax(sharp_a):.4g}", flush=True)
    if s > 1:
        sharp_a = sharp_a.reshape((y1 - y0), s, (x1 - x0), s, 3).mean((1, 3))
    return sharp_a - es.sky_ref[y0:y1, x0:x1]


def sharpness(img):
    """S_F (Krotkov 1988): log magnitude of the 2-D Fourier transform, averaged over frequencies."""
    return float(np.mean(np.log(np.abs(np.fft.fft2(img)) + 1e-12)))


def sky_sigma(img):
    return float(sep.Background(np.ascontiguousarray(img, np.float32)).globalrms)


def photometry(coadd, img, thresh):
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


def heldout_chi2(es, idx_b, x, r, kernels_b, window, smask, device):
    """Mean of (y - D H x)^2 / v - 1 over held-out exposures, per channel, sky / sources."""
    y0, y1, x0, x1 = window
    Y, V, Mk = es.windows(idx_b, y0, y1, x0, x1)
    ks = kernels_b.shape[-1]
    X, Yl = M.latent_shape(y1 - y0, x1 - x0, ks, r)
    o = int(round(M.exposure_origin(ks, r)))
    # embed the latent estimate (defined on the window) into the padded latent; outside the
    # window, continue it by edge replication (only the outer PSF radius is affected, which
    # is excluded below)
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
        out[key] = (s / np.maximum(n, 1) - 1).round(4).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--methods", default="imagemm")
    ap.add_argument("--out", default=os.path.join(ROOT, "experiments", "results_imagemm.json"))
    args = ap.parse_args()
    dev = torch.device("mps") if torch.backends.mps.is_available() else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    sess = Session(os.path.join(ROOT, DATA[args.name]), os.path.join(ROOT, "output"))
    es = sess.exposure_set()
    import cv2
    ref = es.ref - es.sky_ref
    L = ref.mean(-1)
    Lo = cv2.dilate(cv2.erode(L, np.ones((15, 15))), np.ones((15, 15)))
    sm = cv2.blur(Lo, (args.size, args.size))
    h2 = args.size // 2 + 64
    sm[:h2], sm[-h2:], sm[:, :h2], sm[:, -h2:] = -np.inf, -np.inf, -np.inf, -np.inf
    cy, cx = np.unravel_index(np.argmax(sm), sm.shape)
    y0, x0 = int(cy - args.size // 2), int(cx - args.size // 2)
    window = (y0, y0 + args.size, x0, x0 + args.size)
    idx = es.usable()
    idx_a, idx_b = idx[0::2], idx[1::2]
    smask = es.smask[window[0]:window[1], window[2]:window[3]]
    coadd = ref[window[0]:window[1], window[2]:window[3]]
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    r_ds = res.setdefault(args.name, {"window": list(window)})

    def report(name, x, t, extra=None, r=1, sigma=None):
        Kb = es.kernels(idx_b, (extra or {}).get("psf_model", "empirical"))
        if r > 1:
            Kb, _ = M.superresolved_kernels(Kb, r, sigma, device=dev)
        row = {"heldout_chi2_excess": heldout_chi2(es, idx_b, x, r, Kb, window, smask, dev),
               "S_F": [sharpness(x[..., c]) for c in range(3)],
               "sigma_sky": [sky_sigma(x[..., c]) for c in range(3)],
               "seconds": round(t, 1)}
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        rng = float(coadd.max() - coadd.min())
        row["PSNR_vs_coadd"] = float(peak_signal_noise_ratio(coadd, x, data_range=rng))
        row["SSIM_vs_coadd"] = float(structural_similarity(coadd, x, data_range=rng, channel_axis=-1))
        mc, dm = photometry(coadd.mean(-1), x.mean(-1), thresh=5 * sky_sigma(coadd.mean(-1)))
        bright = mc <= np.percentile(mc, 50)
        row["photometry_dm_bright_median"] = float(np.median(dm[bright]))
        row["photometry_dm_faint_median"] = float(np.median(dm[~bright]))
        row.update(extra or {})
        r_ds[name] = row
        json.dump(res, open(args.out, "w"), indent=1, default=float)
        np.save(os.path.join(ROOT, "experiments", "cache", f"{args.name}_bench_{name}.npy"), x)
        print(f"{name:18s} held-out chi2 excess sky {row['heldout_chi2_excess']['sky']} src {row['heldout_chi2_excess']['src']} | "
              f"S_F {np.round(row['S_F'], 3)} sigma_sky {np.round(row['sigma_sky'], 3)} | PSNR {row['PSNR_vs_coadd']:.1f} "
              f"SSIM {row['SSIM_vs_coadd']:.3f} | dm bright {row['photometry_dm_bright_median']:+.3f} "
              f"faint {row['photometry_dm_faint_median']:+.3f} | {t:.0f}s", flush=True)

    # reference: the coadd of set A (inverse-variance mean of the A subs) - not a latent
    # estimate, so its held-out chi2 includes its own blur; the paper's metrics compare to it
    t = time.time()
    Ya, Va, Ma = es.windows(idx_a, *window)
    w = np.where(Ma > 0, 1 / Va, 0)
    coadd_a = np.moveaxis((w * Ya).sum(0) / np.maximum(w.sum(0), 1e-30), 0, -1)
    report("coadd_A (not deconvolved)", coadd_a, time.time() - t)
    for m in args.methods.split(","):
        opts = dict(METHODS[m])
        t = time.time()
        r, sigma = opts.pop("r", 1), opts.pop("sigma", None)
        if "network" in opts:
            x = run_network(sess, es, window, opts["network"], dev)
            report(m, x, time.time() - t, {"groups": opts["network"]})
            continue
        if opts.pop("n2n", False):
            parts = [M.restore_cutout(es, *window, idx=idx_a[p::2], device=dev, max_iters=5000, **opts) for p in (0, 1)]
            x, n2n_info = M.n2n_pass(parts[0][0], parts[1][0], iters=2000, patch=128, device=dev)
            extra = {"iterations": [q[1]["iterations"] for q in parts], "converged": [q[1]["converged"] for q in parts]}
        else:
            kern = None
            if r > 1:
                kern, _ = M.superresolved_kernels(es.kernels(idx_a, opts.get("psf_model", "empirical")), r, sigma, device=dev)
            log = (lambda k, c: print(f"  {m} it {k} criterion {c:.2e}", flush=True) if k % 50 == 0 else None)
            mi = opts.pop("max_iters", 5000)
            x, info = M.restore_cutout(es, *window, idx=idx_a, r=r, kernels=kern, device=dev, max_iters=mi,
                                       log=log, **opts)
            extra = {"iterations": info["iterations"], "converged": info["converged"]}
        if r > 1:
            # paper metrics against the coadd need a common grid: the latent's s x s block means
            # (sample j at reference coordinate j / r: pixel i is the mean of samples r i ... r i + r - 1
            # shifted by (r - 1)/2, the co-sited grid; for r = 2 average the 3 x 3 stencil 1/4 1/2 1/4)
            xx = torch.from_numpy(np.ascontiguousarray(np.moveaxis(x, -1, 0)))[None]
            k1 = torch.tensor([0.25, 0.5, 0.25])
            k2 = (k1[:, None] * k1[None, :])[None, None].repeat(3, 1, 1, 1)
            x1 = torch.nn.functional.conv2d(torch.nn.functional.pad(xx, (1, 1, 1, 1), mode="replicate"), k2, groups=3)
            x1 = x1[0, :, ::2, ::2].permute(1, 2, 0).numpy()
            report(m, x1, time.time() - t, {**extra, **opts, "r": r, "sigma": sigma, "metrics_grid": "1x (block mean)"},
                   r=1)
            r_ds[m]["heldout_chi2_excess_r2"] = heldout_chi2(es, idx_b, x, r, M.superresolved_kernels(
                es.kernels(idx_b), r, sigma, device=dev)[0], window, smask, dev)
            json.dump(res, open(args.out, "w"), indent=1, default=float)
            print(f"{m:18s} held-out chi2 excess on its own 2x grid (Eq. 11 kernels): {r_ds[m]['heldout_chi2_excess_r2']}", flush=True)
        else:
            report(m, x, time.time() - t, {**extra, **opts})


if __name__ == "__main__":
    main()
