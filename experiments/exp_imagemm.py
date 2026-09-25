"""ImageMM (arXiv:2501.03002) multi-frame restoration on the held-out benchmark.

Needs a stack made with psf_groups > 1 (see bench.DATASETS "...g").  Every method
sees only the even frames (half A) at test time; Noise2Noise steps train on the
training bands of the A/B pair, exactly like the denoiser benchmark.

usage: python exp_imagemm.py IC5070g --methods mm30_g3,mm30_g1,mm30_g3_raw,...
  mm{K}_g{G}          ImageMM, K iterations (or "auto"), G seeing groups (1 = all merged), + N2N
  ..._raw             ImageMM output of half A without the N2N step
  ..._k{kappa}_h{d}   update clipping kappa, Huber threshold d (0 = plain L2)
  ..._s{sigma}        latent resolution: PSFs refined so that f = h * g_sigma (px)
"""
import argparse
import json
import os
import re

import numpy as np
import torch

import bench
import models
from exp_deconv import SCRATCH, blur_np, channel_psfs, get_denoiser, star_metrics
from astrophoto import imagemm as M
from astrophoto.denoise import Stabiliser, deconv_floor, pick_device, train_n2n


def crop_groups(groups, sl):
    return {"info": groups["info"], "mean": [[m[sl] for m in gm] for gm in groups["mean"]],
            "w": [[m[sl] for m in gm] for gm in groups["w"]]}


def merged(groups):
    """All groups folded into one exposure per parity (a single, mixed PSF)."""
    out = {"info": [{}], "mean": [[]], "w": [[]]}
    for p in (0, 1):
        m, w = M._combine(groups, par=p)
        out["mean"][0].append(m)
        out["w"][0].append(w)
    return out


def parse(m):
    g = dict(iters="auto", G=3, kappa=2.0, huber=2.0, sigma=0.0, raw=m.endswith("_raw"))
    it = re.match(r"mm(auto|\d+)", m).group(1)
    g["iters"] = it if it == "auto" else int(it)
    g["res"] = 0.0
    for key, pat, conv in (("G", r"_g(\d+)", int), ("kappa", r"_k([\d.]+)", float),
                           ("huber", r"_h([\d.]+)", float), ("sigma", r"_s([\d.]+)", float),
                           ("res", r"_r([\d.]+)", float)):
        mm = re.search(pat, m)
        if mm:
            g[key] = conv(mm.group(1))
    return g


def run_net(m, B, groups, law, den_net, ga, psfs, dev):
    import copy
    import cv2
    from astrophoto.denoise import _sky_map, train_n2n_deconv
    G = int(re.search(r"_g(\d+)", m).group(1))
    hess = float(re.search(r"_h([\d.]+)", m).group(1)) if "_h" in m else 0.3
    stab = B["stab"]
    da = models.infer(den_net, ga, device=dev, tta=8)
    db = models.infer(den_net, stab.fwd(B["b"]), device=dev, tta=8)
    a, b, sc = B["a"], B["b"], B["scale"]
    var = cv2.GaussianBlur(0.5 * (a - b) ** 2, (0, 0), 10 * sc)
    var = np.maximum(var, np.percentile(var[::4, ::4], 1, axis=(0, 1)) * 0.5).astype(np.float32)
    unsat = (B["full"].max(-1) < 0.5 * B["sat"]).astype(np.uint8)
    weight = cv2.erode(unsat, np.ones((9, 9), np.uint8)).astype(np.float32)
    sky = _sky_map(B["full"], int(64 * sc))
    mf, extra = None, {}
    if G > 1:
        gp, gfw = M.group_psfs(groups, B["sat"], max_half=psfs.shape[-1] // 2)
        mf = {"groups": groups, "psfs": gp, "law": law}
        extra["group_psf_fwhm"] = gfw
    net = train_n2n_deconv(copy.deepcopy(den_net), da, db, a, b, stab, psfs, var, weight, sky, iters=2000,
                           batch=12, hessian=hess, device=dev, sample_mask=B["train"], mf=mf)
    return stab.inv(models.infer(net, da, device=dev, tta=8)), extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=["IC5070g"])
    ap.add_argument("--methods", default="mmauto_g3,mmauto_g1")
    ap.add_argument("--out", default="results_imagemm.json")
    args = ap.parse_args()
    dev = pick_device()
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for name in args.datasets:
        B = bench.prepare(name)
        r = res.setdefault(name, {})
        groups = crop_groups(M.load_groups(os.path.join(B["dir"], "groups")), B["sl"])
        r["groups"] = groups["info"]
        den_net = get_denoiser(name, B, dev)
        ga = B["stab"].fwd(B["a"])
        den = B["stab"].inv(models.infer(den_net, ga, device=dev, tta=8))
        np.save(os.path.join(SCRATCH, f"{name}_den.npy"), den)
        psfs = channel_psfs(den, B["sat"])
        blur = lambda x: blur_np(x, psfs)
        law = M.noise_law(groups, B["sat"])
        r["noise_law"] = {k: v.tolist() for k, v in law.items()}
        valid = (B["full"].max(-1) < 0.5 * B["sat"])
        import cv2
        valid = cv2.erode(valid.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        gpsf_cache = {}

        def report(m, est, t=0.0, extra=None):
            s = {"fid_" + k: v for k, v in bench.score(est, B, blur=blur).items()}
            s.update(star_metrics(est, den, B["sat"]))
            s["seconds"] = round(t, 1)
            s.update(extra or {})
            r[m] = s
            print(f"{name:8s} {m:26s} fid lin {s['fid_lin']:.3f} str {s['fid_str']:.3f} | FWHM {s['fwhm']:.2f}px "
                  f"ring {s['ring_pct']:+.1f}% moat {s['moat_sigma']:+.1f}σ bgnoise x{s['bg_noise_x']:.2f} | {s['seconds']}s",
                  flush=True)
            json.dump(res, open(args.out, "w"), indent=1, default=float)
            np.save(os.path.join(SCRATCH, f"{name}_{m}.npy"), est.astype(np.float32))

        s = {"fid_" + k: v for k, v in bench.score(den, B).items()}
        s.update(star_metrics(den, den, B["sat"]))
        r["denoised only"] = s
        print(f"{name:8s} {'denoised only':26s} fid lin {s['fid_lin']:.3f} str {s['fid_str']:.3f} | FWHM {s['fwhm']:.2f}px")
        for m in args.methods.split(","):
            if m.startswith("net_"):
                # the production deconvolution network (denoise.train_n2n_deconv): net_g1 = single
                # PSF (as shipped), net_g3 = ImageMM multi-frame likelihood over the seeing groups
                with bench.Timer() as t:
                    est, extra = run_net(m, B, groups, law, den_net, ga, psfs, dev)
                report(m, est, t.dt, extra)
                report(m + "+floor", deconv_floor(est, den, psfs.shape[-1]), t.dt, extra)
                continue
            p = parse(m)
            with bench.Timer() as t:
                gr = groups if p["G"] > 1 else merged(groups)
                key = (p["G"], p["sigma"], p["res"])
                if key not in gpsf_cache:
                    # Moffat-extended PSF of every group (a single merged group for G = 1),
                    # refined to the latent resolution exactly as imagemm.restore does
                    gp, gfw = M.group_psfs(gr, B["sat"])
                    sigma = p["sigma"] or p["res"] * min(float(np.median(f)) for f in gfw) / 2.355
                    if sigma > 0:
                        gp = np.stack([np.stack([M.refine_psf(k, 1, sigma, device=dev) for k in q]) for q in gp])
                    gpsf_cache[key] = (gp, gfw, sigma)
                gp, gfw, sigma = gpsf_cache[key]
                iters = p["iters"]
                extra = {"group_psf_fwhm": gfw, "latent_sigma": sigma}
                if iters == "auto":
                    iters, hist = M.calibrate_iters(gr, gp, law, B["full"], valid, p["kappa"], p["huber"], device=dev)
                    extra["convergence"] = hist
                extra["iters"] = iters
                mm = [M.imagemm(gr, par, gp, law, B["a"] if par == 0 else B["b"], valid, iters=iters,
                                kappa=p["kappa"], huber=p["huber"], device=dev) for par in (0, 1)]
                floored = None
                if p["raw"]:
                    est = mm[0]
                else:
                    stab = Stabiliser(mm[0], mm[1])
                    g0, g1 = stab.fwd(mm[0]), stab.fwd(mm[1])
                    net = train_n2n(g0, g1, iters=2000, batch=16, device=dev, sample_mask=B["train"])
                    est = stab.inv(models.infer(net, g0, device=dev, tta=8))
                    floored = deconv_floor(est, den, gp.shape[-1])
            report(m, est, t.dt, extra)
            if floored is not None:
                report(m + "+floor", floored, t.dt, extra)
        if dev.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
