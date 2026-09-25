"""Deconvolution / deblurring shoot-out.

Every method sees only half A (plus the PSF measured from it).  Scored on the
held-out bands against half B through the forward model k*x̂ (bench.score with
blur), plus star FWHM, ringing and background-noise amplification.

usage: python exp_deconv.py [datasets...] [--methods ...]
"""
import argparse
import json
import os

import cv2
import numpy as np
import sep
import torch
import torch.nn.functional as F

import bench
import models
from astrophoto.denoise import pick_device, _autocast
from astrophoto.postprocess import estimate_psf, deconvolve, luminance

SCRATCH = os.environ.get("BENCH_CACHE", os.path.join(bench.ROOT, "experiments", "cache"))
os.makedirs(SCRATCH, exist_ok=True)


# ------------------------------------------------------------------ helpers
def channel_psfs(img, sat):
    psfs = []
    for c in range(3):
        p, fw = estimate_psf(img[..., c], sat)
        psfs.append(p)
    n = max(p.shape[0] for p in psfs)
    out = []
    for p in psfs:
        pad = (n - p.shape[0]) // 2
        out.append(np.pad(p, pad))
    return np.stack(out)          # (3, n, n)


def sky_map(img, box=64):
    """Smooth per-channel sky background (SEP mesh, stars/nebula-robust)."""
    return np.stack([sep.Background(np.ascontiguousarray(img[..., c]), bw=box, bh=box).back()
                     for c in range(img.shape[2])], -1).astype(np.float32)


def blur_np(x, psfs):
    return np.stack([cv2.filter2D(x[..., c], -1, psfs[c], borderType=cv2.BORDER_REFLECT) for c in range(3)], -1)


def star_metrics(x, ref_den, sat):
    """Median FWHM (px) of isolated unsaturated stars, ringing depth, background noise."""
    L = np.ascontiguousarray(luminance(x), np.float32)
    Lr = np.ascontiguousarray(luminance(ref_den), np.float32)
    bk = sep.Background(Lr, bw=64, bh=64)
    objs = sep.extract(Lr - bk.back(), 15, err=bk.globalrms, minarea=5)
    ok = (objs["peak"] < 0.4 * sat) & (objs["flag"] == 0) & (objs["a"] / np.maximum(objs["b"], 1e-3) < 1.5)
    xy = np.stack([objs["x"], objs["y"]], 1)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(xy).query(xy, k=2)
    ok &= d[:, 1] > 25
    o = objs[ok]
    bx = sep.Background(L, bw=64, bh=64)
    sub = L - bx.back()
    fw = 2 * sep.flux_radius(sub, o["x"], o["y"], np.full(len(o), 12.0), 0.5, subpix=5)[0]
    # ringing: minimum of the azimuthal profile at 1-4 FWHM relative to peak
    f0 = float(np.nanmedian(fw))
    rings, moat = [], []
    yy, xx = np.mgrid[-20:21, -20:21]
    rr = np.hypot(yy, xx)
    for s in o[np.argsort(-o["peak"])][:80]:
        xi, yi = int(round(s["x"])), int(round(s["y"]))
        if xi < 21 or yi < 21 or xi > L.shape[1] - 22 or yi > L.shape[0] - 22:
            continue
        cut = sub[yi - 20:yi + 21, xi - 20:xi + 21]
        pk = cut[rr <= 1.5].max()
        prof = [cut[(rr >= r) & (rr < r + 1)].mean() for r in np.arange(max(1.0, 0.8 * f0), 4 * f0, 1.0)]
        rings.append(min(prof) / pk)
        moat.append(min(prof) / bk.globalrms)
    # background noise in stretched units, star-free faint pixels
    st = asinh_norm(ref_den)
    faint = (sub < 2 * bk.globalrms) & (cv2.dilate((Lr - bk.back() > 5 * bk.globalrms).astype(np.uint8), np.ones((15, 15))) == 0)
    hp = lambda im: im - cv2.GaussianBlur(im, (0, 0), 2)
    nx = np.std(hp(st(L))[faint])
    nr = np.std(hp(st(Lr))[faint])
    return {"fwhm": f0, "ring_pct": float(100 * np.median(rings)) if rings else float("nan"),
            "moat_sigma": float(np.median(moat)) if moat else float("nan"), "bg_noise_x": float(nx / nr)}


def asinh_norm(ref):
    L = luminance(ref)
    bg = np.median(L)
    s = 1.4826 * np.median(np.abs(L - bg))
    return lambda im: np.arcsinh((im - bg) / (3 * s))


def stab_torch(stab, device):
    k = torch.tensor(stab.k * stab.sigma, device=device).view(1, 3, 1, 1)
    b = torch.tensor(stab.bg, device=device).view(1, 3, 1, 1)
    return (lambda g: torch.sinh(g) * k + b), (lambda x: torch.asinh((x - b) / k))


def get_denoiser(name, B, dev, arch="unet", iters=2000):
    """N2N denoiser trained on the training bands (cached)."""
    path = os.path.join(SCRATCH, f"{name}_{arch}_{iters}.pt")
    net = models.make(arch)
    if os.path.exists(path):
        net.load_state_dict(torch.load(path, map_location="cpu"))
        return net.to(dev).eval()
    ga, gb = B["stab"].fwd(B["a"]), B["stab"].fwd(B["b"])
    net = models.train(net, ga, gb, B["train"], iters=iters, device=dev, tag=f"{name}/denoiser")
    torch.save(net.state_dict(), path)
    return net


# ------------------------------------------------------------------ R0 / R1
def rl_plain(den, psfs, iters=30):
    off = max(0.0, -float(den.min())) + 1.0
    obs = den + off
    est = obs.copy()
    for _ in range(iters):
        conv = blur_np(est, psfs)
        est = est * blur_np(obs / np.maximum(conv, 1e-6), psfs[:, ::-1, ::-1])
    return est - off


# ------------------------------------------------------------------ R2: N2N deconvolution network
def train_n2n_deconv(B, psfs, dev, init=None, iters=2000, patch=128, batch=12, hess=0.05, lr=3e-4, tag="",
                     inputs=None, floor=0.0, margin=0.0):
    """Train g: stab(A) -> x (sharp); loss chi2(k * x, B) + Hessian(stab(x)).

    Genuine Noise2Noise pairs (the two half-stacks) replace the re-corruption
    of ZS-DeconvNet (Qiao et al. 2024); Hessian regularisation as there.
    """
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    net = models.make("unet") if init is None else init
    net = net.to(dev).train()
    inv, fwd = stab_torch(B["stab"], dev)
    ga, gb = inputs if inputs is not None else (B["stab"].fwd(B["a"]), B["stab"].fwd(B["b"]))
    var = B["nm"].var()
    w = (1.0 / var) * (B["full"].max(-1, keepdims=True) < 0.5 * B["sat"])
    w = w * (cv2.erode((B["full"].max(-1) < 0.5 * B["sat"]).astype(np.uint8), np.ones((9, 9)))[..., None] > 0)
    T = lambda z: torch.from_numpy(np.ascontiguousarray(z.transpose(2, 0, 1)))
    tga, tgb, tla, tlb, tw = T(ga), T(gb), T(B["a"]), T(B["b"]), T(w.astype(np.float32))
    # smooth sky level per channel: real flux never goes below it (a noise-free x̂ should not either)
    sky = sky_map(B["full"])
    tsky, tsig = T(sky), T(np.sqrt(var / 2).astype(np.float32))
    kt = torch.from_numpy(psfs[:, None].copy()).to(dev)
    r = psfs.shape[1] // 2
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    h, wd, _ = ga.shape
    cand = models.patch_sampler(B["train"], h, wd, patch)
    for it in range(iters):
        pick = cand[rng.integers(0, len(cand), batch)]
        inp, tgt, wt, skys, sigs = [], [], [], [], []
        for (y, x), s in zip(pick, rng.random(batch) < 0.5):
            sl = (slice(None), slice(y, y + patch), slice(x, x + patch))
            i_, t_ = (tgb, tla) if s else (tga, tlb)
            a_, b_, c_, d_, e_ = i_[sl], t_[sl], tw[sl], tsky[sl], tsig[sl]
            k = int(rng.integers(0, 4))        # PSF is 4-fold symmetrised -> rotations are exact
            a_, b_, c_, d_, e_ = (torch.rot90(z, k, (1, 2)) for z in (a_, b_, c_, d_, e_))
            inp.append(a_); tgt.append(b_); wt.append(c_); skys.append(d_); sigs.append(e_)
        inp, tgt, wt = torch.stack(inp).to(dev), torch.stack(tgt).to(dev), torch.stack(wt).to(dev)
        skyp, sigp = torch.stack(skys).to(dev), torch.stack(sigs).to(dev)
        with _autocast(dev):
            g = net(inp).float()
        x = inv(g)
        kx = F.conv2d(F.pad(x, (r, r, r, r), mode="reflect"), kt, groups=3)
        sl = (slice(None), slice(None), slice(r, -r), slice(r, -r))
        chi2 = ((kx - tgt) ** 2 * wt)[sl].mean() / (wt[sl] > 0).float().mean().clamp_min(1e-3)   # true chi^2 per pixel
        dxx = g[..., :, 2:] - 2 * g[..., :, 1:-1] + g[..., :, :-2]
        dyy = g[..., 2:, :] - 2 * g[..., 1:-1, :] + g[..., :-2, :]
        dxy = g[..., 1:, 1:] - g[..., 1:, :-1] - g[..., :-1, 1:] + g[..., :-1, :-1]
        hl = dxx.pow(2).mean() + dyy.pow(2).mean() + 2 * dxy.pow(2).mean()
        under = F.relu(skyp - margin * sigp - x) / sigp            # undershoot below sky, in noise units
        fl = under.pow(2).mean()
        loss = chi2 + hess * hl + floor * fl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 500 == 0 or it == iters - 1:
            print(f"    [{tag}] {it + 1}/{iters} chi2 {chi2.item():.4f} hess {hl.item():.4f} floor {fl.item():.4f}", flush=True)
    return net.eval()


# ------------------------------------------------------------------ R3: plug-and-play (DPIR) with a noise-conditional N2N denoiser
class CondNet(torch.nn.Module):
    """Denoiser conditioned on a per-pixel noise-level map (FFDNet/DRUNet style input)."""

    def __init__(self):
        super().__init__()
        self.net = models.NAFNet(ch=3, cin=4)

    def forward(self, x):
        return self.net(x)


def train_cond(B, dev, iters=3000, patch=128, batch=12, max_extra=4.0, tag=""):
    """Noisier-input Noise2Noise: input = A + extra N(0, s^2) with level map, target = B.

    The target's noise is still independent, so the network learns E[x | input]
    for every noise level - the prior DPIR needs - from this dataset alone.
    """
    rng = np.random.default_rng(1)
    torch.manual_seed(1)
    ga, gb = B["stab"].fwd(B["a"]), B["stab"].fwd(B["b"])
    base = np.sqrt(cv2.GaussianBlur(0.5 * (ga - gb) ** 2, (0, 0), 10)).mean(-1, keepdims=True).astype(np.float32)
    T = lambda z: torch.from_numpy(np.ascontiguousarray(z.transpose(2, 0, 1)))
    tga, tgb, tbase = T(ga), T(gb), T(base)
    net = CondNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=iters, pct_start=0.15)
    h, w, _ = ga.shape
    cand = models.patch_sampler(B["train"], h, w, patch)
    for it in range(iters):
        pick = cand[rng.integers(0, len(cand), batch)]
        inp, tgt = [], []
        for (y, x), s in zip(pick, rng.random(batch) < 0.5):
            sl = (slice(None), slice(y, y + patch), slice(x, x + patch))
            a_, b_ = (tgb[sl], tga[sl]) if s else (tga[sl], tgb[sl])
            bs = tbase[sl]
            extra = float(rng.uniform(0, max_extra) ** 2 / max_extra) * float(bs.mean())   # favour low levels
            a_ = a_ + extra * torch.randn_like(a_)
            lvl = torch.sqrt(bs ** 2 + extra ** 2)
            k = int(rng.integers(0, 4))
            inp.append(torch.rot90(torch.cat([a_, lvl], 0), k, (1, 2)))
            tgt.append(torch.rot90(b_, k, (1, 2)))
        inp, tgt = torch.stack(inp).to(dev), torch.stack(tgt).to(dev)
        with _autocast(dev):
            pred = net(inp).float()
        loss = F.mse_loss(pred, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 500 == 0 or it == iters - 1:
            print(f"    [{tag}] {it + 1}/{iters} loss {loss.item():.4f}", flush=True)
    return net.eval(), base


@torch.no_grad()
def dpir(B, psfs, cnet, base, dev, n_iter=8, a0=3.0, a1=0.7, lam=1.0, cg_iters=6):
    """Half-quadratic splitting (Zhang et al. 2021, arXiv:2008.13751) with a weighted
    (Poisson-Gaussian) data term solved by conjugate gradients."""
    inv, fwd = stab_torch(B["stab"], dev)
    T = lambda z: torch.from_numpy(np.ascontiguousarray(z.transpose(2, 0, 1)))[None].to(dev)
    y = T(B["a"])
    var = T(B["nm"].var())
    sat_ok = T((B["full"].max(-1, keepdims=True) < 0.5 * B["sat"]).astype(np.float32))
    W = sat_ok / var
    kt = torch.from_numpy(psfs[:, None].copy()).to(dev)
    r = psfs.shape[1] // 2
    K = lambda x: F.conv2d(F.pad(x, (r, r, r, r), mode="reflect"), kt, groups=3)   # symmetric PSF: K^T = K
    tbase = T(base)
    # initial estimate: denoised observation
    g0 = fwd(y)
    z = inv(cnet(torch.cat([g0, tbase], 1)))
    x = z.clone()
    alphas = np.geomspace(a0, a1, n_iter)
    sig_lin = torch.sqrt(var)
    for a in alphas:
        # data step: argmin_x ||K x - y||^2_W + mu ||x - z||^2, mu = lam / (a sigma)^2 per pixel
        mu = lam / (a * sig_lin) ** 2
        A = lambda v: K(W * K(v)) + mu * v
        bvec = K(W * y) + mu * z
        rr = bvec - A(x); p = rr.clone(); rs = (rr * rr).sum()
        for _ in range(cg_iters):
            Ap = A(p)
            al = rs / (p * Ap).sum()
            x = x + al * p
            rr = rr - al * Ap
            rs_new = (rr * rr).sum()
            p = rr + (rs_new / rs) * p
            rs = rs_new
        # prior step: denoise at the noise level implied by mu (in stabilised units)
        gx = fwd(x)
        deriv = 1.0 / (torch.tensor(B["stab"].k * B["stab"].sigma, device=dev).view(1, 3, 1, 1) *
                       torch.sqrt(1 + ((x - torch.tensor(B["stab"].bg, device=dev).view(1, 3, 1, 1)) /
                                       torch.tensor(B["stab"].k * B["stab"].sigma, device=dev).view(1, 3, 1, 1)) ** 2))
        lvl = (a * sig_lin * deriv).mean(1, keepdim=True)
        with _autocast(dev):
            z = inv(cnet(torch.cat([gx, lvl], 1)).float())
    return z[0].cpu().numpy().transpose(1, 2, 0)


def snr_gate(est, den, B, lo=3.0, hi=15.0):
    """Keep the deconvolved estimate only where the data has signal (per-pixel SNR of
    the denoised luminance above the local sky, in units of the full-stack noise)."""
    L = luminance(den)
    bk = sep.Background(np.ascontiguousarray(L, np.float32), bw=64, bh=64).back()
    sig = np.sqrt(luminance(B["nm"].var()) / 2)
    snr = (cv2.GaussianBlur(L, (0, 0), 1.0) - bk) / sig
    t = np.clip((snr - lo) / (hi - lo), 0, 1)
    m = cv2.GaussianBlur((t * t * (3 - 2 * t)).astype(np.float32), (0, 0), 1.0)[..., None]
    return den + m * (est - den)


def parse_hf(m):
    """n2n_deconv_h30_f100 -> hess 0.30, floor 1.00"""
    hess, floor, margin = 0.05, 0.0, 0.0
    for part in m.split("_"):
        if part[:1] == "m" and part[1:].isdigit():
            margin = int(part[1:]) / 100
        if part[:1] == "h" and part[1:].isdigit():
            hess = int(part[1:]) / 100
        if part[:1] == "f" and part[1:].isdigit():
            floor = int(part[1:]) / 100
    return hess, floor, margin


# ------------------------------------------------------------------ main
def tiled(fn, g, tile=512):
    return models.infer(fn, g, tile=tile)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=list(bench.DATASETS))
    ap.add_argument("--methods", default="prod_rl,rl30,n2n_deconv,n2n_deconv_h02,dpir")
    ap.add_argument("--out", default="results_deconv.json")
    args = ap.parse_args()
    dev = pick_device()
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for name in args.datasets:
        B = bench.prepare(name)
        r = res.setdefault(name, {})
        den_net = get_denoiser(name, B, dev)
        ga = B["stab"].fwd(B["a"])
        den = B["stab"].inv(models.infer(den_net, ga, device=dev, tta=8))
        psfs = channel_psfs(den, B["sat"])
        blur = lambda x: blur_np(x, psfs)
        r["psf_px"] = int(psfs.shape[1])

        def report(m, est, t=0.0):
            s = {"fid_" + k: v for k, v in bench.score(est, B, blur=blur).items()}
            s.update(star_metrics(est, den, B["sat"]))
            s["seconds"] = round(t, 1)
            r[m] = s
            print(f"{name:7s} {m:18s} fid lin {s['fid_lin']:.3f} str {s['fid_str']:.3f} | FWHM {s['fwhm']:.2f}px "
                  f"ring {s['ring_pct']:+.1f}% moat {s['moat_sigma']:+.1f}σ bgnoise x{s['bg_noise_x']:.2f} | {s['seconds']}s", flush=True)
            json.dump(res, open(args.out, "w"), indent=1)
            np.save(os.path.join(SCRATCH, f"{name}_{m}.npy"), est.astype(np.float32))

        # reference: the denoised estimate itself is an estimate of k*x, so score it unblurred
        s = {"fid_" + k: v for k, v in bench.score(den, B).items()}
        s.update(star_metrics(den, den, B["sat"]))
        r["denoised only"] = s
        np.save(os.path.join(SCRATCH, f"{name}_den.npy"), den)
        print(f"{name:7s} {'denoised only':18s} fid lin {s['fid_lin']:.3f} str {s['fid_str']:.3f} | FWHM {s['fwhm']:.2f}px ring {s['ring_pct']:+.1f}%", flush=True)
        for m in args.methods.split(","):
            with bench.Timer() as t:
                if m == "prod_rl":
                    est, _ = deconvolve(den, 1.0, B["sat"], noise_ref=float(np.sqrt(np.median(B["nm"].var())) / np.sqrt(2)))
                elif m == "rl30":
                    est = rl_plain(den, psfs, 30)
                elif m.startswith("n2n_deconv2s"):
                    # two-stage (ZS-DeconvNet): input = N2N-denoised half, target = the other RAW half
                    hess, floor, margin = parse_hf(m)
                    dn = get_denoiser(name, B, dev)
                    da = models.infer(dn, ga, device=dev, tta=8)
                    db = models.infer(dn, B["stab"].fwd(B["b"]), device=dev, tta=8)
                    init = get_denoiser(name, B, dev)
                    net = train_n2n_deconv(B, psfs, dev, init=init, hess=hess, tag=f"{name}/{m}", inputs=(da, db), floor=floor, margin=margin)
                    est = B["stab"].inv(models.infer(net, da, device=dev, tta=8))
                elif m.startswith("n2n_deconv"):
                    hess, floor, margin = parse_hf(m)
                    init = get_denoiser(name, B, dev)
                    net = train_n2n_deconv(B, psfs, dev, init=init, hess=hess, floor=floor, margin=margin, tag=f"{name}/{m}")
                    est = B["stab"].inv(models.infer(net, ga, device=dev, tta=8))
                elif m == "dpir":
                    cpath = os.path.join(SCRATCH, f"{name}_cond.pt")
                    cnet = CondNet()
                    if os.path.exists(cpath):
                        cnet.load_state_dict(torch.load(cpath, map_location="cpu"))
                        cnet = cnet.to(dev).eval()
                        gb = B["stab"].fwd(B["b"])
                        base = np.sqrt(cv2.GaussianBlur(0.5 * (ga - gb) ** 2, (0, 0), 10)).mean(-1, keepdims=True).astype(np.float32)
                    else:
                        cnet, base = train_cond(B, dev, tag=f"{name}/cond")
                        torch.save(cnet.state_dict(), cpath)
                    est = dpir(B, psfs, cnet, base, dev)
                else:
                    raise ValueError(m)
            report(m, est, t.dt)
            if m != "prod_rl":
                report(m + "+snrgate", snr_gate(est, den, B), t.dt)
        if dev.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
