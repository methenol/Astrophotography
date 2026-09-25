"""Self-supervised Noise2Noise denoising trained per dataset.

The stacker produces two half-stacks (even/odd frames) that contain the same
signal with *independent* noise.  A CNN trained to map half A -> half B (and
vice versa) with an L2 loss learns E[signal | noisy input] (Lehtinen et al.
2018, "Noise2Noise"), i.e. a denoiser tuned to this exact camera, sky and
integration – no pretrained model, no hallucinated detail from other images.

Training happens in a variance-stabilised domain g(x) = asinh((x-b)/(k*sigma)),
which is invertible, so the denoiser returns linear data and slots into the
linear stage of the pipeline.

The same half-stack pairs also train a *deconvolution* network (two-stage
design of ZS-DeconvNet, Qiao et al. 2024): its input is the denoised half A,
and its output x, blurred by the PSF measured from the stars, must predict
the other, raw half B (chi-squared loss with the measured per-pixel noise).
Because B's noise is independent of everything the network sees, the only
way to lower the loss is to recover the true, sharper sky.  Hessian
regularisation (as in ZS-DeconvNet) and a physical sky-floor prior (real flux
never dips below the local sky, which is what removes the dark "moats"
deconvolution normally leaves around stars) constrain what the PSF cannot see.
Benchmarks and ablations: experiments/README.md.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pick_device(preference: str = "auto") -> torch.device:
    """Choose a compute device: NVIDIA CUDA, Apple Metal (MPS) or CPU.

    ``preference`` may be "auto", "cuda", "cuda:1", "mps" or "cpu".  The
    ``ASTROPHOTO_DEVICE`` environment variable overrides "auto".
    """
    import os
    pref = (preference or "auto").lower()
    if pref == "auto":
        pref = os.environ.get("ASTROPHOTO_DEVICE", "auto").lower()
    if pref.startswith("cuda") and torch.cuda.is_available():
        return torch.device(pref)
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_info() -> dict:
    """Describe available accelerators (shown in the web UI)."""
    info = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(),
            "default": str(pick_device()), "gpus": []}
    if info["cuda"]:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            info["gpus"].append({"id": f"cuda:{i}", "name": p.name, "vram_gb": round(p.total_memory / 2**30, 1)})
    if info["mps"]:
        info["gpus"].append({"id": "mps", "name": "Apple Silicon GPU (Metal)", "vram_gb": None})
    return info


def _batch_and_tile(device: torch.device) -> tuple[int, int]:
    """Size training batch / inference tile to the available VRAM."""
    if device.type == "cuda":
        vram = torch.cuda.get_device_properties(device).total_memory / 2**30
        if vram >= 16:
            return 32, 1024
        if vram >= 8:
            return 16, 768
        return 8, 512
    if device.type == "mps":
        return 16, 512
    return 8, 384


def _autocast(device: torch.device):
    """Mixed precision on CUDA (tensor cores on RTX cards); full precision elsewhere."""
    import contextlib
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


class _Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Compact 3-level U-Net with a residual output (predicts the correction)."""

    def __init__(self, ch=3, base=32):
        super().__init__()
        self.e1 = _Block(ch, base)
        self.e2 = _Block(base, base * 2)
        self.e3 = _Block(base * 2, base * 4)
        self.bott = _Block(base * 4, base * 4)
        self.d3 = _Block(base * 8, base * 2)
        self.d2 = _Block(base * 4, base)
        self.d1 = _Block(base * 2, base)
        self.out = nn.Conv2d(base, ch, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.avg_pool2d(e1, 2))
        e3 = self.e3(F.avg_pool2d(e2, 2))
        b = self.bott(F.avg_pool2d(e3, 2))
        d3 = self.d3(torch.cat([F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False), e1], 1))
        return x + self.out(d1)


class Stabiliser:
    """Invertible asinh variance-stabilising transform, per channel."""

    def __init__(self, a: np.ndarray, b: np.ndarray, k: float = 3.0):
        diff = (a - b)[::4, ::4]
        self.sigma = (1.4826 * np.median(np.abs(diff - np.median(diff, axis=(0, 1))), axis=(0, 1)) / np.sqrt(2)).astype(np.float32)
        self.sigma = np.maximum(self.sigma, 1e-6)
        self.bg = np.median(((a + b) / 2)[::4, ::4], axis=(0, 1)).astype(np.float32)
        self.k = k

    def fwd(self, x):
        return np.arcsinh((x - self.bg) / (self.k * self.sigma)).astype(np.float32)

    def inv(self, g):
        return (np.sinh(g) * self.k * self.sigma + self.bg).astype(np.float32)


def _to_t(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None]


def train_n2n(ga: np.ndarray, gb: np.ndarray, iters: int = 2000, patch: int = 128, batch: int = 16,
              device=None, progress=None, cancel=None, seed: int = 0, sample_mask: np.ndarray | None = None) -> UNet:
    device = device or pick_device()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = UNet().to(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    use_scaler = device.type == "cuda" and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=iters, pct_start=0.15)
    h, w, _ = ga.shape
    ta, tb = torch.from_numpy(ga.transpose(2, 0, 1)).contiguous(), torch.from_numpy(gb.transpose(2, 0, 1)).contiguous()
    # candidate patch corners: only where the patch is inside well-covered data
    if sample_mask is not None:
        m = sample_mask[: h - patch, : w - patch]
        from scipy.ndimage import minimum_filter
        ok = minimum_filter(sample_mask.astype(np.uint8), size=patch, origin=-(patch // 2))[: h - patch, : w - patch] > 0
        cand = np.argwhere(ok)
        if len(cand) < 100:
            cand = np.argwhere(np.ones_like(m, bool))
    else:
        cand = None
    t0 = time.time()
    for it in range(iters):
        if cancel and cancel():
            raise RuntimeError("cancelled")
        if cand is not None:
            pick = cand[rng.integers(0, len(cand), batch)]
            ys, xs = pick[:, 0], pick[:, 1]
        else:
            ys = rng.integers(0, h - patch, batch)
            xs = rng.integers(0, w - patch, batch)
        swap = rng.random(batch) < 0.5
        inp, tgt = [], []
        for y, x, s in zip(ys, xs, swap):
            pa = ta[:, y:y + patch, x:x + patch]
            pb = tb[:, y:y + patch, x:x + patch]
            if s:
                pa, pb = pb, pa
            k = int(rng.integers(0, 4))
            pa, pb = torch.rot90(pa, k, (1, 2)), torch.rot90(pb, k, (1, 2))
            if rng.random() < 0.5:
                pa, pb = pa.flip(2), pb.flip(2)
            inp.append(pa)
            tgt.append(pb)
        inp = torch.stack(inp).to(device)
        tgt = torch.stack(tgt).to(device)
        with _autocast(device):
            pred = net(inp)
        loss = F.mse_loss(pred.float(), tgt)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if progress and (it % 25 == 0 or it == iters - 1):
            progress(it + 1, iters, f"Training Noise2Noise denoiser {it + 1}/{iters} (loss {loss.item():.4f}, {time.time() - t0:.0f}s)")
    return net.eval()


_D4 = [(0, False), (1, False), (2, False), (3, False), (0, True), (1, True), (2, True), (3, True)]


def _edge_weights(n: int, overlap: int, start_inner: bool, end_inner: bool) -> np.ndarray:
    """1-D blending weights for one tile side-by-side with its neighbours.

    Predictions within ~overlap/3 of an interior tile edge are unreliable (the
    network sees reflected padding instead of real sky), so they get weight 0;
    the rest of the overlap is a smooth ramp.  Sides on the true image border
    keep full weight."""
    w = np.ones(n, np.float32)
    dead = overlap // 3
    t = np.linspace(0, 1, overlap - dead, dtype=np.float32)
    ramp = np.concatenate([np.zeros(dead, np.float32), t * t * (3 - 2 * t)])
    if start_inner:
        w[:overlap] = ramp
    if end_inner:
        w[-overlap:] = np.minimum(w[-overlap:], ramp[::-1])
    return w


@torch.no_grad()
def infer(net: nn.Module, g: np.ndarray, tile: int = 512, overlap: int = 96, device=None, tta: int = 1) -> np.ndarray:
    """Tiled inference with feathered overlaps.

    ``tta`` averages the prediction over that many rotations/flips of the input
    (geometric self-ensemble, Timofte et al. 2016, arXiv:1511.02228): free
    extra noise reduction that also removes any orientation bias of the network.
    """
    device = device or next(net.parameters()).device
    h, w, c = g.shape
    tile = min(tile, max(h, w))
    overlap = min(overlap, tile // 3)
    out = np.zeros_like(g)
    acc = np.zeros((h, w, 1), np.float32)
    step = tile - overlap
    tfs = _D4[:: 8 // tta] if tta in (1, 2, 4) else _D4
    ys = sorted({min(y, max(h - tile, 0)) for y in range(0, max(h - overlap, 1), step)})
    xs = sorted({min(x, max(w - tile, 0)) for x in range(0, max(w - overlap, 1), step)})
    for ya in ys:
        for xa in xs:
            y1, x1 = min(ya + tile, h), min(xa + tile, w)
            patch = g[ya:y1, xa:x1]
            ph, pw = patch.shape[:2]
            padh, padw = (-ph) % 8, (-pw) % 8
            t = _to_t(patch).to(device)
            if padh or padw:
                t = F.pad(t, (0, padw, 0, padh), mode="reflect")
            r = 0
            for k, fl in tfs:
                tt = torch.rot90(t, k, (2, 3))
                if fl:
                    tt = tt.flip(3)
                with _autocast(device):
                    y = net(tt).float()
                if fl:
                    y = y.flip(3)
                r = r + torch.rot90(y, -k, (2, 3))
            r = (r / len(tfs))[0, :, :ph, :pw].cpu().numpy().transpose(1, 2, 0)
            wgt = np.outer(_edge_weights(ph, overlap, ya > 0, y1 < h), _edge_weights(pw, overlap, xa > 0, x1 < w))[..., None]
            out[ya:y1, xa:x1] += r * wgt
            acc[ya:y1, xa:x1] += wgt
    return out / np.maximum(acc, 1e-6)


def channel_psfs(img: np.ndarray, sat: float) -> np.ndarray | None:
    """Per-channel empirical PSFs (refractors focus colours differently), padded to one size."""
    from .postprocess import estimate_psf
    psfs = []
    for c in range(img.shape[2]):
        p, _ = estimate_psf(img[..., c], sat)
        if p is None:
            return None
        psfs.append(p)
    n = max(p.shape[0] for p in psfs)
    return np.stack([np.pad(p, (n - p.shape[0]) // 2) for p in psfs]).astype(np.float32)


def _sky_map(img: np.ndarray, box: int) -> np.ndarray:
    import sep
    return np.stack([sep.Background(np.ascontiguousarray(img[..., c], np.float32), bw=box, bh=box).back()
                     for c in range(img.shape[2])], -1).astype(np.float32)


def deconv_floor(sharp: np.ndarray, den: np.ndarray, size: int) -> np.ndarray:
    """Deconvolution may move a star's halo light into its core, but it must never make
    a pixel darker than the local sky unless the data itself is darker there (a dark
    lane).  The sky is a robust local median of the denoised image over ~3 PSF widths
    (stars barely affect a median).  This removes the faint dark disks the stretch
    would otherwise reveal around bright stars and tight groups, where the true PSF is
    wider than the median one used for training."""
    from scipy.ndimage import median_filter
    f = 4
    k = max(5, (3 * size) // f | 1)
    small = cv2.resize(den, (den.shape[1] // f, den.shape[0] // f), interpolation=cv2.INTER_AREA)
    sky = median_filter(small, size=(k, k, 1), mode="reflect")
    sky = cv2.resize(sky, (den.shape[1], den.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.maximum(sharp, np.minimum(den, sky)).astype(np.float32)


def train_n2n_deconv(net: nn.Module, da: np.ndarray, db: np.ndarray, a: np.ndarray, b: np.ndarray,
                     stab: Stabiliser, psfs: np.ndarray, var: np.ndarray, weight: np.ndarray, sky: np.ndarray,
                     iters: int = 2000, patch: int = 128, batch: int = 12, hessian: float = 0.3,
                     floor: float = 3.0, margin: float = 0.5, device=None, progress=None, cancel=None,
                     sample_mask: np.ndarray | None = None, seed: int = 0) -> nn.Module:
    """Self-supervised deconvolution network (see module docstring).

    da/db: denoised halves (stabilised domain) = network inputs; a/b: raw linear halves
    = targets; var: per-pixel noise variance of one half; weight: 0 where the data is
    saturated; sky: smooth per-channel sky level (linear).
    loss = chi2(psf * x, other half) + hessian * |Hessian(g(x))|^2 + floor * |undershoot below sky|^2
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = net.to(device).train()
    k = torch.tensor(stab.k * stab.sigma, device=device).view(1, -1, 1, 1)
    bg = torch.tensor(stab.bg, device=device).view(1, -1, 1, 1)
    inv = lambda g: torch.sinh(g) * k + bg
    T = lambda z: torch.from_numpy(np.ascontiguousarray(z.transpose(2, 0, 1), dtype=np.float32))
    tda, tdb, ta, tb = T(da), T(db), T(a), T(b)
    tw = T(weight[..., None] / var)
    tsky, tsig = T(sky), T(np.sqrt(var / 2))
    kt = torch.from_numpy(psfs[:, None].copy()).to(device)
    r = psfs.shape[1] // 2
    nc = psfs.shape[0]
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    h, w, _ = da.shape
    cand = None
    if sample_mask is not None:
        from scipy.ndimage import minimum_filter
        ok = minimum_filter(sample_mask.astype(np.uint8), size=patch, origin=-(patch // 2))[: h - patch, : w - patch] > 0
        cand = np.argwhere(ok)
        if len(cand) < 100:
            cand = None
    t0 = time.time()
    for it in range(iters):
        if cancel and cancel():
            raise RuntimeError("cancelled")
        if cand is not None:
            pick = cand[rng.integers(0, len(cand), batch)]
        else:
            pick = np.stack([rng.integers(0, h - patch, batch), rng.integers(0, w - patch, batch)], 1)
        parts = [[] for _ in range(5)]
        for (y, x), s in zip(pick, rng.random(batch) < 0.5):
            sl = (slice(None), slice(y, y + patch), slice(x, x + patch))
            src = (tdb, ta) if s else (tda, tb)
            rot = int(rng.integers(0, 4))            # the PSF is 4-fold symmetrised: rotations are exact
            for lst, z in zip(parts, (src[0][sl], src[1][sl], tw[sl], tsky[sl], tsig[sl])):
                lst.append(torch.rot90(z, rot, (1, 2)))
        inp, tgt, wt, skyp, sigp = (torch.stack(z).to(device) for z in parts)
        with _autocast(device):
            g = net(inp)
        g = g.float()
        x = inv(g)
        kx = F.conv2d(F.pad(x, (r, r, r, r), mode="reflect"), kt, groups=nc)
        sl = (slice(None), slice(None), slice(r, -r), slice(r, -r))
        chi2 = ((kx - tgt) ** 2 * wt)[sl].mean() / (wt[sl] > 0).float().mean().clamp_min(1e-3)
        dxx = g[..., :, 2:] - 2 * g[..., :, 1:-1] + g[..., :, :-2]
        dyy = g[..., 2:, :] - 2 * g[..., 1:-1, :] + g[..., :-2, :]
        dxy = g[..., 1:, 1:] - g[..., 1:, :-1] - g[..., :-1, 1:] + g[..., :-1, :-1]
        hess = dxx.pow(2).mean() + dyy.pow(2).mean() + 2 * dxy.pow(2).mean()
        under = F.relu(skyp - margin * sigp - x) / sigp
        loss = chi2 + hessian * hess + floor * under.pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if progress and (it % 25 == 0 or it == iters - 1):
            progress(it + 1, iters, f"Training deconvolution network {it + 1}/{iters} "
                                    f"(chi2 {chi2.item():.3f}, {time.time() - t0:.0f}s)")
    return net.eval()


def n2n_restore(half_a: np.ndarray, half_b: np.ndarray, full: np.ndarray | None = None,
                iters: int = 2000, device: str = "auto", progress=None, cancel=None,
                coverage: np.ndarray | None = None, deconvolve: bool = True,
                deconv_iters: int | None = None, sat: float = 63471.0,
                px_scale: float = 1.0, save_path: str | None = None) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Train on the half-stacks; return (denoised, deconvolved-or-None, info), all linear.

    High-SNR pixels (bright star cores, > ~80 sigma) are rare in training data
    and noise there is invisible, so the denoiser smoothly hands them back to the
    original stack – photometry of stars is preserved exactly.  Saturated stars
    carry no shape information, so the deconvolved image hands those back to the
    denoised one.  ``save_path``: keep the trained networks (a few MB) for re-use.
    """
    import copy
    info = {}
    mask = None
    if coverage is not None:
        mask = coverage >= 0.5 * np.percentile(coverage[coverage > 0], 90)
    if mask is not None and mask.mean() > 0.05:
        ys, xs = np.nonzero(mask[::4, ::4])
        y0, y1, x0, x1 = ys.min() * 4, ys.max() * 4, xs.min() * 4, xs.max() * 4
        stab = Stabiliser(half_a[y0:y1, x0:x1], half_b[y0:y1, x0:x1])
    else:
        mask = None
        stab = Stabiliser(half_a, half_b)
    ga, gb = stab.fwd(half_a), stab.fwd(half_b)
    dev = pick_device(device)
    batch, tile = _batch_and_tile(dev)
    tta = 8 if dev.type != "cpu" else 2
    if dev.type == "cpu":
        iters = min(iters, 600)
    net = train_n2n(ga, gb, iters=iters, batch=batch, device=dev, progress=progress, cancel=cancel,
                    sample_mask=mask)
    if progress:
        progress(0, 2, "Denoising half-stack A")
    da = infer(net, ga, tile=tile, tta=tta)
    if progress:
        progress(1, 2, "Denoising half-stack B")
    db = infer(net, gb, tile=tile, tta=tta)
    del ga, gb
    # ensemble of both independent reconstructions (removes residual noise further)
    g_den = 0.5 * (da + db)
    den = stab.inv(g_den)
    if full is None:
        full = 0.5 * (half_a + half_b)
    from scipy.ndimage import maximum_filter
    g_full = np.abs(stab.fwd(full)).max(axis=2, keepdims=True)
    g_full = maximum_filter(g_full, size=(5, 5, 1))
    keep = np.clip((g_full - 3.5) / 1.5, 0, 1)
    del g_full
    den = (den * (1 - keep) + full * keep).astype(np.float32)
    del keep
    sharp = psfs = dnet = None
    if deconvolve:
        psfs = channel_psfs(den, sat)
        if psfs is None:
            info["deconvolution"] = "skipped: not enough isolated stars to measure the PSF"
        else:
            info["psf_size"] = int(psfs.shape[1])
            var = cv2.GaussianBlur(0.5 * (half_a - half_b) ** 2, (0, 0), 10 * px_scale)
            var = np.maximum(var, np.percentile(var[::4, ::4], 1, axis=(0, 1)) * 0.5).astype(np.float32)
            unsat = (full.max(-1) < 0.5 * sat).astype(np.uint8)
            weight = cv2.erode(unsat, np.ones((9, 9), np.uint8)).astype(np.float32)
            sky = _sky_map(full, int(64 * px_scale))
            dnet = train_n2n_deconv(copy.deepcopy(net), da, db, half_a, half_b, stab, psfs, var, weight, sky,
                                    iters=deconv_iters or (iters if dev.type != "cpu" else min(iters, 400)),
                                    batch=max(4, batch * 3 // 4), device=dev, progress=progress,
                                    cancel=cancel, sample_mask=mask)
            del da, db, sky
            if progress:
                progress(0, 1, "Deconvolving")
            ov = int(min(tile // 3, max(128, 4 * psfs.shape[1])))
            sharp = stab.inv(infer(dnet, g_den, tile=max(tile, 3 * ov), overlap=ov, tta=tta))
            sharp = deconv_floor(sharp, den, psfs.shape[1])
            del var
            # saturated stars: no shape information -> keep the denoised profile there
            satm = cv2.dilate(1 - unsat, np.ones((2 * psfs.shape[1] + 1,) * 2, np.uint8)).astype(np.float32)
            satm = cv2.GaussianBlur(satm, (0, 0), psfs.shape[1] / 3)[..., None]
            sharp = (sharp * (1 - satm) + den * satm).astype(np.float32)
    if save_path:
        torch.save({"denoiser": net.state_dict(), "deconv": dnet.state_dict() if dnet is not None else None,
                    "stab": {"sigma": stab.sigma, "bg": stab.bg, "k": stab.k},
                    "psfs": psfs}, save_path)
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    elif dev.type == "mps":
        torch.mps.empty_cache()
    return den, sharp, info


def n2n_denoise(half_a: np.ndarray, half_b: np.ndarray, full: np.ndarray | None = None,
                iters: int = 2000, device: str = "auto", progress=None, cancel=None,
                coverage: np.ndarray | None = None) -> np.ndarray:
    """Denoise only (backwards-compatible wrapper around :func:`n2n_restore`)."""
    return n2n_restore(half_a, half_b, full, iters, device, progress, cancel, coverage, deconvolve=False)[0]
