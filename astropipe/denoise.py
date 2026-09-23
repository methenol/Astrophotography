"""Self-supervised Noise2Noise denoising trained per dataset.

The stacker produces two half-stacks (even/odd frames) that contain the same
signal with *independent* noise.  A CNN trained to map half A -> half B (and
vice versa) with an L2 loss learns E[signal | noisy input] (Lehtinen et al.
2018, "Noise2Noise"), i.e. a denoiser tuned to this exact camera, sky and
integration – no pretrained model, no hallucinated detail from other images.

Training happens in a variance-stabilised domain g(x) = asinh((x-b)/(k*sigma)),
which is invertible, so the denoiser returns linear data and slots into the
linear stage of the pipeline.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pick_device(preference: str = "auto") -> torch.device:
    """Choose a compute device: NVIDIA CUDA, Apple Metal (MPS) or CPU.

    ``preference`` may be "auto", "cuda", "cuda:1", "mps" or "cpu".  The
    ``ASTROPIPE_DEVICE`` environment variable overrides "auto".
    """
    import os
    pref = (preference or "auto").lower()
    if pref == "auto":
        pref = os.environ.get("ASTROPIPE_DEVICE", "auto").lower()
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


@torch.no_grad()
def infer(net: UNet, g: np.ndarray, tile: int = 512, overlap: int = 32, device=None) -> np.ndarray:
    """Tiled inference with feathered overlaps."""
    device = device or next(net.parameters()).device
    h, w, c = g.shape
    out = np.zeros_like(g)
    acc = np.zeros((h, w, 1), np.float32)
    step = tile - 2 * overlap
    ramp = np.ones(tile, np.float32)
    ramp[:overlap] = np.linspace(0.05, 1, overlap)
    ramp[-overlap:] = np.linspace(1, 0.05, overlap)
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            ya, xa = max(0, y1 - tile), max(0, x1 - tile)
            patch = g[ya:y1, xa:x1]
            ph, pw = patch.shape[:2]
            padh, padw = (-ph) % 8, (-pw) % 8
            t = _to_t(patch).to(device)
            if padh or padw:
                t = F.pad(t, (0, padw, 0, padh), mode="reflect")
            with _autocast(device):
                r = net(t)
            r = r[0, :, :ph, :pw].float().cpu().numpy().transpose(1, 2, 0)
            wgt = np.outer(ramp[:ph] if ph == tile else np.ones(ph), ramp[:pw] if pw == tile else np.ones(pw))[..., None]
            out[ya:y1, xa:x1] += r * wgt
            acc[ya:y1, xa:x1] += wgt
    return out / np.maximum(acc, 1e-6)


def n2n_denoise(half_a: np.ndarray, half_b: np.ndarray, full: np.ndarray | None = None,
                iters: int = 2000, device: str = "auto", progress=None, cancel=None,
                coverage: np.ndarray | None = None) -> np.ndarray:
    """Train on the half-stacks and return the denoised linear full stack.

    High-SNR pixels (bright star cores, > ~80 sigma) are rare in training data
    and noise there is invisible, so they are smoothly handed back to the
    original stack – photometry of stars is preserved exactly.
    """
    mask = None
    if coverage is not None:
        mask = coverage >= 0.5 * np.percentile(coverage[coverage > 0], 90)
    if mask is not None and mask.mean() > 0.05:
        ys, xs = np.nonzero(mask[::4, ::4])
        y0, y1, x0, x1 = ys.min() * 4, ys.max() * 4, xs.min() * 4, xs.max() * 4
        stab = Stabiliser(half_a[y0:y1, x0:x1], half_b[y0:y1, x0:x1])
    else:
        stab = Stabiliser(half_a, half_b)
    ga, gb = stab.fwd(half_a), stab.fwd(half_b)
    dev = pick_device(device)
    batch, tile = _batch_and_tile(dev)
    if dev.type == "cpu":
        iters = min(iters, 600)
    net = train_n2n(ga, gb, iters=iters, batch=batch, device=dev, progress=progress, cancel=cancel,
                    sample_mask=mask)
    if progress:
        progress(0, 2, "Denoising half-stack A")
    da = infer(net, ga, tile=tile)
    if progress:
        progress(1, 2, "Denoising half-stack B")
    db = infer(net, gb, tile=tile)
    # ensemble of both independent reconstructions (removes residual noise further)
    den = stab.inv(0.5 * (da + db))
    if full is None:
        full = 0.5 * (half_a + half_b)
    g_full = np.abs(stab.fwd(full)).max(axis=2, keepdims=True)
    from scipy.ndimage import maximum_filter
    g_full = maximum_filter(g_full, size=(5, 5, 1))
    keep = np.clip((g_full - 3.5) / 1.5, 0, 1)
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return (den * (1 - keep) + full * keep).astype(np.float32)
