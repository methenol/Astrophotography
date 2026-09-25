"""Candidate restoration networks and a generic self-supervised trainer.

Architectures (all predict a residual on the variance-stabilised image):
  unet      - the production 3-level U-Net (astrophoto.denoise.UNet)
  unet_res  - wider 4-level U-Net with residual blocks
  nafnet    - NAFNet (Chen et al. 2022, arXiv:2204.04676): LayerNorm + depthwise
              conv + SimpleGate + simplified channel attention, no nonlinear activations
  zsn2n     - ZS-N2N (Mansour & Heckel 2023, arXiv:2303.11253): tiny 2-layer net
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from astrophoto.denoise import UNet, _autocast


# ------------------------------------------------------------------ NAFNet
class LayerNorm2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1, c, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        return (x - mu) / torch.sqrt(var + 1e-6) * self.w + self.b


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c, expand=2):
        super().__init__()
        dw = c * expand
        self.n1 = LayerNorm2d(c)
        self.c1 = nn.Conv2d(c, dw, 1)
        self.c2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1))
        self.c3 = nn.Conv2d(dw // 2, c, 1)
        self.n2 = LayerNorm2d(c)
        self.c4 = nn.Conv2d(c, c * 2, 1)
        self.c5 = nn.Conv2d(c, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.sg(self.c2(self.c1(self.n1(x))))
        y = self.c3(y * self.sca(y))
        x = x + y * self.beta
        y = self.c5(self.sg(self.c4(self.n2(x))))
        return x + y * self.gamma


class NAFNet(nn.Module):
    def __init__(self, ch=3, width=32, enc=(1, 1, 2), mid=2, dec=(1, 1, 1), cin=None):
        super().__init__()
        cin = cin or ch
        self.intro = nn.Conv2d(cin, width, 3, padding=1)
        self.ending = nn.Conv2d(width, ch, 3, padding=1)
        self.encs, self.downs, self.ups, self.decs = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        c = width
        for n in enc:
            self.encs.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, c * 2, 2, stride=2))
            c *= 2
        self.mid = nn.Sequential(*[NAFBlock(c) for _ in range(mid)])
        for n in dec:
            self.ups.append(nn.Sequential(nn.Conv2d(c, c * 2, 1), nn.PixelShuffle(2)))
            c //= 2
            self.decs.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
        self.ch = ch

    def forward(self, x):
        inp = x
        x = self.intro(x)
        skips = []
        for e, d in zip(self.encs, self.downs):
            x = e(x)
            skips.append(x)
            x = d(x)
        x = self.mid(x)
        for u, d, s in zip(self.ups, self.decs, skips[::-1]):
            x = d(u(x) + s)
        return inp[:, : self.ch] + self.ending(x)


# ------------------------------------------------------------------ wider residual U-Net
class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        y = F.leaky_relu(self.c1(x), 0.1)
        y = self.c2(y)
        return F.leaky_relu(y + self.skip(x), 0.1)


class UNetRes(nn.Module):
    def __init__(self, ch=3, base=48, levels=4, cin=None):
        super().__init__()
        cin = cin or ch
        cs = [base * 2 ** i for i in range(levels)]
        self.enc = nn.ModuleList()
        c = cin
        for co in cs:
            self.enc.append(ResBlock(c, co))
            c = co
        self.bott = ResBlock(c, c)
        self.dec = nn.ModuleList()
        for co in cs[::-1]:
            self.dec.append(ResBlock(c + co, co))
            c = co
        self.out = nn.Conv2d(c, ch, 1)
        self.ch = ch

    def forward(self, x):
        inp = x
        skips = []
        for e in self.enc:
            x = e(x)
            skips.append(x)
            x = F.avg_pool2d(x, 2)
        x = self.bott(x)
        for d, s in zip(self.dec, skips[::-1]):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = d(torch.cat([x, s], 1))
        return inp[:, : self.ch] + self.out(x)


# ------------------------------------------------------------------ ZS-N2N
class ZSN2N(nn.Module):
    def __init__(self, ch=3, width=48):
        super().__init__()
        self.c1 = nn.Conv2d(ch, width, 3, padding=1)
        self.c2 = nn.Conv2d(width, width, 3, padding=1)
        self.c3 = nn.Conv2d(width, ch, 1)

    def forward(self, x):
        y = F.leaky_relu(self.c1(x), 0.2)
        y = F.leaky_relu(self.c2(y), 0.2)
        return x + self.c3(y)


def make(arch: str, cin=None) -> nn.Module:
    if arch == "unet":
        return UNet()
    if arch == "unet_res":
        return UNetRes(cin=cin)
    if arch == "nafnet":
        return NAFNet(cin=cin)
    if arch == "nafnet_w":
        return NAFNet(width=48, enc=(2, 2, 4), mid=4, dec=(2, 2, 2), cin=cin)
    if arch == "zsn2n":
        return ZSN2N()
    raise ValueError(arch)


def n_params(net):
    return sum(p.numel() for p in net.parameters())


# ------------------------------------------------------------------ training
def patch_sampler(mask, h, w, patch):
    if mask is None:
        return None
    from scipy.ndimage import minimum_filter
    ok = minimum_filter(mask.astype(np.uint8), size=patch, origin=-(patch // 2))[: h - patch, : w - patch] > 0
    return np.argwhere(ok)


def train(net, ga, gb, mask, iters=2000, patch=128, batch=16, device=None, lr=1e-3, seed=0,
          loss_fn=None, log_every=500, tag=""):
    """Noise2Noise training A<->B on patches fully inside ``mask``.

    ``loss_fn(pred, tgt, inp)`` defaults to MSE.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr / 3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=iters, pct_start=0.15)
    h, w, _ = ga.shape
    ta = torch.from_numpy(ga.transpose(2, 0, 1)).contiguous()
    tb = torch.from_numpy(gb.transpose(2, 0, 1)).contiguous()
    cand = patch_sampler(mask, h, w, patch)
    t0 = time.time()
    for it in range(iters):
        pick = cand[rng.integers(0, len(cand), batch)]
        inp, tgt = [], []
        for (y, x), s in zip(pick, rng.random(batch) < 0.5):
            pa, pb = ta[:, y:y + patch, x:x + patch], tb[:, y:y + patch, x:x + patch]
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
        pred = pred.float()
        loss = loss_fn(pred, tgt, inp) if loss_fn else F.mse_loss(pred, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if log_every and (it % log_every == 0 or it == iters - 1):
            print(f"    [{tag}] {it + 1}/{iters} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    return net.eval()


D4 = [(0, False), (1, False), (2, False), (3, False), (0, True), (1, True), (2, True), (3, True)]


@torch.no_grad()
def infer(net, g, tile=512, overlap=32, device=None, tta=1):
    """Tiled inference; ``tta`` = number of dihedral transforms averaged (1, 2, 4 or 8)."""
    device = device or next(net.parameters()).device
    h, w, c = g.shape
    out = np.zeros((h, w, net_out_ch(net, c)), np.float32)
    acc = np.zeros((h, w, 1), np.float32)
    step = tile - 2 * overlap
    ramp = np.ones(tile, np.float32)
    ramp[:overlap] = np.linspace(0.05, 1, overlap)
    ramp[-overlap:] = np.linspace(1, 0.05, overlap)
    tf = D4[:: 8 // tta] if tta in (1, 2, 4) else D4
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            ya, xa = max(0, y1 - tile), max(0, x1 - tile)
            patch = g[ya:y1, xa:x1]
            ph, pw = patch.shape[:2]
            t = torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1)))[None].to(device)
            padh, padw = (-ph) % 16, (-pw) % 16
            if padh or padw:
                t = F.pad(t, (0, padw, 0, padh), mode="reflect")
            r = 0
            for k, fl in tf:
                tt = torch.rot90(t, k, (2, 3))
                if fl:
                    tt = tt.flip(3)
                with _autocast(device):
                    y = net(tt).float()
                if fl:
                    y = y.flip(3)
                r = r + torch.rot90(y, -k, (2, 3))
            r = (r / len(tf))[0, :, :ph, :pw].cpu().numpy().transpose(1, 2, 0)
            wgt = np.outer(ramp[:ph] if ph == tile else np.ones(ph), ramp[:pw] if pw == tile else np.ones(pw))[..., None]
            out[ya:y1, xa:x1] += r * wgt
            acc[ya:y1, xa:x1] += wgt
    return out / np.maximum(acc, 1e-6)


def net_out_ch(net, c):
    return getattr(net, "ch", c)
