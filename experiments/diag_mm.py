"""Quick ImageMM ablation on the raw restoration of half A (no Noise2Noise): which
ingredient causes / removes ringing.  usage: python diag_mm.py IC5070g"""
import os
import sys

import cv2
import numpy as np
import torch

import bench
from exp_deconv import star_metrics
from exp_imagemm import crop_groups
from astrophoto import imagemm as M
from astrophoto.denoise import pick_device

name = sys.argv[1] if len(sys.argv) > 1 else "IC5070g"
B = bench.load(name)
dev = pick_device()
s = 1024
y0 = (B["a"].shape[0] - s) // 2
sub = (slice(y0, y0 + s), slice(y0, y0 + s))
sl = tuple(slice(o.start + i.start, o.start + i.stop) for o, i in zip(B["sl"], sub))
groups = crop_groups(M.load_groups(os.path.join(B["dir"], "groups")), sl)
a = B["a"][sub]
full = B["full"][sub]
valid = cv2.erode((full.max(-1) < 0.5 * B["sat"]).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
law = M.noise_law(groups, B["sat"])
den = cv2.GaussianBlur(full, (0, 0), 0.7)
big = crop_groups(M.load_groups(os.path.join(B["dir"], "groups")), B["sl"])   # enough stars for PSFs
ps_old = M.group_psfs(big, B["sat"], max_half=48)[0]
ps_mof = M.group_psfs(big, B["sat"])[0]
variants = [("moffat", ps_mof, 2.0, True, 30)]
for sg in [float(v) for v in os.environ.get("SIGMAS", "").split(",") if v]:
    ps_s = M.group_psfs(big, B["sat"], sigma=sg, device=dev)[0]
    for it in (30, 80):
        variants.append((f"moffat sigma {sg} it {it}", ps_s, 2.0, True, it))
if os.environ.get("SIGMAS"):
    VARIANTS = variants
orig = M._mm_tile
for label, ps, huber, accel, it in VARIANTS if os.environ.get("SIGMAS") else [("old psf", ps_old, 2.0, True, 30), ("moffat", ps_mof, 2.0, True, 30),
                                    ("moffat, L2", ps_mof, 0.0, True, 30), ("moffat, no accel 30", ps_mof, 2.0, False, 30),
                                    ("moffat, no accel 100", ps_mof, 2.0, False, 100), ("moffat 15", ps_mof, 2.0, True, 15)]:
    M._mm_tile = lambda *q, **k: orig(*q, **{**k, "accel": accel})
    x = M.imagemm(groups, 0, ps, law, a, valid, iters=it, huber=huber, device=dev)
    m = star_metrics(x, den, B["sat"])
    print(f"{label:24s} FWHM {m['fwhm']:.2f} ring {m['ring_pct']:+.1f}% moat {m['moat_sigma']:+.1f}σ bg x{m['bg_noise_x']:.2f}", flush=True)
    np.save(os.path.join("cache", f"diag_{label.replace(' ', '_').replace(',', '')}.npy"), x)
M._mm_tile = orig
if dev.type == "mps":
    torch.mps.empty_cache()
