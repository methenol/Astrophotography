"""ImageMM (Algorithm 3) on a real cutout, with every prepared sub of the session.

    python experiments/run_imagemm_cutout.py M27 [size] [--r 2] [--sigma 1.1] [--groups N] [--psf moffat] [--accel]

Writes cache/<name>_imagemm_<tag>.npy and a side-by-side PNG (coadd on the reference
grid vs ImageMM latent, one shared asinh stretch).
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from test_exposures import DATA  # noqa: E402
from astrophoto import imagemm as M  # noqa: E402
from astrophoto.pipeline import Session  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("size", nargs="?", type=int, default=384)
ap.add_argument("--r", type=int, default=1)
ap.add_argument("--sigma", type=float, default=None)
ap.add_argument("--groups", type=int, default=0)
ap.add_argument("--psf", default="empirical")
ap.add_argument("--accel", action="store_true")
ap.add_argument("--l2", action="store_true")
ap.add_argument("--center", type=int, nargs=2, default=None, help="y x on the reference grid")
args = ap.parse_args()

sess = Session(os.path.join(ROOT, DATA[args.name]), os.path.join(ROOT, "output"))
es = sess.exposure_set(progress=lambda i, n, m: print(m, flush=True) if i % 20 == 0 else None)
ref = es.ref - es.sky_ref
if args.center:
    cy, cx = args.center
else:                                  # the brightest extended structure (stars removed by an opening)
    L = ref.mean(-1)
    Lo = cv2.dilate(cv2.erode(L, np.ones((15, 15))), np.ones((15, 15)))
    sm = cv2.blur(Lo, (args.size, args.size))
    h2 = args.size // 2 + 64
    sm[:h2], sm[-h2:], sm[:, :h2], sm[:, -h2:] = -np.inf, -np.inf, -np.inf, -np.inf
    cy, cx = np.unravel_index(np.argmax(sm), sm.shape)
y0, x0 = int(cy - args.size // 2), int(cx - args.size // 2)
y1, x1 = y0 + args.size, x0 + args.size
dev = torch.device("mps") if torch.backends.mps.is_available() else None
idx = es.usable()
kern = None
if args.r > 1 or args.sigma is not None:
    sigma = 1.1 if args.sigma is None else args.sigma
    t = time.time()
    kern, mse = M.superresolved_kernels(es.kernels(idx, args.psf), args.r, sigma, device=dev)
    print(f"Eq. 11: {kern.shape[0] * kern.shape[1]} kernels in {time.time() - t:.0f}s, mse max {mse.max():.1e}")
t = time.time()
lat, info = M.restore_cutout(es, y0, y1, x0, x1, idx=idx, r=args.r, kernels=kern, robust=not args.l2,
                             psf_model=args.psf, n_groups=args.groups, accelerate=args.accel, device=dev,
                             max_iters=5000, log=lambda k, c: print(f"  it {k} crit {c:.2e}", flush=True) if k % 50 == 0 else None)
dt = time.time() - t
print(f"window y {y0}:{y1} x {x0}:{x1}; {info['n_exposures']} exposures, PSF {info['psf_size']} px; "
      f"{info['iterations']} iterations, converged {info['converged']}, {dt:.0f}s ({dt / info['iterations']:.2f}s/it)")
tag = f"r{args.r}" + (f"_s{args.sigma}" if args.sigma is not None else "") + (f"_g{args.groups}" if args.groups else "") + \
      (f"_{args.psf}" if args.psf != "empirical" else "") + ("_accel" if args.accel else "") + ("_l2" if args.l2 else "")
os.makedirs(os.path.join(ROOT, "experiments", "cache"), exist_ok=True)
np.save(os.path.join(ROOT, "experiments", "cache", f"{args.name}_imagemm_{tag}.npy"), lat)
# comparison figure: coadd (background-subtracted, same reference grid) vs latent
co = ref[y0:y1, x0:x1]
if args.r > 1:
    co = cv2.resize(co, (lat.shape[1], lat.shape[0]), interpolation=cv2.INTER_NEAREST)
sig = 1.4826 * np.median(np.abs(co - np.median(co, (0, 1))), (0, 1))
hi = np.percentile(co, 99.8, (0, 1))


def stretch(z):
    y = np.arcsinh(np.maximum(z, 0) / (2 * sig)) / np.arcsinh(hi / (2 * sig))
    return (np.clip(y, 0, 1)[..., ::-1] * 255).astype(np.uint8)


out = np.concatenate([stretch(co), np.full((co.shape[0], 6, 3), 255, np.uint8), stretch(lat)], 1)
z = max(1, 900 // out.shape[1])
out = cv2.resize(out, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
fn = os.path.join(ROOT, "experiments", "cache", f"{args.name}_imagemm_{tag}.png")
cv2.imwrite(fn, out)
print(fn)
