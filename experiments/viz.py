"""Side-by-side crops of benchmark outputs with one shared stretch.

usage: python viz.py DATASET method1 method2 ... [--size 256] [--out file.png]
Methods are cache/<DATASET>_<method>.npy ('den' = denoised only, 'rawA' = half A).
"""
import argparse
import os

import cv2
import numpy as np

import bench
from astrophoto.postprocess import luminance

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("methods", nargs="+")
ap.add_argument("--size", type=int, default=240)
ap.add_argument("--zoom", type=int, default=2)
ap.add_argument("--out", default=None)
args = ap.parse_args()

B = bench.load(args.dataset)
imgs = {}
for m in args.methods:
    imgs[m] = B["a"] if m == "rawA" else np.load(os.path.join("cache", f"{args.dataset}_{m}.npy"))
ref = imgs.get("den", next(iter(imgs.values())))
L = luminance(ref)
bg = np.median(L, axis=(0, 1))
chan_bg = np.median(ref.reshape(-1, 3), 0)
sig = 1.4826 * np.median(np.abs(L - bg))
hi = np.percentile(L, 99.9) - bg


def stretch(x):
    y = np.arcsinh(np.maximum(x - chan_bg + 2 * sig, 0) / (2 * sig)) / np.arcsinh(hi / (2 * sig))
    return np.clip(y, 0, 1)


# pick three crops: brightest structure, a star field, and faint background
h, w = L.shape
s = args.size
# remove stars (morphological opening) so the "bright" crop lands on extended structure
Ls = cv2.dilate(cv2.erode(L.astype(np.float32), np.ones((15, 15))), np.ones((15, 15)))
sm = cv2.blur(Ls, (s, s))
cy, cx = np.unravel_index(np.argmax(sm[s // 2:-s // 2, s // 2:-s // 2]), (h - s, w - s))
cy, cx = cy + s // 2, cx + s // 2
cy2, cx2 = np.unravel_index(np.argmin(sm[s // 2:-s // 2, s // 2:-s // 2]), (h - s, w - s))
cy2, cx2 = cy2 + s // 2, cx2 + s // 2
crops = [(cy, cx), ((cy + cy2) // 2, (cx + cx2) // 2), (cy2, cx2)]
rows = []
for (y, x) in crops:
    tiles = []
    for m, im in imgs.items():
        c = stretch(im[y - s // 2:y + s // 2, x - s // 2:x + s // 2])
        c = cv2.resize(c, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_NEAREST)
        c = (c[..., ::-1] * 255).astype(np.uint8).copy()
        cv2.putText(c, m, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(np.pad(c, ((2, 2), (2, 2), (0, 0))))
    rows.append(np.concatenate(tiles, 1))
out = args.out or f"cmp_{args.dataset}.png"
cv2.imwrite(out, np.concatenate(rows, 0))
print(out)
