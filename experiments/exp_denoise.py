"""Denoiser shoot-out on held-out bands (see bench.py for the metric).

usage: python exp_denoise.py [datasets...] [--methods m1,m2] [--iters N]
"""
import argparse
import json
import os

import numpy as np
import torch

import bench
import models
from astrophoto.denoise import pick_device

METHODS = {
    # name: (arch, iters multiplier, tta, lr)
    "unet":          ("unet", 1, 1, 1e-3),        # production baseline
    "unet+tta8":     ("unet", 1, 8, 1e-3),
    "unet_2x_iters": ("unet", 2, 1, 1e-3),
    "unet_res":      ("unet_res", 1, 1, 1e-3),
    "unet_res+tta8": ("unet_res", 1, 8, 1e-3),
    "nafnet":        ("nafnet", 1, 1, 1e-3),
    "nafnet+tta8":   ("nafnet", 1, 8, 1e-3),
    "nafnet_w":      ("nafnet_w", 1, 1, 1e-3),
    "zsn2n":         ("zsn2n", 1, 1, 1e-3),
}


def nlmeans(B):
    from skimage.restoration import denoise_nl_means
    g = B["stab"].fwd(B["a"])
    sig = float(np.median(np.std(B["stab"].fwd(B["a"]) - B["stab"].fwd(B["b"]), axis=(0, 1))) / np.sqrt(2))
    out = denoise_nl_means(g, h=0.8 * sig, sigma=sig, patch_size=5, patch_distance=6, channel_axis=-1, fast_mode=True)
    return B["stab"].inv(out.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=list(bench.DATASETS))
    ap.add_argument("--methods", default=",".join(["classic"] + list(METHODS)))
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--out", default="results_denoise.json")
    args = ap.parse_args()
    dev = pick_device()
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    cache = {}
    for name in args.datasets:
        B = bench.prepare(name)
        ga, gb = B["stab"].fwd(B["a"]), B["stab"].fwd(B["b"])
        r = res.setdefault(name, {})
        r["raw A"] = bench.score(B["a"], B)
        for m in args.methods.split(","):
            with bench.Timer() as t:
                if m == "classic":
                    est = nlmeans(B)
                    m = "nl-means (classic)"
                else:
                    arch, mult, tta, lr = METHODS[m]
                    key = (arch, mult, lr)
                    if key not in cache:
                        net = models.make(arch)
                        with bench.Timer() as tt:
                            net = models.train(net, ga, gb, B["train"], iters=args.iters * mult, device=dev, lr=lr, tag=f"{name}/{m}")
                        cache[key] = (net, tt.dt)
                    net, ttrain = cache[key]
                    est = B["stab"].inv(models.infer(net, ga, device=dev, tta=tta))
            s = bench.score(est, B)
            s["seconds"] = round(t.dt, 1)
            if m in METHODS:
                s["params"] = models.n_params(cache[(METHODS[m][0], METHODS[m][1], METHODS[m][3])][0])
            r[m] = s
            print(f"{name:7s} {m:22s} lin {s['lin']:.3f} ({s['lin_db']:+.2f} dB)  str {s['str']:.3f} ({s['str_db']:+.2f} dB)  {s['seconds']}s", flush=True)
            json.dump(res, open(args.out, "w"), indent=1)
        cache.clear()
        if dev.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
