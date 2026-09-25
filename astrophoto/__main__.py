"""Command line interface.

    python -m astrophoto run "images/IC 5070_sub"            # full pipeline -> JPG/TIFF
    python -m astrophoto run DIR --device cuda --scale 1.5 --palette hoo
    python -m astrophoto run DIR --deconv imagemm --imagemm-r 2   # ImageMM on the subs, 2x super-resolution
    python -m astrophoto analyse DIR                           # frame quality report only
    python -m astrophoto devices                               # list GPUs
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _progress():
    last = [0.0]

    def cb(i, n, msg):
        now = time.time()
        if now - last[0] > 0.5 or i == n:
            last[0] = now
            pct = 100 * i / max(n, 1)
            sys.stdout.write(f"\r  [{pct:5.1f}%] {msg:<70}")
            sys.stdout.flush()
            if i == n:
                sys.stdout.write("\n")
    return cb


def main(argv=None):
    from .pipeline import DEFAULTS, STACK_DEFAULTS, Session

    ap = argparse.ArgumentParser(prog="astrophoto", description="Seestar astrophotography pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="analyse, stack, denoise, process and export")
    r.add_argument("folder")
    r.add_argument("--workdir", default="output")
    r.add_argument("--mode", default=STACK_DEFAULTS["mode"], choices=["auto", "drizzle", "demosaic"])
    r.add_argument("--scale", type=float, default=STACK_DEFAULTS["scale"])
    r.add_argument("--sensitivity", type=float, default=STACK_DEFAULTS["sensitivity"])
    r.add_argument("--device", default="auto", help="auto | cuda | cuda:N | mps | cpu")
    r.add_argument("--denoise-iters", type=int, default=STACK_DEFAULTS["denoise_iters"])
    r.add_argument("--no-ai-deconv", action="store_true",
                   help="skip the self-supervised deconvolution network (use Richardson-Lucy instead)")
    r.add_argument("--deconv", default=STACK_DEFAULTS["deconv_method"], choices=["network", "imagemm", "none"],
                   help="restoration: N2N deconvolution network, ImageMM (arXiv:2501.03002) on the subs, or none")
    r.add_argument("--imagemm-r", type=int, default=STACK_DEFAULTS["imagemm_r"], help="ImageMM super-resolution factor")
    r.add_argument("--imagemm-sigma", type=float, default=STACK_DEFAULTS["imagemm_sigma"],
                   help="g_sigma of Eq. 11 in latent pixels (0: none for r=1, 1.1 for r>1)")
    r.add_argument("--imagemm-l2", action="store_true", help="L2 loss (Algorithm 1/2) instead of Huber (Algorithm 3)")
    r.add_argument("--imagemm-epsilon", type=float, default=STACK_DEFAULTS["imagemm_epsilon"], help="stopping tolerance")
    r.add_argument("--imagemm-stop", default=STACK_DEFAULTS["imagemm_stop"], choices=["c15", "elementwise"],
                   help="stopping rule: Eq. C15 (the paper) or elementwise mean |u'_k/u'_k-1 - 1|")
    r.add_argument("--imagemm-max-iters", type=int, default=STACK_DEFAULTS["imagemm_max_iters"])
    r.add_argument("--imagemm-psf", default=STACK_DEFAULTS["imagemm_psf"], choices=["empirical", "moffat"])
    r.add_argument("--imagemm-groups", type=int, default=STACK_DEFAULTS["imagemm_groups"],
                   help="0: every sub (the paper); N: N seeing-group coadds")
    r.add_argument("--imagemm-accelerate", action="store_true", help="Biggs-Andrews acceleration (not in the paper)")
    r.add_argument("--imagemm-n2n", action="store_true",
                   help="ImageMM on the even and on the odd subs, combined by a Noise2Noise pass")
    r.add_argument("--network-groups", type=int, default=STACK_DEFAULTS["network_groups"],
                   help="deconvolution network: ImageMM multi-frame likelihood over N seeing groups (0 = off)")
    r.add_argument("--quality", type=int, default=95)
    r.add_argument("--upscale", type=float, default=1.0)
    r.add_argument("--params", help="JSON file or string with processing parameters")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            r.add_argument(f"--{k.replace('_', '-')}", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        elif isinstance(v, (int, float)):
            r.add_argument(f"--{k.replace('_', '-')}", type=float, default=None)
        else:
            r.add_argument(f"--{k.replace('_', '-')}", default=None)
    a = sub.add_parser("analyse", help="frame quality report")
    a.add_argument("folder")
    a.add_argument("--workdir", default="output")
    a.add_argument("--sensitivity", type=float, default=1.0)
    sub.add_parser("devices", help="show compute devices")
    args = ap.parse_args(argv)

    if args.cmd == "devices":
        from .denoise import device_info
        print(json.dumps(device_info(), indent=2))
        return

    s = Session(args.folder, args.workdir)
    if args.cmd == "analyse":
        table = s.run_analysis(args.sensitivity, _progress())
        acc = [f for f in table if f["accepted"]]
        print(f"{len(acc)}/{len(table)} frames accepted")
        for f in table:
            if not f["accepted"] or f["obstructed"] > 0:
                print(f"  {f['name']}: {'REJECT' if not f['accepted'] else 'partial'} "
                      f"{'; '.join(f['reasons']) or f'{f['obstructed'] * 100:.0f}% masked'}")
        return

    proc = {}
    if args.params:
        try:
            proc = json.loads(args.params)
        except json.JSONDecodeError:
            proc = json.load(open(args.params))
    for k in DEFAULTS:
        v = getattr(args, k, None)
        if v is not None:
            proc[k] = v
    stack = {"mode": args.mode, "scale": args.scale, "sensitivity": args.sensitivity,
             "device": args.device, "denoise_iters": args.denoise_iters,
             "ai_deconvolution": not args.no_ai_deconv,
             "deconv_method": "none" if args.no_ai_deconv else args.deconv,
             "imagemm_r": args.imagemm_r, "imagemm_sigma": args.imagemm_sigma, "imagemm_robust": not args.imagemm_l2,
             "imagemm_epsilon": args.imagemm_epsilon, "imagemm_stop": args.imagemm_stop,
             "imagemm_max_iters": args.imagemm_max_iters,
             "imagemm_psf": args.imagemm_psf, "imagemm_groups": args.imagemm_groups,
             "imagemm_accelerate": args.imagemm_accelerate, "imagemm_n2n": args.imagemm_n2n,
             "network_groups": args.network_groups}
    files = s.run_all(stack, proc, progress=_progress(), quality=args.quality, upscale=args.upscale)
    print(json.dumps(files, indent=2))


if __name__ == "__main__":
    main()
