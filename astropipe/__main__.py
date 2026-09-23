"""Command line interface.

    python -m astropipe run "images/IC 5070_sub"            # full pipeline -> JPG/TIFF
    python -m astropipe run DIR --device cuda --scale 1.5 --palette hoo
    python -m astropipe analyse DIR                           # frame quality report only
    python -m astropipe devices                               # list GPUs
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

    ap = argparse.ArgumentParser(prog="astropipe", description="Seestar astrophotography pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="analyse, stack, denoise, process and export")
    r.add_argument("folder")
    r.add_argument("--workdir", default="output")
    r.add_argument("--mode", default=STACK_DEFAULTS["mode"], choices=["auto", "drizzle", "demosaic"])
    r.add_argument("--scale", type=float, default=STACK_DEFAULTS["scale"])
    r.add_argument("--sensitivity", type=float, default=STACK_DEFAULTS["sensitivity"])
    r.add_argument("--device", default="auto", help="auto | cuda | cuda:N | mps | cpu")
    r.add_argument("--denoise-iters", type=int, default=STACK_DEFAULTS["denoise_iters"])
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
             "device": args.device, "denoise_iters": args.denoise_iters}
    files = s.run_all(stack, proc, progress=_progress(), quality=args.quality, upscale=args.upscale)
    print(json.dumps(files, indent=2))


if __name__ == "__main__":
    main()
