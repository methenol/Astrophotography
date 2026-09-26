"""Default restoration of a whole session through the pipeline, timed, then exported.

    python experiments/run_full.py M27
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from test_exposures import DATA  # noqa: E402
from astrophoto.pipeline import STACK_DEFAULTS, Session  # noqa: E402

name = sys.argv[1]
s = Session(os.path.join(ROOT, DATA[name]), os.path.join(ROOT, "output"))
last = [0.0]


def progress(i, n, msg):
    if time.time() - last[0] > 30 or i == n:
        last[0] = time.time()
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


t = time.time()
s.run_denoise(dict(STACK_DEFAULTS), progress)
t_rest = time.time() - t
meta = json.load(open(s._p("restore_meta.json")))
conv = [q["converged"] for q in meta["tiles"]]
its = [q["iterations"] for q in meta["tiles"]]
print(f"restoration {t_rest / 60:.1f} min; {len(its)} cutouts, iterations {min(its)}-{max(its)} (median "
      f"{sorted(its)[len(its) // 2]}), converged {sum(conv)}/{len(conv)}", flush=True)
t = time.time()
files = s.export({}, quality=95)
print("export", files, f"{time.time() - t:.0f}s", flush=True)
