"""Prepare (and save) the ImageMM exposures of a session: python experiments/prepare_exposures.py M27"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_exposures import DATA, ROOT, reference_1x  # noqa: E402
from astrophoto.exposures import ExposureSet  # noqa: E402
from astrophoto.pipeline import Session  # noqa: E402

name = sys.argv[1]
sess = Session(os.path.join(ROOT, DATA[name]), os.path.join(ROOT, "output"))
es = ExposureSet(sess.infos, sess.analysis, sess.defects, reference_1x(sess), sess.meta["saturation"])
t = time.time()
es.prepare(progress=lambda i, n, m: print(f"{m} ({time.time() - t:.0f}s)", flush=True) if i % 10 == 0 or i == n else None)
os.makedirs(sess._p("imagemm"), exist_ok=True)
es.save(sess._p("imagemm/exposures.pkl"))
print("usable exposures", len(es.usable()), "of", len(es.items), "ptc", es.ptc, f"{time.time() - t:.0f}s")
