"""Checks of astrophoto/exposures.py on real subs (the inputs ImageMM needs).

    python experiments/test_exposures.py M27 [n_exposures]
"""
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from astrophoto.exposures import ExposureSet  # noqa: E402
from astrophoto.pipeline import Session, _load_fits  # noqa: E402

DATA = {"M27": "images/M 27_sub", "IC5070": "images/IC 5070_sub", "M31": "images/M 31_sub"}


def reference_1x(sess):
    """The session's coadd on the reference (1x) grid.  A drizzled stack at scale s has
    output pixel u centred on reference (u + 0.5)/s - 0.5, so s x s area averaging
    returns exactly the reference pixel grid for integer s."""
    st = _load_fits(sess._p("stack.fits"))
    s = float(sess.meta.get("scale", 1.0))
    W0, H0 = sess.infos[0].width, sess.infos[0].height
    if abs(s - 1) < 1e-6:
        return st
    return cv2.resize(st, (W0, H0), interpolation=cv2.INTER_AREA)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "M27"
    nexp = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    sess = Session(os.path.join(ROOT, DATA[name]), os.path.join(ROOT, "output"))
    ref = reference_1x(sess)
    es = ExposureSet(sess.infos, sess.analysis, sess.defects, ref, sess.meta["saturation"])
    es.items = es.items[:nexp]
    t = time.time()
    es.prepare(progress=lambda i, n, m: print(f"\r{m}", end="", flush=True))
    print(f"\nprepared {nexp} exposures in {time.time() - t:.0f}s; reference catalogue {len(es.cat['x'])} stars; "
          f"sky model {es.sky_info}")
    for k, p in enumerate(es.params):
        rf = p["refine"]
        fn = p["residual"]
        print(f"  exp {k:3d}: FWHM {p['fwhm']:.2f}  T {np.round(p['T'], 4)} +- {np.round(p['T_err'], 4)}  "
              f"reg: initial {rf['rms_before'] if rf else float('nan'):.3f} px, after refinement {fn['rms_before'] if fn else float('nan'):.3f} px "
              f"(centroid noise {fn['noise_rms'] if fn else float('nan'):.3f}, {fn['n'] if fn else 0} stars, deg {rf['deg'] if rf else '-'})  "
              f"PSF stars {p['psf_stars']}")
    print("photon transfer:", {k: np.round(v, 4).tolist() if hasattr(v, 'tolist') else v for k, v in es.ptc.items()})
    # --- variance check: normalised differences of consecutive exposures, by level
    y0, y1, x0, x1 = es.H0 // 2 - 256, es.H0 // 2 + 256, es.W0 // 2 - 256, es.W0 // 2 + 256
    smask = es.smask[y0:y1, x0:x1]
    zs, levs = [], []
    for k in range(0, nexp - 1, 2):
        ya, va, ma = es.window(k, y0, y1, x0, x1)
        yb, vb, mb = es.window(k + 1, y0, y1, x0, x1)
        ok = (ma[..., 0] > 0) & (mb[..., 0] > 0) & ~smask
        z = (ya - yb) / np.sqrt(va + vb)
        zs.append(z[ok])
        levs.append((es.ref[y0:y1, x0:x1])[ok])
    z = np.concatenate(zs)
    lev = np.concatenate(levs)
    print("normalised pair differences (should be ~1 at every level):")
    for c in range(3):
        q = np.quantile(lev[:, c], [0, .25, .5, .75, .95, 1])
        row = []
        for a, b in zip(q[:-1], q[1:]):
            s = (lev[:, c] >= a) & (lev[:, c] < b)
            row.append(f"{a:7.0f}-{b:<7.0f}: {1.4826 * np.median(np.abs(z[s, c])):.3f}")
        print(f"  ch{c}: " + " | ".join(row))
    # --- background: star-free sky of the background-subtracted exposure
    yk, vk, mk = es.window(0, 0, es.H0, 0, es.W0)
    # select dark sky on a smoothed reference: selecting the darkest pixels of a noisy image
    # would select negative noise and bias the check low
    L = cv2.GaussianBlur(es.ref.mean(-1) - es.sky_ref.mean(-1), (0, 0), 15)
    darkest = (L < np.percentile(L[~es.smask], 20)) & ~es.smask & (mk[..., 0] > 0)
    print("median background-subtracted sky in the darkest 20% (should be ~0 relative to the noise):",
          np.round(np.median(yk[darkest], 0), 2), "noise", np.round(np.sqrt(np.median(vk[darkest], 0)), 2))
    np.save(os.path.join(ROOT, "experiments", "cache", f"{name}_psf_exp0.npy"), np.stack(es.params[0]["psf"]))
    json.dump({"ptc": {k: np.asarray(v).tolist() for k, v in es.ptc.items()}},
              open(os.path.join(ROOT, "experiments", "cache", f"{name}_exposures_check.json"), "w"))


if __name__ == "__main__":
    main()
