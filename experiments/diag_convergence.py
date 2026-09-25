"""Convergence of ImageMM (Algorithm 3) on real data: Eq. C15, the elementwise mean |xi - 1|,
the data chi^2 and the distance to the final iterate, per iteration.

    python experiments/diag_convergence.py M27 256 1500
"""
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from test_exposures import DATA  # noqa: E402
from astrophoto import imagemm as M  # noqa: E402
from astrophoto.pipeline import Session  # noqa: E402

name, size, iters = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
sess = Session(os.path.join(ROOT, DATA[name]), os.path.join(ROOT, "output"))
es = sess.exposure_set()
cy, cx = 1853, 1047                                    # the nebula (run_imagemm_cutout centre)
y0, x0 = cy - size // 2, cx - size // 2
idx = es.usable()
dev = torch.device("mps")
Y, V, Mk = es.windows(idx, y0, y0 + size, x0, x0 + size)
K = es.kernels(idx)
y, v, m, k = (torch.from_numpy(z).to(dev) for z in (Y, V, Mk, K))
ops = M.Operators(k, 1)
x = M.initial_guess(y, m, k.shape[-1])
W = torch.where(m > 0, 1 / v, torch.zeros_like(v))
sd = v.sqrt()
frac = ops.adjoint_sum(m, 0, len(idx))
m_eff = (frac / len(idx) > 0.1).float()
M_eff = float(m_eff.sum())
u_prev, hist, snaps = None, [], {}
t0 = time.time()
for it in range(1, iters + 1):
    fx = ops.forward(x, 0, len(idx))
    r = (y - fx) / sd
    Wk = W * M.huber_psi(r, 2.0)
    num = ops.adjoint_sum(Wk * y, 0, len(idx))
    den = ops.adjoint_sum(Wk * fx, 0, len(idx))
    u = torch.where(den > 0, num / den.clamp_min(1e-30), torch.ones_like(den)).clamp(0.5, 2.0)
    x = x * u
    chi2 = float((m * r ** 2).sum() / m.sum())
    row = {"it": it, "chi2": chi2}
    if u_prev is not None:
        xi = u / u_prev
        row["C15"] = abs(float((m_eff * xi).sum()) / M_eff - 1)
        row["mean_abs"] = float((m_eff * (xi - 1).abs()).sum()) / M_eff
        row["clipped_frac"] = float((m_eff * ((u <= 0.5 + 1e-6) | (u >= 2 - 1e-6)).float()).sum()) / M_eff
    hist.append(row)
    u_prev = u
    if it in (14, 30, 50, 100, 200, 300, 500, 700, 1000, 1500, 2000):
        snaps[it] = x.clone()
    if it % 50 == 0:
        print(json.dumps(row), f"{time.time() - t0:.0f}s", flush=True)
final = x
o = k.shape[-1] // 2
field = lambda z: z[0, :, o:o + size, o:o + size]          # the padding is excluded
peak = float(field(final).max())
print("latent peak incl. padding", float(final.max()), "inside the field", peak)
for it, s in snaps.items():
    d = (field(s) - field(final)).abs()
    print(f"iteration {it:5d}: max |x - x_final| / peak {float(d.max()) / peak:.2e}, rms {float(d.pow(2).mean().sqrt()) / peak:.2e}")
first = lambda key, eps: next((h["it"] for h in hist if key in h and h[key] < eps), None)
for eps in (1e-4, 1e-5, 1e-6):
    print(f"eps {eps:.0e}: C15 first satisfied at {first('C15', eps)}, mean|xi-1| at {first('mean_abs', eps)}")
json.dump(hist, open(os.path.join(ROOT, "experiments", "cache", f"{name}_convergence.json"), "w"))
np.save(os.path.join(ROOT, "experiments", "cache", f"{name}_convergence_final.npy"), field(final).permute(1, 2, 0).cpu().numpy())
