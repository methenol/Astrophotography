"""How far are the Eq. C15-stopped runs (plain and Biggs-Andrews accelerated) from the
fixed point?  Reference: a very long plain run.  python experiments/diag_accel.py"""
import numpy as np
import torch

import test_imagemm as T
from astrophoto import imagemm as M

size, n = 160, 20
x_true, pos, flux = T.scene(size)
fwhms = T.RNG.uniform(3.0, 6.0, n)
y, v, m, k, _ = T.make_exposures(x_true, 1, n, size, fwhms, T.RNG.uniform(3, 6, n), gain=2.0)
x0 = M.initial_guess(y, m, k.shape[-1])
ref, ri = M.mm_restore(y, v, m, k, x0, robust=True, max_iters=20000, epsilon=0)
print("reference: 20000 plain iterations")
peak = float(ref.max())
for acc in (False, True):
    for eps in (1e-5, 1e-6, 1e-7):
        x, info = M.mm_restore(y, v, m, k, x0, robust=True, max_iters=20000, epsilon=eps, accelerate=acc)
        d = (x - ref).abs()
        print(f"accelerate={acc!s:5} eps={eps:.0e}: {info['iterations']:5d} it, max |x - x*| / peak {float(d.max()) / peak:.2e}, "
              f"rms {float(d.pow(2).mean().sqrt()) / peak:.2e}")
