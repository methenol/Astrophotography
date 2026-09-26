"""Tunable experiments: what a trial runs and how it is scored.

A task declares its parameters (search space, with the pipeline default of each and the
pipeline setting it maps to), its metrics (with the direction of improvement and whether
they need a ground truth) and two functions: ``prepare`` loads everything a study shares
between trials, once; ``run`` executes one trial and returns its metrics and a preview.

Real datasets are the pipeline's own sessions (read only, except the caches the pipeline
itself would write, such as the prepared ImageMM exposures); stacking trials work in the
study's own folder.  Synthetic datasets (``synthetic.py``) are scored against their exact
truth as well.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import time

import cv2
import numpy as np
import torch

from . import metrics as MX
from . import synthetic as SY


# ----------------------------------------------------------------------------- datasets
class Dataset:
    """A real session (``{"kind": "real", "folder": <subs>}``) or a generated synthetic one
    (``{"kind": "synthetic", "dir": <output/lab/synthetic/name>}``)."""

    def __init__(self, spec: dict, workdir: str):
        self.spec = spec
        self.kind = spec["kind"]
        if self.kind == "synthetic":
            self.dir = spec["dir"]
            self.folder = os.path.join(self.dir, "subs")
            self.workdir = os.path.join(self.dir, "work")
        else:
            self.dir = None
            self.folder = spec["folder"]
            self.workdir = workdir
        self.name = spec.get("name") or os.path.basename(os.path.normpath(self.dir or self.folder))

    @property
    def synthetic(self) -> bool:
        return self.kind == "synthetic"

    def session(self, workdir: str | None = None):
        from ..pipeline import Session
        return Session(self.folder, workdir or self.workdir)

    def ensure_stacked(self, log, cancel) -> "object":
        """The dataset's session, analysed and stacked with the pipeline defaults if needed
        (synthetic datasets only: a real session must have been stacked in the pipeline)."""
        s = self.session()
        s.cancel_flag = cancel
        if s.status()["stacked"]:
            return s
        if not self.synthetic:
            raise RuntimeError(f"{self.folder} has not been stacked: run 'Register & integrate' first")
        from ..pipeline import STACK_DEFAULTS
        log("Analysing and stacking the synthetic subs (once per dataset)")
        s.run_analysis(1.0, _prog(log))
        s.run_stack({**STACK_DEFAULTS, **self.spec.get("stack", {})}, _prog(log))
        return s


def _prog(log, every: int = 25):
    def f(i, n, msg):
        if i == 1 or i == n or i % every == 0:
            log(msg)
    return f


def _device(name: str):
    from ..denoise import pick_device
    return pick_device(name or "auto")


def stretch(img: np.ndarray, ref: dict | None = None) -> tuple[np.ndarray, dict]:
    """8-bit preview in an asinh stretch; ``ref`` (from the first trial) keeps it identical
    across the trials of a study."""
    if ref is None:
        L = img.mean(-1)
        med = float(np.median(L))
        sig = float(1.4826 * np.median(np.abs(L - med))) or 1e-6
        hi = float(np.percentile(L, 99.8))
        ref = {"bg": med, "k": 2 * sig, "top": float(np.arcsinh(max(hi - med, 1e-6) / (2 * sig)))}
    y = np.arcsinh(np.maximum(img - ref["bg"], 0) / ref["k"]) / ref["top"]
    return (np.clip(y, 0, 1)[..., ::-1] * 255 + 0.5).astype(np.uint8), ref


def _auto_window(ref: np.ndarray, size: int, where: str = "auto") -> tuple[int, int, int, int]:
    """A size x size window of the reference grid: the brightest extended structure (stars
    removed by a morphological opening) or the centre."""
    H, W = ref.shape[:2]
    size = min(size, H - 140, W - 140)
    if where == "center":
        cy, cx = H // 2, W // 2
    else:
        L = ref.mean(-1)
        Lo = cv2.dilate(cv2.erode(L, np.ones((15, 15))), np.ones((15, 15)))
        sm = cv2.blur(Lo, (size, size))
        h2 = size // 2 + 64
        sm[:h2], sm[-h2:], sm[:, :h2], sm[:, -h2:] = -np.inf, -np.inf, -np.inf, -np.inf
        cy, cx = np.unravel_index(np.argmax(sm), sm.shape)
    y0, x0 = int(cy - size // 2), int(cx - size // 2)
    return y0, y0 + size, x0, x0 + size


def _mean(v):
    v = [q for q in (v or []) if q is not None]
    return float(np.mean(v)) if v else float("nan")


# ----------------------------------------------------------------------------- base
class Task:
    name = ""
    label = ""
    description = ""
    params: list[dict] = []
    metrics: dict[str, dict] = {}
    options: list[dict] = []
    default_objective = ""

    def describe(self) -> dict:
        return {"name": self.name, "label": self.label, "description": self.description, "params": self.params,
                "metrics": self.metrics, "options": self.options, "default_objective": self.default_objective}

    def defaults(self) -> dict:
        return {p["name"]: p["default"] for p in self.params}

    def prepare(self, ds: Dataset, opts: dict, device, log, cancel, study_dir: str) -> dict:
        raise NotImplementedError

    def run(self, ctx: dict, p: dict, log, cancel) -> tuple[dict, np.ndarray | None]:
        raise NotImplementedError


TRUTH_METRICS = {
    "truth_nrmse": {"label": "Truth: normalised rms error", "direction": "minimize", "truth": True},
    "truth_faint_nrmse": {"label": "Truth: error on faint / extended emission", "direction": "minimize", "truth": True},
    "truth_ssim": {"label": "Truth: SSIM (stretched)", "direction": "maximize", "truth": True},
    "truth_psnr": {"label": "Truth: PSNR (dB)", "direction": "maximize", "truth": True},
    "truth_star_dmag_mad": {"label": "Truth: star photometry scatter (mag)", "direction": "minimize", "truth": True},
    "truth_star_dmag_abs": {"label": "Truth: |star photometry bias| (mag)", "direction": "minimize", "truth": True},
}


def _truth_row(m: dict) -> dict:
    out = {f"truth_{k}": v for k, v in m.items() if k in ("nrmse", "faint_nrmse", "ssim", "psnr", "star_dmag_mad")}
    if "star_dmag_median" in m:
        out["truth_star_dmag_abs"] = abs(m["star_dmag_median"])
        out["truth_star_dmag_median"] = m["star_dmag_median"]
    return out


# ----------------------------------------------------------------------------- ImageMM
class ImageMMTask(Task):
    name = "imagemm"
    label = "ImageMM restoration"
    description = ("ImageMM (arXiv:2501.03002) on a window of the reference grid, restored from the even subs; "
                   "scored on the odd subs through each one's own PSF (held-out χ² excess, 0 = perfect), "
                   "with the paper's metrics, and against the exact truth on synthetic data. The held-out score "
                   "predicts data through the measured PSFs, so it rewards fitting the data under the pipeline's "
                   "own model; on synthetic data the truth metrics measure closeness to the real sky.")
    params = [
        {"name": "robust", "label": "Robust (Huber, Algorithm 3)", "type": "bool", "default": True, "tune": True,
         "pipeline": "imagemm_robust"},
        {"name": "delta", "label": "Huber δ", "type": "float", "low": 0.5, "high": 6.0, "default": 2.0, "tune": True,
         "pipeline": "imagemm_delta"},
        {"name": "kappa", "label": "Update clip κ", "type": "float", "low": 1.2, "high": 6.0, "default": 2.0,
         "tune": False, "pipeline": "imagemm_kappa"},
        {"name": "psf_model", "label": "PSF model", "type": "categorical", "choices": ["empirical", "moffat"],
         "default": "empirical", "tune": True, "pipeline": "imagemm_psf"},
        {"name": "n_groups", "label": "Seeing groups (0 = every sub)", "type": "int", "low": 0, "high": 32,
         "default": 0, "tune": True, "pipeline": "imagemm_groups"},
        {"name": "accelerate", "label": "Biggs–Andrews acceleration", "type": "bool", "default": True, "tune": True,
         "pipeline": "imagemm_accelerate"},
        {"name": "stop", "label": "Stopping rule", "type": "categorical", "choices": ["c15", "elementwise"],
         "default": "c15", "tune": False, "pipeline": "imagemm_stop"},
        {"name": "epsilon", "label": "Tolerance ε", "type": "float", "low": 1e-8, "high": 1e-3, "log": True,
         "default": 1e-6, "tune": False, "pipeline": "imagemm_epsilon"},
        {"name": "max_iters", "label": "Max iterations", "type": "int", "low": 50, "high": 5000, "log": True,
         "default": 1000, "tune": False, "pipeline": "imagemm_max_iters"},
        {"name": "r", "label": "Super-resolution r", "type": "categorical", "choices": [1, 2], "default": 1,
         "tune": False, "pipeline": "imagemm_r"},
        {"name": "sigma", "label": "g_σ of Eq. 11 (0 = none at r = 1, 1.1 at r = 2)", "type": "float", "low": 0.0,
         "high": 1.6, "default": 0.0, "tune": False, "pipeline": "imagemm_sigma"},
    ]
    metrics = {
        "heldout_src": {"label": "Held-out χ² excess, sources", "direction": "minimize"},
        "heldout_sky": {"label": "Held-out χ² excess, sky", "direction": "minimize"},
        "seconds": {"label": "Run time (s)", "direction": "minimize"},
        "iterations": {"label": "Iterations", "direction": "minimize"},
        "S_F": {"label": "Sharpness S_F", "direction": "maximize"},
        "sigma_sky": {"label": "σ_sky", "direction": "minimize"},
        "ssim_vs_coadd": {"label": "SSIM vs coadd", "direction": "maximize"},
        **TRUTH_METRICS,
    }
    options = [
        {"name": "window", "label": "Window (px, reference grid)", "type": "int", "default": 256},
        {"name": "where", "label": "Window position", "type": "categorical", "choices": ["auto", "center"],
         "default": "auto"},
        {"name": "sigma_eval", "label": "Truth comparison resolution σ (px)", "type": "float", "default": 1.0},
    ]
    default_objective = "heldout_src"

    def prepare(self, ds, opts, device, log, cancel, study_dir):
        s = ds.ensure_stacked(log, cancel)
        log("Loading the prepared exposures (the first time prepares every sub)")
        es = s.exposure_set(progress=_prog(log, 20))
        ref = es.ref - es.sky_ref
        window = _auto_window(ref, int(opts.get("window", 256)), opts.get("where", "auto"))
        idx = es.usable()
        y0, y1, x0, x1 = window
        ctx = {"es": es, "window": window, "idx_a": idx[0::2], "idx_b": idx[1::2],
               "smask": es.smask[y0:y1, x0:x1], "device": device, "ds": ds, "opts": opts, "truth": {}}
        Ya, Va, Ma = es.windows(ctx["idx_a"], *window)
        w = np.where(Ma > 0, 1 / Va, 0)
        ctx["coadd"] = np.moveaxis((w * Ya).sum(0) / np.maximum(w.sum(0), 1e-30), 0, -1)
        if ds.synthetic:
            ctx["ref_idx"] = s.analysis["ref_idx"]
        log(f"Window y {y0}:{y1} x {x0}:{x1}; {len(ctx['idx_a'])} subs restored, {len(ctx['idx_b'])} held out")
        return ctx

    def _truth(self, ctx, r, sigma):
        key = (r, sigma)
        if key not in ctx["truth"]:
            ds, es = ctx["ds"], ctx["es"]
            psf = SY.Gaussian(sigma) if sigma else None
            H, W = es.H0, es.W0
            full = SY.truth_image(ds.dir, ctx["ref_idx"], (H * r, W * r), float(r), psf, device=ctx["device"]) * r * r
            stars = SY.truth_image(ds.dir, ctx["ref_idx"], (H * r, W * r), float(r), psf, device=ctx["device"],
                                   extended=False) * r * r
            y0, y1, x0, x1 = ctx["window"]
            sl = (slice(y0 * r, y1 * r), slice(x0 * r, x1 * r))
            pos = SY.star_positions(ds.dir, ctx["ref_idx"], float(r))
            pos = {"x": pos["x"] - x0 * r, "y": pos["y"] - y0 * r, "flux": pos["flux"] * r * r}
            ctx["truth"][key] = (full[sl], stars[sl], pos)
        return ctx["truth"][key]

    def run(self, ctx, p, log, cancel):
        from .. import imagemm as M
        es, window, dev = ctx["es"], ctx["window"], ctx["device"]
        r = int(p["r"])
        sigma = float(p["sigma"]) if float(p["sigma"]) > 0 else (1.1 if r > 1 else None)
        # the held-out subs are always predicted through their measured (empirical) PSFs, the
        # same yardstick for every trial whatever PSF model the trial restores with
        kern = None
        kb = es.kernels(ctx["idx_b"], "empirical")
        if r > 1 or sigma:
            log(f"Eq. 11 kernels (r = {r}, σ = {sigma})")
            kern, _ = M.superresolved_kernels(es.kernels(ctx["idx_a"], p["psf_model"]), r, sigma, device=dev)
            kb, _ = M.superresolved_kernels(kb, r, sigma, device=dev)
        n_groups = min(int(p["n_groups"]), len(ctx["idx_a"]))
        def it_log(k, c):
            if cancel.is_set():
                raise RuntimeError("cancelled")
            if k % 100 == 0:
                log(f"iteration {k}, criterion {c:.2e}")
        t = time.time()
        x, info = M.restore_cutout(es, *window, idx=ctx["idx_a"], r=r, kernels=kern, robust=bool(p["robust"]),
                                   psf_model=p["psf_model"], n_groups=n_groups, device=dev,
                                   delta=float(p["delta"]), kappa=float(p["kappa"]), epsilon=float(p["epsilon"]),
                                   stop=p["stop"], max_iters=int(p["max_iters"]), accelerate=bool(p["accelerate"]),
                                   log=it_log)
        dt = time.time() - t
        ch = MX.heldout_chi2(es, ctx["idx_b"], x, r, kb, window, ctx["smask"], dev)   # on the exposure grid
        out = {"heldout_src": _mean(ch["src"]), "heldout_sky": _mean(ch["sky"]), "heldout_src_rgb": ch["src"],
               "heldout_sky_rgb": ch["sky"], "seconds": dt, "iterations": int(info["iterations"]),
               "converged": bool(info["converged"]), "n_groups_used": n_groups}
        x1 = x if r == 1 else x.reshape(x.shape[0] // r, r, x.shape[1] // r, r, 3).mean((1, 3))
        out.update(MX.paper_metrics(x1, ctx["coadd"]))
        if ctx["ds"].synthetic:
            full, stars, pos = self._truth(ctx, r, sigma)
            valid = np.ones(full.shape[:2], bool)
            e = int(4 * r)
            valid[:e], valid[-e:], valid[:, :e], valid[:, -e:] = False, False, False, False
            unsat = (ctx["coadd"].max(-1) < 0.5 * es.sat)
            unsat = cv2.erode(unsat.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            if r > 1:
                unsat = np.repeat(np.repeat(unsat, r, 0), r, 1)
            m = MX.truth_metrics(x, full, valid & unsat, star_truth=stars, stars=pos,
                                 sigma_eval=float(ctx["opts"].get("sigma_eval", 1.0)) * r)
            out.update(_truth_row(m))
        return out, x1


# ----------------------------------------------------------------------------- N2N denoiser
def _deep_crop(cov: np.ndarray, size: int):
    s = min(size, cov.shape[0] - 16, cov.shape[1] - 16)
    s -= s % 16
    c = cv2.blur(cov, (max(s // 4, 3), max(s // 4, 3)))
    c[: s // 2], c[-s // 2:], c[:, : s // 2], c[:, -s // 2:] = 0, 0, 0, 0
    cy, cx = np.unravel_index(np.argmax(c), c.shape)
    y0, x0 = max(0, cy - s // 2), max(0, cx - s // 2)
    return slice(y0, y0 + s), slice(x0, x0 + s)


class DenoiseTask(Task):
    name = "denoise"
    label = "Noise2Noise denoiser"
    description = ("The pipeline's Noise2Noise U-Net trained on the two half-stacks (two of every three 256 px "
                   "bands), applied to half A and scored against half B on the held-out bands "
                   "(error / half-stack noise variance: 1 = no better than raw, lower is better), "
                   "and against the truth on synthetic data.")
    params = [
        {"name": "iters", "label": "Training steps", "type": "int", "low": 250, "high": 8000, "log": True,
         "default": 2000, "tune": True, "pipeline": "denoise_iters"},
        {"name": "max_lr", "label": "Peak learning rate (one-cycle)", "type": "float", "low": 1e-4, "high": 5e-3,
         "log": True, "default": 1e-3, "tune": True},
        {"name": "patch", "label": "Patch size", "type": "categorical", "choices": [64, 96, 128, 192, 256],
         "default": 128, "tune": True},
        {"name": "batch", "label": "Batch size", "type": "int", "low": 4, "high": 32, "default": 16, "tune": False},
        {"name": "base", "label": "U-Net width (first level)", "type": "categorical", "choices": [16, 24, 32, 48],
         "default": 32, "tune": False},
        {"name": "tta", "label": "Self-ensemble (rotations / flips)", "type": "categorical", "choices": [1, 8],
         "default": 8, "tune": False},
    ]
    metrics = {
        "heldout_lin": {"label": "Held-out error, linear (× noise var.)", "direction": "minimize"},
        "heldout_str": {"label": "Held-out error, stretched (× noise var.)", "direction": "minimize"},
        "seconds": {"label": "Run time (s)", "direction": "minimize"},
        **TRUTH_METRICS,
    }
    options = [{"name": "crop", "label": "Crop (px per side, stack grid)", "type": "int", "default": 1024},
               {"name": "sigma_eval", "label": "Truth comparison resolution σ (px)", "type": "float", "default": 1.0}]
    default_objective = "heldout_str"

    def prepare(self, ds, opts, device, log, cancel, study_dir):
        from ..denoise import Stabiliser
        from ..pipeline import _load_fits
        s = ds.ensure_stacked(log, cancel)
        cov = _load_fits(s._p("coverage.fits"))
        sl = _deep_crop(cov, int(opts.get("crop", 1024)))
        a, b = _load_fits(s._p("half_a.fits"))[sl], _load_fits(s._p("half_b.fits"))[sl]
        full = _load_fits(s._p("stack.fits"))[sl]
        cov = cov[sl]
        train, test = MX.split_masks(a.shape)
        good = cov >= 0.6 * np.percentile(cov, 90)
        sat = s.meta.get("saturation", 63471.0)
        unsat = cv2.erode((full.max(-1) < 0.5 * sat).astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
        stab = Stabiliser(a, b)
        ctx = {"a": a, "b": b, "full": full, "train": train & good, "test": test, "valid": good & unsat,
               "stab": stab, "var": MX.NoiseModel(a, b).v, "device": device, "ds": ds, "opts": opts,
               "ga": stab.fwd(a), "gb": stab.fwd(b), "sl": sl, "scale": float(s.meta.get("scale", 1.0))}
        if ds.synthetic:
            fr = [i for i, f in enumerate(s.analysis["frames"]) if f["accepted"]]
            sc = ctx["scale"]
            psf = SY.effective_psf(ds.dir, fr, [s.analysis["frames"][i]["weight"] for i in fr], sc)
            H, W = s.meta["shape"][:2]
            ref = s.analysis["ref_idx"]
            ctx["truth"] = SY.truth_image(ds.dir, ref, (H, W), sc, psf, device=device)[sl]
            ctx["truth_stars"] = SY.truth_image(ds.dir, ref, (H, W), sc, psf, device=device, extended=False)[sl]
            pos = SY.star_positions(ds.dir, ref, sc)
            ctx["truth_pos"] = {"x": pos["x"] - sl[1].start, "y": pos["y"] - sl[0].start, "flux": pos["flux"]}
        log(f"Crop {a.shape[0]}×{a.shape[1]} px of the stack; {int(ctx['train'].mean() * 100)} % of it for training")
        return ctx

    def run(self, ctx, p, log, cancel):
        from ..denoise import _batch_and_tile, infer, train_n2n
        dev = ctx["device"]
        _, tile = _batch_and_tile(dev)
        t = time.time()
        net = train_n2n(ctx["ga"], ctx["gb"], iters=int(p["iters"]), patch=int(p["patch"]), batch=int(p["batch"]),
                        device=dev, progress=_prog(log, 250), cancel=cancel.is_set, sample_mask=ctx["train"],
                        max_lr=float(p["max_lr"]), base=int(p["base"]))
        xa = ctx["stab"].inv(infer(net, ctx["ga"], tile=tile, tta=int(p["tta"])))
        dt = time.time() - t
        out = MX.halfstack_score(xa, ctx["a"], ctx["b"], ctx["test"], ctx["valid"], ctx["stab"], ctx["var"])
        out["seconds"] = dt
        if ctx["ds"].synthetic:
            m = MX.truth_metrics(xa, ctx["truth"], ctx["valid"], star_truth=ctx["truth_stars"], stars=ctx["truth_pos"],
                                 sigma_eval=float(ctx["opts"].get("sigma_eval", 1.0)) * ctx["scale"])
            out.update(_truth_row(m))
        return out, xa


# ----------------------------------------------------------------------------- N2N restoration network
class NetworkTask(Task):
    name = "network"
    label = "Noise2Noise restoration network"
    description = ("The pipeline's N2N denoiser plus deconvolution network, trained with the benchmark window "
                   "(and a margin) excluded, predicting the window from half A only; scored like ImageMM on the "
                   "held-out subs through their own PSFs, and against the truth on synthetic data. "
                   "Needs the prepared ImageMM exposures for scoring.")
    params = [
        {"name": "iters", "label": "Denoiser training steps", "type": "int", "low": 250, "high": 8000, "log": True,
         "default": 2000, "tune": True, "pipeline": "denoise_iters"},
        {"name": "deconv_iters", "label": "Deconvolution training steps", "type": "int", "low": 250, "high": 8000,
         "log": True, "default": 2000, "tune": True},
        {"name": "groups", "label": "Multi-frame loss groups (0 = half-stack loss)", "type": "int", "low": 0,
         "high": 16, "default": 0, "tune": True, "pipeline": "network_groups"},
        {"name": "max_lr", "label": "Denoiser peak learning rate", "type": "float", "low": 1e-4, "high": 5e-3,
         "log": True, "default": 1e-3, "tune": False},
    ]
    metrics = {k: v for k, v in ImageMMTask.metrics.items() if k != "iterations"}
    options = [
        {"name": "window", "label": "Window (px, reference grid)", "type": "int", "default": 256},
        {"name": "where", "label": "Window position", "type": "categorical", "choices": ["auto", "center"],
         "default": "auto"},
        {"name": "train_crop", "label": "Training region (px per side, reference grid; 0 = whole stack)",
         "type": "int", "default": 1536},
        {"name": "sigma_eval", "label": "Truth comparison resolution σ (px)", "type": "float", "default": 1.0},
    ]
    default_objective = "heldout_src"

    def prepare(self, ds, opts, device, log, cancel, study_dir):
        ctx = ImageMMTask().prepare(ds, opts, device, log, cancel, study_dir)
        from ..denoise import Stabiliser
        from ..pipeline import _load_fits
        s = ds.session()
        ctx["session"] = s
        sc = int(round(float(s.meta.get("scale", 1.0))))
        a, b = _load_fits(s._p("half_a.fits")), _load_fits(s._p("half_b.fits"))
        full, cov = _load_fits(s._p("stack.fits")), _load_fits(s._p("coverage.fits"))
        y0, y1, x0, x1 = ctx["window"]
        tc = int(opts.get("train_crop", 1536))
        if tc:
            cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
            H0, W0 = a.shape[0] // sc, a.shape[1] // sc
            Y0, X0 = int(np.clip(cy - tc // 2, 0, max(H0 - tc, 0))), int(np.clip(cx - tc // 2, 0, max(W0 - tc, 0)))
            Y1, X1 = min(H0, Y0 + tc), min(W0, X0 + tc)
        else:
            Y0, X0, Y1, X1 = 0, 0, a.shape[0] // sc, a.shape[1] // sc
        sl = (slice(Y0 * sc, Y1 * sc), slice(X0 * sc, X1 * sc))
        ctx.update(a=a[sl], b=b[sl], full=full[sl], cov=cov[sl], off=(Y0, X0), sc=sc,
                   sat=s.meta.get("saturation", 63471.0))
        ctx["stab"] = Stabiliser(ctx["a"], ctx["b"])
        ctx["ga"], ctx["gb"] = ctx["stab"].fwd(ctx["a"]), ctx["stab"].fwd(ctx["b"])
        return ctx

    def run(self, ctx, p, log, cancel):
        from ..denoise import _batch_and_tile, _sky_map, channel_psfs, infer, train_n2n, train_n2n_deconv
        dev, es, s = ctx["device"], ctx["es"], ctx["sc"]
        y0, y1, x0, x1 = ctx["window"]
        Y0, X0 = ctx["off"]
        a, b, full, cov = ctx["a"], ctx["b"], ctx["full"], ctx["cov"]
        marg = 64
        mask = cov >= 0.5 * np.percentile(cov[cov > 0], 90)
        wy0, wy1, wx0, wx1 = (y0 - Y0) * s, (y1 - Y0) * s, (x0 - X0) * s, (x1 - X0) * s
        mask[max(0, wy0 - marg * s):wy1 + marg * s, max(0, wx0 - marg * s):wx1 + marg * s] = False
        batch, tile = _batch_and_tile(dev)
        t = time.time()
        net = train_n2n(ctx["ga"], ctx["gb"], iters=int(p["iters"]), batch=batch, device=dev, sample_mask=mask,
                        progress=_prog(log, 250), cancel=cancel.is_set, max_lr=float(p["max_lr"]))
        da, db = infer(net, ctx["ga"], tile=tile, tta=8), infer(net, ctx["gb"], tile=tile, tta=8)
        stab = ctx["stab"]
        den = stab.inv(0.5 * (da + db))
        psfs = channel_psfs(den, ctx["sat"])
        if psfs is None:
            raise RuntimeError("not enough isolated stars to measure the PSF")
        var = cv2.GaussianBlur(0.5 * (a - b) ** 2, (0, 0), 10 * s)
        var = np.maximum(var, np.percentile(var[::4, ::4], 1, axis=(0, 1)) * 0.5).astype(np.float32)
        unsat = (full.max(-1) < 0.5 * ctx["sat"]).astype(np.uint8)
        weight = cv2.erode(unsat, np.ones((9, 9), np.uint8)).astype(np.float32)
        sky = _sky_map(full, int(64 * s))
        mf = None
        if int(p["groups"]):
            mf = ctx["session"].multiframe_targets(int(p["groups"]), 1.1, "empirical")
            # the targets cover the whole reference (1x) grid: crop them to the training region,
            # whose origin (Y0, X0) is on that grid (the stack crop starts at s Y0, s X0)
            H0, W0 = es.H0, es.W0
            h1, w1 = a.shape[0] // s, a.shape[1] // s
            sets = []
            for T_ in mf["sets"]:
                T2 = dict(T_)
                for k in ("y", "v", "m"):
                    T2[k] = T_[k][:, Y0:Y0 + h1, X0:X0 + w1]
                    assert T_[k].shape[1:3] == (H0, W0)
                T2["kernels"] = torch.as_tensor(T_["kernels"], dtype=torch.float32, device=dev)
                sets.append(T2)
            mf = {**mf, "sets": sets}
        dnet = train_n2n_deconv(copy.deepcopy(net), da, db, a, b, stab, psfs, var, weight, sky,
                                iters=int(p["deconv_iters"]), batch=max(4, batch * 3 // 4), device=dev,
                                sample_mask=mask, mf=mf, progress=_prog(log, 250), cancel=cancel.is_set)
        ctxp = 64 * s
        Ya, Yb, Xa, Xb = wy0 - ctxp, wy1 + ctxp, wx0 - ctxp, wx1 + ctxp
        if Ya < 0 or Xa < 0 or Yb > a.shape[0] or Xb > a.shape[1]:
            raise RuntimeError("the window needs 64 px of context inside the training region")
        x = stab.inv(infer(dnet, da[Ya:Yb, Xa:Xb], tile=tile, tta=8))[ctxp:-ctxp, ctxp:-ctxp]
        if s > 1:
            x = x.reshape((y1 - y0), s, (x1 - x0), s, 3).mean((1, 3))
        x = x - es.sky_ref[y0:y1, x0:x1]
        dt = time.time() - t
        kb = es.kernels(ctx["idx_b"], "empirical")
        ch = MX.heldout_chi2(es, ctx["idx_b"], x, 1, kb, ctx["window"], ctx["smask"], dev)
        out = {"heldout_src": _mean(ch["src"]), "heldout_sky": _mean(ch["sky"]), "heldout_src_rgb": ch["src"],
               "heldout_sky_rgb": ch["sky"], "seconds": dt}
        out.update(MX.paper_metrics(x, ctx["coadd"]))
        if ctx["ds"].synthetic:
            full_t, stars_t, pos = ImageMMTask()._truth(ctx, 1, None)
            valid = np.ones(full_t.shape[:2], bool)
            valid[:4], valid[-4:], valid[:, :4], valid[:, -4:] = False, False, False, False
            unsat = cv2.erode((ctx["coadd"].max(-1) < 0.5 * es.sat).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            m = MX.truth_metrics(x, full_t, valid & unsat, star_truth=stars_t, stars=pos,
                                 sigma_eval=float(ctx["opts"].get("sigma_eval", 1.0)))
            out.update(_truth_row(m))
        return out, x


# ----------------------------------------------------------------------------- stacking
class StackTask(Task):
    name = "stack"
    label = "Registration & integration"
    description = ("Re-stacks the accepted subs with each trial's settings (in the study's own folder: the "
                   "dataset's session is not touched).  Real data: background noise and star FWHM (both on the "
                   "native pixel scale); synthetic data also against the truth seen through the subs' mean PSF.")
    params = [
        {"name": "sigma_low", "label": "Rejection σ low", "type": "float", "low": 1.5, "high": 8.0, "default": 4.0,
         "tune": True, "pipeline": "sigma_low"},
        {"name": "sigma_high", "label": "Rejection σ high", "type": "float", "low": 1.5, "high": 8.0, "default": 3.0,
         "tune": True, "pipeline": "sigma_high"},
        {"name": "local_norm", "label": "Local normalisation", "type": "bool", "default": True, "tune": True,
         "pipeline": "local_norm"},
        {"name": "sensitivity", "label": "Frame rejection sensitivity", "type": "float", "low": 0.5, "high": 2.0,
         "default": 1.0, "tune": False, "pipeline": "sensitivity"},
        {"name": "mode", "label": "Resampling", "type": "categorical", "choices": ["auto", "drizzle", "demosaic"],
         "default": "auto", "tune": False, "pipeline": "mode"},
        {"name": "scale", "label": "Output scale", "type": "categorical", "choices": [1.0, 1.5, 2.0], "default": 1.0,
         "tune": False, "pipeline": "scale"},
    ]
    metrics = {
        "bg_noise": {"label": "Background noise (native pixels)", "direction": "minimize"},
        "fwhm": {"label": "Star FWHM (native pixels)", "direction": "minimize"},
        "seconds": {"label": "Run time (s)", "direction": "minimize"},
        **TRUTH_METRICS,
    }
    options = [{"name": "crop", "label": "Scored crop (px per side, native grid)", "type": "int", "default": 1024},
               {"name": "sigma_eval", "label": "Truth comparison resolution σ (native px)", "type": "float",
                "default": 1.0}]
    default_objective = "bg_noise"

    def prepare(self, ds, opts, device, log, cancel, study_dir):
        base = ds.session()
        if base.analysis is None:
            if ds.synthetic:
                ds.ensure_stacked(log, cancel)
                base = ds.session()
            else:
                raise RuntimeError(f"{ds.folder} has not been analysed: run 'Analyse frames' first")
        return {"ds": ds, "base": base, "device": device, "opts": opts, "study_dir": study_dir, "n": 0}

    def run(self, ctx, p, log, cancel):
        from ..pipeline import STACK_DEFAULTS, Session, _load_fits
        base, ds = ctx["base"], ctx["ds"]
        ctx["n"] += 1
        wd = os.path.join(ctx["study_dir"], "work", f"trial_{ctx['n']}")
        s = Session(ds.folder, wd)
        for f in ("analysis.pkl", "defects.npy"):
            if os.path.exists(base._p(f)):
                shutil.copy(base._p(f), s._p(f))
        s._load_state()
        s.cancel_flag = cancel
        t = time.time()
        try:
            meta = s.run_stack({**STACK_DEFAULTS, **{k: p[k] for k in ("sigma_low", "sigma_high", "local_norm",
                                                                         "sensitivity", "mode", "scale")}},
                               _prog(log, 20))
            dt = time.time() - t
            st, cov = _load_fits(s._p("stack.fits")), _load_fits(s._p("coverage.fits"))
            sc = float(meta["scale"])
            c = int(ctx["opts"].get("crop", 1024) * sc)
            sl = _deep_crop(cov, c)
            x = st[sl]
            x1 = cv2.resize(x, (int(x.shape[1] / sc), int(x.shape[0] / sc)), interpolation=cv2.INTER_AREA) if sc != 1 else x
            out = {"bg_noise": MX.background_noise(x1), "fwhm": MX.star_fwhm(x, meta["saturation"]) / sc,
                   "seconds": dt, "n_frames": meta["n_frames"]}
            if ds.synthetic:
                fr = [i for i, f in enumerate(s.analysis["frames"]) if f["accepted"]]
                psf = SY.effective_psf(ds.dir, fr, [s.analysis["frames"][i]["weight"] for i in fr], sc)
                H, W = st.shape[:2]
                ref = s.analysis["ref_idx"]
                tr = SY.truth_image(ds.dir, ref, (H, W), sc, psf, device=ctx["device"])[sl]
                ts = SY.truth_image(ds.dir, ref, (H, W), sc, psf, device=ctx["device"], extended=False)[sl]
                pos = SY.star_positions(ds.dir, ref, sc)
                pos = {"x": pos["x"] - sl[1].start, "y": pos["y"] - sl[0].start, "flux": pos["flux"]}
                good = cov[sl] >= 0.6 * np.percentile(cov[sl], 90)
                unsat = cv2.erode((x.max(-1) < 0.5 * meta["saturation"]).astype(np.uint8),
                                  np.ones((int(9 * sc) | 1,) * 2, np.uint8)) > 0
                m = MX.truth_metrics(x, tr, good & unsat, star_truth=ts, stars=pos,
                                     sigma_eval=float(ctx["opts"].get("sigma_eval", 1.0)) * sc)
                out.update(_truth_row(m))
            return out, x1
        finally:
            shutil.rmtree(wd, ignore_errors=True)


TASKS = {t.name: t for t in (ImageMMTask(), DenoiseTask(), NetworkTask(), StackTask())}
