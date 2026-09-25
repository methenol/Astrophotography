"""Orchestration with on-disk caching (used by both the CLI and the web UI).

Heavy stages (analysis, integration, ML denoise) are cached per dataset in
``<workdir>/<dataset-slug>/``; processing parameters can then be iterated on
interactively without touching the raw frames again.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from astropy.io import fits
from PIL import Image

from . import __version__
from .analysis import analyse, finalize_selection
from .frames import FrameInfo, build_defect_map, discover, read_raw, superpixel
from .postprocess import DEFAULTS, is_narrowband, linear_stage, luminance, nonlinear_stage
from .stacking import Integrator

STACK_DEFAULTS = {
    "mode": "auto",          # auto | drizzle | demosaic
    "scale": 1.0,            # output up-sampling (1, 1.5, 2)
    "sigma_low": 4.0,
    "sigma_high": 3.0,
    "local_norm": True,
    "sensitivity": 1.0,      # frame-rejection aggressiveness
    "denoise_iters": 2000,
    "ai_deconvolution": True,  # train the self-supervised deconvolution network after the denoiser
    "device": "auto",        # auto | cuda | cuda:N | mps | cpu
}

LINEAR_KEYS = ["crop", "crop_threshold", "background", "bg_method", "bg_degree",
               "white_balance", "denoise", "deconvolution"]


def slugify(path: str) -> str:
    base = os.path.basename(os.path.normpath(path)) or "dataset"
    h = hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:6]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("_") + "-" + h


def _save_fits(path, arr, header=None):
    data = np.moveaxis(arr, -1, 0) if arr.ndim == 3 else arr
    fits.PrimaryHDU(data.astype(np.float32), header=header).writeto(path, overwrite=True)


def _load_fits(path):
    d = fits.getdata(path).astype(np.float32)
    return np.moveaxis(d, 0, -1) if d.ndim == 3 else d


def clean_json(o):
    """Recursively replace NaN/inf with None and numpy scalars with Python types (strict JSON)."""
    if isinstance(o, dict):
        return {k: clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean_json(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class _CompatUnpickler(pickle.Unpickler):
    """Load caches written before the package was renamed astropipe -> astrophoto."""

    def find_class(self, module, name):
        if module == "astropipe" or module.startswith("astropipe."):
            module = "astrophoto" + module[len("astropipe"):]
        return super().find_class(module, name)


class Cancelled(Exception):
    pass


class Session:
    """All state for one dataset (folder of subs)."""

    def __init__(self, folder: str, workdir: str = "output"):
        self.folder = os.path.abspath(folder)
        self.dir = os.path.join(os.path.abspath(workdir), slugify(folder))
        os.makedirs(self.dir, exist_ok=True)
        self.infos: list[FrameInfo] = []
        self.analysis: dict | None = None
        self.defects: np.ndarray | None = None
        self.overrides: dict[str, bool] = {}
        self._stack_cache: dict | None = None
        self._den_cache: np.ndarray | None = None
        self._sharp_cache: np.ndarray | None = None
        self._lin_cache: tuple[str, np.ndarray, dict] | None = None
        self.lock = threading.RLock()
        self.cancel_flag = threading.Event()
        self._load_state()

    # ------------------------------------------------------------- persistence
    def _p(self, name):
        return os.path.join(self.dir, name)

    def _load_state(self):
        if os.path.exists(self._p("analysis.pkl")):
            try:
                with open(self._p("analysis.pkl"), "rb") as f:
                    st = _CompatUnpickler(f).load()
                self.infos, self.analysis, self.overrides = st["infos"], st["analysis"], st.get("overrides", {})
                if os.path.exists(self._p("defects.npy")):
                    self.defects = np.load(self._p("defects.npy"))
            except Exception:
                self.analysis = None

    def _save_analysis(self):
        with open(self._p("analysis.pkl"), "wb") as f:
            pickle.dump({"infos": self.infos, "analysis": self.analysis, "overrides": self.overrides}, f)
        with open(self._p("frames.json"), "w") as f:
            json.dump(self.frames_table(), f, indent=1, default=_json_default)

    @property
    def meta(self) -> dict:
        p = self._p("stack_meta.json")
        return json.load(open(p)) if os.path.exists(p) else {}

    def status(self) -> dict:
        return {
            "folder": self.folder,
            "workdir": self.dir,
            "n_files": len(self.infos) if self.infos else None,
            "analysed": self.analysis is not None,
            "stacked": os.path.exists(self._p("stack.fits")),
            "denoised": os.path.exists(self._p("denoised.fits")),
            "deconvolved": os.path.exists(self._p("sharp.fits")),
            "stack_meta": clean_json(self.meta),
            "filter": self.infos[0].filter if self.infos else None,
            "object": self.infos[0].object if self.infos else None,
            "narrowband": is_narrowband(self.infos[0].filter) if self.infos else None,
        }

    def _check_cancel(self):
        if self.cancel_flag.is_set():
            raise Cancelled()

    # ------------------------------------------------------------- stage 1
    def scan(self):
        self.infos = discover(self.folder)
        if not self.infos:
            raise RuntimeError(f"No light frames (FITS) found in {self.folder}")
        return self.infos

    def run_analysis(self, sensitivity: float = 1.0, progress=None):
        with self.lock:
            self.scan()
            if progress:
                progress(0, 1, f"Building hot-pixel map from {len(self.infos)} frames")
            self.defects = build_defect_map(self.infos)
            np.save(self._p("defects.npy"), self.defects)
            self._check_cancel()
            self.analysis = analyse(self.infos, self.defects, progress=progress, sensitivity=sensitivity)
            self.analysis["sensitivity"] = sensitivity
            self._apply_overrides()
            self._save_analysis()
            self._make_reference_preview()
            return self.frames_table()

    def reselect(self, sensitivity: float):
        """Re-run the rejection logic with a different aggressiveness (no re-measurement)."""
        with self.lock:
            res = finalize_selection(self.analysis["frames"], self.analysis["ref_idx"],
                                     self.analysis["grid"], sensitivity)
            res["sensitivity"] = sensitivity
            self.analysis = res
            self._apply_overrides()
            self._save_analysis()
            return self.frames_table()

    def set_override(self, name: str, accepted: bool | None):
        with self.lock:
            if accepted is None:
                self.overrides.pop(name, None)
            else:
                self.overrides[name] = bool(accepted)
            self.reselect(self.analysis.get("sensitivity", 1.0))

    def _apply_overrides(self):
        fr = self.analysis["frames"]
        wmax = max((f["weight"] for f in fr), default=1) or 1
        for f in fr:
            f["auto_accepted"] = not f["reject_reasons"]
            if f["name"] in self.overrides:
                f["accepted"] = self.overrides[f["name"]] and f["transform"] is not None
                f["overridden"] = True
                if f["accepted"] and f["weight"] == 0:
                    f["weight"] = 0.5 * wmax
                if not f["accepted"]:
                    f["weight"] = 0.0
            else:
                f["overridden"] = False

    def frames_table(self) -> list[dict]:
        if not self.analysis:
            return []
        out = []
        for info, f in zip(self.infos, self.analysis["frames"]):
            out.append({
                "name": f["name"], "time": info.date_obs, "exptime": info.exptime,
                "n_stars": f["n_stars"], "fwhm": f["fwhm"], "elongation": f["elongation"],
                "background": float(np.mean(f["background"])), "noise": f["noise"],
                "transparency": f["transparency"], "obstructed": f["obstructed_frac"],
                "reg_rms": f["reg_rms"], "anomaly": f.get("anomaly", 0.0), "weight": f["weight"],
                "accepted": f["accepted"], "auto_accepted": f.get("auto_accepted", f["accepted"]),
                "overridden": f.get("overridden", False), "reasons": f["reject_reasons"],
                "tile_mask": f["tile_mask"].tolist() if f.get("tile_mask") is not None else None,
                "is_reference": f["name"] == self.analysis["frames"][self.analysis["ref_idx"]]["name"],
            })
        return clean_json(out)

    def _make_reference_preview(self):
        ref = self.infos[self.analysis["ref_idx"]]
        self.frame_thumbnail(ref.name, 900, self._p("reference.jpg"))

    def frame_thumbnail(self, name: str, size: int = 360, out_path: str | None = None) -> bytes:
        info = next(i for i in self.infos if i.name == name)
        cache = out_path or self._p(f"thumbs/{name}_{size}.jpg")
        if os.path.exists(cache):
            return open(cache, "rb").read()
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        raw = read_raw(info.path, info.bias)
        sp = superpixel(raw, info.bayer)
        img = autostretch(sp)
        h, w = img.shape[:2]
        s = size / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", (img[..., ::-1] * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 85])
        open(cache, "wb").write(buf.tobytes())
        return buf.tobytes()

    # ------------------------------------------------------------- stage 2
    def run_stack(self, params: dict | None = None, progress=None):
        p = {**STACK_DEFAULTS, **(params or {})}
        with self.lock:
            if self.analysis is None:
                self.run_analysis(p["sensitivity"], progress)
            elif abs(self.analysis.get("sensitivity", 1.0) - p["sensitivity"]) > 1e-6:
                self.reselect(p["sensitivity"])
            integ = Integrator(self.infos, self.analysis, self.defects, mode=p["mode"], scale=float(p["scale"]),
                               sigma_low=float(p["sigma_low"]), sigma_high=float(p["sigma_high"]),
                               local_norm=bool(p["local_norm"]), progress=progress,
                               cancel=self.cancel_flag.is_set)
            out = integ.run()
            hdr = fits.Header()
            info0 = self.infos[0]
            for k, v in {"OBJECT": info0.object, "FILTER": info0.filter, "NFRAMES": out["n_frames"],
                         "TOTEXP": out["total_exposure"], "STKMODE": out["mode"], "STKSCALE": out["scale"],
                         "CREATOR": f"AstroPhoto Studio {__version__}", "BAYERPAT": info0.bayer}.items():
                hdr[k] = v
            _save_fits(self._p("stack.fits"), out["stack"], hdr)
            _save_fits(self._p("half_a.fits"), out["half_a"])
            _save_fits(self._p("half_b.fits"), out["half_b"])
            _save_fits(self._p("coverage.fits"), out["coverage"])
            _save_fits(self._p("rejection_map.fits"), out["rejected_frac"])
            meta = {"mode": out["mode"], "scale": out["scale"], "n_frames": out["n_frames"],
                    "total_exposure": out["total_exposure"], "params": p,
                    "saturation": 65535.0 - info0.bias, "filter": info0.filter, "object": info0.object,
                    "created": datetime.now().isoformat(timespec="seconds"),
                    "shape": list(out["stack"].shape)}
            json.dump(meta, open(self._p("stack_meta.json"), "w"), indent=1, default=_json_default)
            for f in ("denoised.fits", "sharp.fits", "restore_meta.json", "restore_nets.pt"):
                if os.path.exists(self._p(f)):
                    os.remove(self._p(f))
            self._stack_cache = {"stack": out["stack"], "coverage": out["coverage"]}
            self._den_cache = None
            self._sharp_cache = None
            self._lin_cache = None
            return meta

    def run_denoise(self, params: dict | None = None, progress=None):
        """AI restoration: Noise2Noise denoiser, then (optionally) the N2N deconvolution network."""
        from .denoise import n2n_restore
        p = {**STACK_DEFAULTS, **(params or {})}
        with self.lock:
            st = self._load_stack()
            a, b = _load_fits(self._p("half_a.fits")), _load_fits(self._p("half_b.fits"))
            den, sharp, info = n2n_restore(
                a, b, st["stack"], iters=int(p["denoise_iters"]), device=p["device"], coverage=st["coverage"],
                progress=progress, cancel=self.cancel_flag.is_set, deconvolve=bool(p["ai_deconvolution"]),
                sat=self.meta.get("saturation", 63471.0), px_scale=float(self.meta.get("scale", 1.0)),
                save_path=self._p("restore_nets.pt"))
            del a, b
            _save_fits(self._p("denoised.fits"), den)
            if sharp is not None:
                _save_fits(self._p("sharp.fits"), sharp)
            elif os.path.exists(self._p("sharp.fits")):
                os.remove(self._p("sharp.fits"))
            info["created"] = datetime.now().isoformat(timespec="seconds")
            json.dump(info, open(self._p("restore_meta.json"), "w"), indent=1, default=_json_default)
            self._den_cache = den
            self._sharp_cache = sharp
            self._lin_cache = None
            return True

    def _load_stack(self):
        if self._stack_cache is None:
            if not os.path.exists(self._p("stack.fits")):
                raise RuntimeError("Dataset has not been stacked yet")
            self._stack_cache = {"stack": _load_fits(self._p("stack.fits")),
                                 "coverage": _load_fits(self._p("coverage.fits"))}
        return self._stack_cache

    def _load_denoised(self):
        if self._den_cache is None and os.path.exists(self._p("denoised.fits")):
            self._den_cache = _load_fits(self._p("denoised.fits"))
        return self._den_cache

    def _load_sharp(self):
        if self._sharp_cache is None and os.path.exists(self._p("sharp.fits")):
            self._sharp_cache = _load_fits(self._p("sharp.fits"))
        return self._sharp_cache

    # ------------------------------------------------------------- stage 3
    def linear(self, params: dict, progress=None):
        p = {**DEFAULTS, **(params or {})}
        key = json.dumps({k: p[k] for k in LINEAR_KEYS}, sort_keys=True) + self.meta.get("created", "")
        with self.lock:
            if self._lin_cache and self._lin_cache[0] == key:
                return self._lin_cache[1], self._lin_cache[2]
            st = self._load_stack()
            den = self._load_denoised()
            lin, info = linear_stage(st["stack"], st["coverage"], den, p, self.meta.get("saturation", 63471.0),
                                     progress=progress, sharp=self._load_sharp() if den is not None else None)
            self._lin_cache = (key, lin, info)
            return lin, info

    def render(self, params: dict, max_size: int | None = 1400, progress=None) -> tuple[np.ndarray, dict]:
        lin, info = self.linear(params, progress)
        f = 1.0
        if max_size and max(lin.shape[:2]) > max_size:
            f = max_size / max(lin.shape[:2])
            lin = cv2.resize(lin, (int(lin.shape[1] * f), int(lin.shape[0] * f)), interpolation=cv2.INTER_AREA)
        params = {**params, "_noise_ref": info.get("noise_ref")}
        out = nonlinear_stage(lin, params, self.meta.get("filter", ""), px_scale=f, progress=progress)
        return out, info

    def render_before(self, params: dict, max_size: int = 1400) -> np.ndarray:
        """Plain auto-stretched stack (same crop) for before/after comparison."""
        st = self._load_stack()
        _, info = self.linear(params)
        img = st["stack"]
        if "crop" in info:
            y0, y1, x0, x1 = info["crop"]
            img = img[y0:y1, x0:x1]
        f = min(1.0, max_size / max(img.shape[:2]))
        img = cv2.resize(img, (int(img.shape[1] * f), int(img.shape[0] * f)), interpolation=cv2.INTER_AREA)
        return autostretch(img)

    def export(self, params: dict, quality: int = 95, upscale: float = 1.0, tiff: bool = True,
               progress=None) -> dict:
        img, info = self.render(params, max_size=None, progress=progress)
        if upscale and upscale > 1:
            img = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LANCZOS4)
            img = np.clip(img, 0, 1)
        os.makedirs(self._p("exports"), exist_ok=True)
        obj = (self.meta.get("object") or "image").replace(" ", "")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = self._p(f"exports/{obj}_{stamp}")
        im8 = Image.fromarray((img * 255 + 0.5).astype(np.uint8))
        desc = (f"{self.meta.get('object', '')} | {self.meta.get('n_frames')} x subs, "
                f"{self.meta.get('total_exposure', 0) / 60:.1f} min | AstroPhoto Studio {__version__}")
        exif = Image.Exif()
        exif[0x010E] = desc  # ImageDescription
        exif[0x0131] = f"AstroPhoto Studio {__version__}"  # Software
        im8.save(base + ".jpg", quality=int(quality), subsampling=0, optimize=True, exif=exif)
        files = {"jpg": base + ".jpg"}
        if tiff:
            import tifffile
            tifffile.imwrite(base + ".tif", (img * 65535 + 0.5).astype(np.uint16), photometric="rgb",
                             compression="zlib", description=desc)
            files["tif"] = base + ".tif"
        json.dump({"params": {**DEFAULTS, **params}, "linear_info": info, "meta": self.meta},
                  open(base + ".json", "w"), indent=1, default=_json_default)
        files["json"] = base + ".json"
        files["size"] = [int(img.shape[1]), int(img.shape[0])]
        return files

    def export_linear_fits(self, params: dict) -> str:
        lin, _ = self.linear(params)
        path = self._p("exports/linear_processed.fits")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _save_fits(path, lin)
        return path

    # ------------------------------------------------------------- one-shot
    def run_all(self, stack_params=None, proc_params=None, progress=None, **export_kw):
        t0 = time.time()
        sp = {**STACK_DEFAULTS, **(stack_params or {})}
        self.run_analysis(sp["sensitivity"], progress)
        self.run_stack(sp, progress)
        if float((proc_params or {}).get("denoise", DEFAULTS["denoise"])) > 0:
            self.run_denoise(sp, progress)
        files = self.export(proc_params or {}, progress=progress, **export_kw)
        files["seconds"] = round(time.time() - t0, 1)
        return files


def autostretch(img: np.ndarray, target: float = 0.2) -> np.ndarray:
    """Linked screen-transfer-function autostretch (PixInsight STF style)."""
    x = img.astype(np.float32)
    L = luminance(x) if x.ndim == 3 else x
    med = np.median(L[::4, ::4])
    mad = 1.4826 * np.median(np.abs(L[::4, ::4] - med))
    lo = med - 2.8 * mad
    hi = np.percentile(L[::4, ::4], 99.95)
    x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
    m0 = (med - lo) / max(hi - lo, 1e-6)
    # midtone transfer so that the median lands on target
    m = m0 * (target - 1) / (2 * m0 * target - m0 - target)
    x = ((m - 1) * x) / ((2 * m - 1) * x - m)
    if x.ndim == 3:
        # neutralise the background for display
        bgc = np.median(x[::4, ::4].reshape(-1, 3), axis=0)
        x = np.clip(x - (bgc - bgc.mean()), 0, 1)
    return np.clip(x, 0, 1).astype(np.float32)
