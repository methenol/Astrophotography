"""AstroPipe web UI (FastAPI).

    python -m webui.server            # http://127.0.0.1:8000
    python -m webui.server --host 0.0.0.0 --port 8080 --images /path/to/seestar/exports
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import threading
import time
import traceback
import uuid
import warnings

import cv2
import numpy as np
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from astropipe import __version__  # noqa: E402
from astropipe.pipeline import DEFAULTS, STACK_DEFAULTS, Cancelled, Session, clean_json, slugify  # noqa: E402

CONFIG = {"images": os.path.join(ROOT, "images"), "workdir": os.path.join(ROOT, "output")}

app = FastAPI(title="AstroPipe", version=__version__)
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

SESSIONS: dict[str, Session] = {}
JOBS: dict[str, dict] = {}
JOB_LOCK = threading.Lock()   # one heavy job at a time (memory)
PREVIEW_CACHE: dict[str, bytes] = {}

# UI metadata for every processing parameter
PARAM_SPEC = [
    {"group": "Linear", "key": "crop", "label": "Auto-crop stacking edges", "type": "bool"},
    {"group": "Linear", "key": "crop_threshold", "label": "Crop: min. frame coverage", "type": "range", "min": 0.1, "max": 1.0, "step": 0.05},
    {"group": "Linear", "key": "background", "label": "Gradient removal", "type": "bool"},
    {"group": "Linear", "key": "bg_method", "label": "Gradient model", "type": "select", "options": ["auto", "poly", "rbf"]},
    {"group": "Linear", "key": "bg_degree", "label": "Polynomial degree", "type": "range", "min": 1, "max": 4, "step": 1},
    {"group": "Linear", "key": "white_balance", "label": "White balance", "type": "select", "options": ["stars", "background", "none"]},
    {"group": "Linear", "key": "denoise", "label": "AI denoise (Noise2Noise)", "type": "range", "min": 0, "max": 1, "step": 0.05},
    {"group": "Linear", "key": "deconvolution", "label": "Deconvolution (sharpen stars/detail)", "type": "range", "min": 0, "max": 1, "step": 0.05},
    {"group": "Stretch", "key": "stretch", "label": "Stretch (background level)", "type": "range", "min": 0.04, "max": 0.35, "step": 0.01},
    {"group": "Stretch", "key": "auto_stretch", "label": "Adapt stretch to target size", "type": "bool"},
    {"group": "Stretch", "key": "hdr", "label": "HDR (protect bright cores)", "type": "range", "min": 0, "max": 1.5, "step": 0.05},
    {"group": "Stretch", "key": "contrast", "label": "GHS focus (contrast)", "type": "range", "min": -1, "max": 10, "step": 0.25},
    {"group": "Stretch", "key": "color_preservation", "label": "Colour-preserving stretch", "type": "range", "min": 0, "max": 1, "step": 0.05},
    {"group": "Colour", "key": "palette", "label": "Palette", "type": "select", "options": ["auto", "natural", "hoo", "foraxx", "hoo_warm"]},
    {"group": "Colour", "key": "oiii_boost", "label": "OIII boost (dual-band)", "type": "range", "min": 0.5, "max": 3, "step": 0.05},
    {"group": "Colour", "key": "synthetic_luminance", "label": "Synthetic luminance (dual-band)", "type": "range", "min": 0, "max": 1, "step": 0.05},
    {"group": "Colour", "key": "oiii_unmix", "label": "Remove Ha leakage from OIII", "type": "bool"},
    {"group": "Colour", "key": "saturation", "label": "Saturation", "type": "range", "min": 0.5, "max": 3, "step": 0.05},
    {"group": "Colour", "key": "chroma_denoise", "label": "Colour noise reduction", "type": "range", "min": 0, "max": 1, "step": 0.05},
    {"group": "Colour", "key": "scnr", "label": "SCNR (remove green cast)", "type": "range", "min": 0, "max": 1, "step": 0.05},
    {"group": "Stars", "key": "star_separation", "label": "Process stars separately", "type": "bool"},
    {"group": "Stars", "key": "star_reduction", "label": "Star reduction", "type": "range", "min": 0, "max": 1, "step": 0.05},
    {"group": "Stars", "key": "star_intensity", "label": "Star brightness", "type": "range", "min": 0, "max": 1.5, "step": 0.05},
    {"group": "Stars", "key": "star_saturation", "label": "Star colour", "type": "range", "min": 0, "max": 2.5, "step": 0.05},
    {"group": "Stars", "key": "star_color_preservation", "label": "Star colour intensity (stretch)", "type": "range", "min": 0, "max": 1, "step": 0.05},
    {"group": "Stars", "key": "halo_suppress", "label": "Halo suppression", "type": "range", "min": 0, "max": 1, "step": 0.05},
    {"group": "Detail", "key": "luminance_denoise", "label": "Fine-grain noise reduction", "type": "range", "min": 0, "max": 1.5, "step": 0.05},
    {"group": "Detail", "key": "local_contrast", "label": "Local contrast (structures)", "type": "range", "min": 0, "max": 2, "step": 0.05},
    {"group": "Detail", "key": "sharpen", "label": "Final sharpening", "type": "range", "min": 0, "max": 1.5, "step": 0.05},
    {"group": "Finish", "key": "black_point", "label": "Black point", "type": "range", "min": 0, "max": 0.2, "step": 0.005},
    {"group": "Finish", "key": "brightness", "label": "Midtones", "type": "range", "min": -1, "max": 1, "step": 0.05},
]

PRESETS = {
    "Balanced": {},
    "Vivid nebula": {"saturation": 2.0, "local_contrast": 0.8, "stretch": 0.18, "contrast": 3.0, "oiii_boost": 1.3,
                     "star_reduction": 0.5, "star_intensity": 0.8},
    "Starless-ish": {"star_reduction": 0.8, "star_intensity": 0.45, "local_contrast": 0.7},
    "Galaxy / broadband": {"palette": "natural", "stretch": 0.12, "contrast": 4.0, "saturation": 1.4,
                           "local_contrast": 0.4, "star_reduction": 0.2, "bg_degree": 2},
    "Natural colour": {"palette": "natural", "saturation": 1.3, "scnr": 0.6},
    "Deep & dark": {"stretch": 0.1, "black_point": 0.04, "contrast": 4.0, "local_contrast": 0.6},
}


def get_session(folder: str) -> Session:
    folder = os.path.abspath(os.path.expanduser(folder))
    if not os.path.isdir(folder):
        raise HTTPException(404, f"Folder not found: {folder}")
    key = slugify(folder)
    if key not in SESSIONS:
        SESSIONS[key] = Session(folder, CONFIG["workdir"])
    return SESSIONS[key]


# ------------------------------------------------------------------ pages

@app.get("/", response_class=HTMLResponse)
def index():
    return open(os.path.join(STATIC, "index.html"), encoding="utf-8").read()


@app.get("/api/system")
def system():
    try:
        from astropipe.denoise import device_info
        dev = device_info()
    except Exception as e:  # torch missing
        dev = {"default": "cpu", "gpus": [], "error": str(e)}
    return {"version": __version__, "devices": dev, "defaults": DEFAULTS, "stack_defaults": STACK_DEFAULTS,
            "param_spec": PARAM_SPEC, "presets": PRESETS, "images_root": CONFIG["images"]}


@app.get("/api/datasets")
def datasets(root: str | None = None):
    root = os.path.abspath(os.path.expanduser(root or CONFIG["images"]))
    out = []
    if not os.path.isdir(root):
        return {"root": root, "datasets": []}
    candidates = [root] + sorted(d for d in glob.glob(os.path.join(root, "**"), recursive=True) if os.path.isdir(d))
    for d in candidates:
        fits_files = [f for ext in ("*.fit", "*.fits", "*.fts") for f in glob.glob(os.path.join(d, ext))]
        if not fits_files:
            continue
        info = {"path": d, "name": os.path.relpath(d, root) if d != root else os.path.basename(d),
                "n_fits": len(fits_files)}
        try:
            from astropy.io import fits
            h = fits.getheader(sorted(fits_files)[0])
            info.update({"object": str(h.get("OBJECT", "")).strip(), "filter": str(h.get("FILTER", "")).strip(),
                         "exptime": float(h.get("EXPTIME", 0) or 0), "instrument": str(h.get("CREATOR", "")).strip()})
            info["total_min"] = round(info["exptime"] * len(fits_files) / 60, 1)
        except Exception:
            pass
        cache_dir = os.path.join(CONFIG["workdir"], slugify(d))
        info["cached"] = {"analysed": os.path.exists(os.path.join(cache_dir, "analysis.pkl")),
                          "stacked": os.path.exists(os.path.join(cache_dir, "stack.fits")),
                          "denoised": os.path.exists(os.path.join(cache_dir, "denoised.fits"))}
        out.append(info)
    return {"root": root, "datasets": out}


@app.post("/api/open")
def open_dataset(body: dict = Body(...)):
    s = get_session(body["folder"])
    if not s.infos:
        s.scan()
    return clean_json({"status": s.status(), "frames": s.frames_table(), "medians": (s.analysis or {}).get("medians")})


@app.get("/api/frames")
def frames(folder: str):
    s = get_session(folder)
    return clean_json({"frames": s.frames_table(), "medians": (s.analysis or {}).get("medians"),
                       "sensitivity": (s.analysis or {}).get("sensitivity", 1.0)})


@app.post("/api/frames/override")
def override(body: dict = Body(...)):
    s = get_session(body["folder"])
    s.set_override(body["name"], body.get("accepted"))
    return {"frames": s.frames_table()}


@app.post("/api/frames/sensitivity")
def sensitivity(body: dict = Body(...)):
    s = get_session(body["folder"])
    return {"frames": s.reselect(float(body["sensitivity"]))}


@app.get("/api/thumb")
def thumb(folder: str, name: str, size: int = 360):
    s = get_session(folder)
    if not s.infos:
        s.scan()
    return Response(s.frame_thumbnail(name, size), media_type="image/jpeg")


# ------------------------------------------------------------------ jobs

def _run_job(job_id: str, kind: str, folder: str, params: dict, stack_params: dict, export_opts: dict):
    job = JOBS[job_id]
    s = get_session(folder)
    s.cancel_flag.clear()

    def progress(i, n, msg):
        job["progress"] = i / max(n, 1)
        job["message"] = msg
        if not job["log"] or job["log"][-1][1] != msg.split(" (")[0].rsplit(" ", 1)[0]:
            job["log"].append([round(time.time() - job["started"], 1), msg.split(" (")[0].rsplit(" ", 1)[0]])
            job["log"] = job["log"][-200:]

    with JOB_LOCK:
        job["state"] = "running"
        job["started"] = time.time()
        try:
            if kind == "analyse":
                s.run_analysis(float(stack_params.get("sensitivity", 1.0)), progress)
                job["result"] = {"n": len(s.infos)}
            elif kind == "stack":
                job["result"] = s.run_stack(stack_params, progress)
            elif kind == "denoise":
                s.run_denoise(stack_params, progress)
                job["result"] = {"ok": True}
            elif kind == "all":
                sp = {**STACK_DEFAULTS, **stack_params}
                s.run_analysis(sp["sensitivity"], progress)
                s.run_stack(sp, progress)
                s.run_denoise(sp, progress)
                job["result"] = s.export(params, progress=progress, **export_opts)
            elif kind == "export":
                job["result"] = s.export(params, progress=progress, **export_opts)
            job["state"] = "done"
            job["progress"] = 1.0
            job["message"] = "Finished"
        except Cancelled:
            job["state"] = "cancelled"
            job["message"] = "Cancelled"
        except Exception as e:
            if "cancelled" in str(e):
                job["state"] = "cancelled"
                job["message"] = "Cancelled"
            else:
                job["state"] = "error"
                job["message"] = f"{type(e).__name__}: {e}"
                job["traceback"] = traceback.format_exc()
        finally:
            job["ended"] = time.time()
            PREVIEW_CACHE.clear()


@app.post("/api/jobs")
def start_job(body: dict = Body(...)):
    kind = body["kind"]
    if kind not in ("analyse", "stack", "denoise", "all", "export"):
        raise HTTPException(400, "unknown job kind")
    if any(j["state"] in ("queued", "running") for j in JOBS.values()):
        raise HTTPException(409, "Another job is already running")
    job_id = uuid.uuid4().hex[:10]
    JOBS[job_id] = {"id": job_id, "kind": kind, "folder": body["folder"], "state": "queued", "progress": 0.0,
                    "message": "Queued", "log": [], "started": time.time(), "result": None}
    t = threading.Thread(target=_run_job, args=(job_id, kind, body["folder"], body.get("params") or {},
                                                body.get("stack_params") or {}, body.get("export") or {}),
                         daemon=True)
    t.start()
    return JOBS[job_id]


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404)
    j = dict(JOBS[job_id])
    j["elapsed"] = round((j.get("ended") or time.time()) - j["started"], 1)
    return JSONResponse(clean_json(j))


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404)
    get_session(j["folder"]).cancel_flag.set()
    return {"ok": True}


# ------------------------------------------------------------------ previews

def _jpeg(img: np.ndarray, q: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", (np.clip(img, 0, 1)[..., ::-1] * 255 + 0.5).astype(np.uint8),
                           [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes()


@app.post("/api/preview")
def preview(body: dict = Body(...)):
    s = get_session(body["folder"])
    params = body.get("params") or {}
    which = body.get("which", "after")
    size = body.get("size", 1400)
    size = None if size in (0, None, "full") else int(size)
    import json as _json
    key = _json.dumps([s.dir, params, which, size, s.meta.get("created")], sort_keys=True)
    if key in PREVIEW_CACHE:
        return Response(PREVIEW_CACHE[key], media_type="image/jpeg")
    if JOB_LOCK.locked():
        raise HTTPException(409, "A pipeline job is running – preview available when it finishes")
    try:
        t0 = time.time()
        if which == "before":
            img = s.render_before(params, size or 100000)
            info = {}
        else:
            img, info = s.render(params, max_size=size)
        data = _jpeg(img, 92)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    if len(PREVIEW_CACHE) > 24:
        PREVIEW_CACHE.clear()
    PREVIEW_CACHE[key] = data
    return Response(data, media_type="image/jpeg",
                    headers={"X-Render-Time": f"{time.time() - t0:.2f}", "X-Width": str(img.shape[1]),
                             "X-Height": str(img.shape[0])})


@app.get("/api/linear_info")
def linear_info(folder: str):
    s = get_session(folder)
    if s._lin_cache:
        return clean_json(s._lin_cache[2])
    return {}


@app.get("/api/diagnostic")
def diagnostic(folder: str, kind: str):
    s = get_session(folder)
    p = {"coverage": "coverage.fits", "rejection": "rejection_map.fits"}.get(kind)
    if kind == "reference":
        f = os.path.join(s.dir, "reference.jpg")
        if not os.path.exists(f):
            raise HTTPException(404)
        return FileResponse(f, media_type="image/jpeg")
    if not p or not os.path.exists(os.path.join(s.dir, p)):
        raise HTTPException(404)
    from astropipe.pipeline import _load_fits
    m = _load_fits(os.path.join(s.dir, p))
    m = m / max(np.percentile(m, 99.9), 1e-6)
    m = cv2.resize(m, None, fx=min(1, 900 / max(m.shape)), fy=min(1, 900 / max(m.shape)), interpolation=cv2.INTER_AREA)
    col = cv2.applyColorMap((np.clip(m, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    ok, buf = cv2.imencode(".jpg", col)
    return Response(buf.tobytes(), media_type="image/jpeg")


@app.get("/api/exports")
def exports(folder: str):
    s = get_session(folder)
    d = os.path.join(s.dir, "exports")
    items = []
    for f in sorted(glob.glob(os.path.join(d, "*.jpg")), reverse=True):
        base = f[:-4]
        items.append({"jpg": f, "tif": base + ".tif" if os.path.exists(base + ".tif") else None,
                      "name": os.path.basename(f), "size_mb": round(os.path.getsize(f) / 2**20, 1),
                      "created": time.ctime(os.path.getmtime(f))})
    return {"exports": items}


@app.get("/api/download")
def download(path: str, inline: bool = False):
    path = os.path.abspath(path)
    if not path.startswith(os.path.abspath(CONFIG["workdir"])) or not os.path.isfile(path):
        raise HTTPException(403)
    return FileResponse(path, filename=None if inline else os.path.basename(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--images", default=CONFIG["images"], help="root folder that contains Seestar *_sub folders")
    ap.add_argument("--workdir", default=CONFIG["workdir"])
    a = ap.parse_args()
    CONFIG["images"] = os.path.abspath(a.images)
    CONFIG["workdir"] = os.path.abspath(a.workdir)
    import uvicorn
    print(f"AstroPipe {__version__} web UI -> http://{a.host}:{a.port}")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
