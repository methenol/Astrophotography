"""Run one Optuna study (Akiba et al. 2019, KDD) of an experiment-lab task.

    python -m astrophoto.lab.run <study_dir>

``<study_dir>/config.json``::

    {"name": ..., "task": "imagemm" | "denoise" | "network" | "stack",
     "dataset": {"kind": "real", "folder": ...} | {"kind": "synthetic", "dir": ...},
     "options": {...task options...},
     "space": {param: {"tune": bool, "low", "high", "log", "step", "choices", "value"}},
     "objectives": [{"metric": ..., "direction": "minimize" | "maximize"}, ...],
     "sampler": "tpe" | "random" | "nsga2" | "grid", "seed": 0,
     "n_trials": 30, "timeout_min": 0, "device": "auto", "baseline": true}

Trials are stored in ``<study_dir>/study.db`` (SQLite, Optuna RDB storage); every metric is
a user attribute of its trial; the progress is in ``status.json`` and ``log.txt``, and each
trial's result is previewed in ``trials/<number>.jpg`` (one stretch for the whole study).
The first trial is the pipeline default (``baseline``), so every other one is compared
with what the pipeline does today.  SIGTERM stops the study after cancelling the running
trial.
"""
from __future__ import annotations

import json
import math
import os
import signal
import sys
import threading
import time
import traceback
from datetime import datetime

import numpy as np


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


def suggest(trial, name: str, spec: dict, pdef: dict):
    """Sample one parameter, or return its fixed value."""
    if not spec.get("tune"):
        return spec.get("value", pdef["default"])
    t = pdef["type"]
    if t == "bool":
        return trial.suggest_categorical(name, [True, False])
    if t == "categorical":
        return trial.suggest_categorical(name, list(spec.get("choices") or pdef["choices"]))
    lo, hi = float(spec.get("low", pdef.get("low"))), float(spec.get("high", pdef.get("high")))
    log = bool(spec.get("log", pdef.get("log", False)))
    step = spec.get("step")
    if t == "int":
        return trial.suggest_int(name, int(lo), int(hi), log=log, step=int(step) if step and not log else 1)
    return trial.suggest_float(name, lo, hi, log=log, step=float(step) if step and not log else None)


def grid_space(space: dict, task) -> dict:
    """The search space of a grid study: categorical / bool choices, and stepped numeric ranges."""
    out = {}
    for p in task.params:
        sp = space.get(p["name"], {})
        if not sp.get("tune"):
            continue
        if p["type"] == "bool":
            out[p["name"]] = [True, False]
        elif p["type"] == "categorical":
            out[p["name"]] = list(sp.get("choices") or p["choices"])
        else:
            lo, hi = float(sp.get("low", p["low"])), float(sp.get("high", p["high"]))
            step = sp.get("step")
            if not step:
                raise ValueError(f"grid search: give '{p['name']}' a step")
            vals = list(np.arange(lo, hi + 1e-9 * abs(step), float(step)))
            out[p["name"]] = [int(round(v)) for v in vals] if p["type"] == "int" else [float(v) for v in vals]
    return out


def main():
    import optuna
    import torch
    from .tasks import TASKS, Dataset

    study_dir = os.path.abspath(sys.argv[1])
    cfg = json.load(open(os.path.join(study_dir, "config.json")))
    task = TASKS[cfg["task"]]
    status_path = os.path.join(study_dir, "status.json")
    log_path = os.path.join(study_dir, "log.txt")
    os.makedirs(os.path.join(study_dir, "trials"), exist_ok=True)
    status = {"state": "starting", "pid": os.getpid(), "started": time.time(), "message": "Starting",
              "trial": None, "host": os.uname().nodename}
    _write_json(status_path, status)
    cancel = threading.Event()
    lock = threading.Lock()

    def log(msg):
        line = f"{datetime.now().strftime('%H:%M:%S')}  {msg}"
        with lock:
            with open(log_path, "a") as f:
                f.write(line + "\n")
            status["message"] = msg
            _write_json(status_path, status)

    def on_term(signum, frame):
        cancel.set()
        log("Stop requested: cancelling the running trial")
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    try:
        from ..denoise import pick_device
        device = pick_device(cfg.get("device") or "auto")
        log(f"Task {task.label}; dataset {cfg['dataset'].get('name') or cfg['dataset']}; device {device}")
        ds = Dataset(cfg["dataset"], cfg.get("workdir") or os.path.dirname(os.path.dirname(study_dir)))
        status["state"] = "preparing"
        ctx = task.prepare(ds, cfg.get("options", {}), device, log, cancel, study_dir)
        objectives = cfg["objectives"]
        directions = [o.get("direction") or task.metrics[o["metric"]]["direction"] for o in objectives]
        space = cfg.get("space", {})
        pdefs = {p["name"]: p for p in task.params}
        seed = int(cfg.get("seed", 0))
        name = cfg.get("sampler", "tpe")
        if name == "random":
            sampler = optuna.samplers.RandomSampler(seed=seed)
        elif name == "nsga2":
            sampler = optuna.samplers.NSGAIISampler(seed=seed)
        elif name == "grid":
            sampler = optuna.samplers.GridSampler(grid_space(space, task), seed=seed)
        else:
            # multivariate TPE (Falkner et al. 2018) models interactions between parameters
            sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True,
                                                 n_startup_trials=min(10, max(3, int(cfg.get("n_trials", 30)) // 4)))
        storage = f"sqlite:///{os.path.join(study_dir, 'study.db')}"
        study = optuna.create_study(study_name=cfg["name"], storage=storage, directions=directions, sampler=sampler,
                                    load_if_exists=True)
        study.set_metric_names([o["metric"] for o in objectives])
        if cfg.get("baseline", True) and not study.trials and name != "grid":
            base = {k: v for k, v in task.defaults().items() if space.get(k, {}).get("tune")}
            if base:
                study.enqueue_trial(base, user_attrs={"baseline": True})
        stretch_path = os.path.join(study_dir, "stretch.json")
        n_trials = int(cfg.get("n_trials", 30))
        status["state"] = "running"
        status["n_trials"] = n_trials

        def objective(trial):
            if cancel.is_set():
                study.stop()
                raise optuna.TrialPruned("stopped")
            p = {k: suggest(trial, k, space.get(k, {}), pdefs[k]) for k in pdefs}
            status["trial"] = trial.number
            log(f"Trial {trial.number}: " + ", ".join(f"{k}={v}" for k, v in p.items() if space.get(k, {}).get("tune")))
            trial.set_user_attr("params_all", p)
            t0 = time.time()
            try:
                m, img = task.run(ctx, p, log, cancel)
            except Exception as e:
                if cancel.is_set():
                    study.stop()
                    raise optuna.TrialPruned("stopped")
                trial.set_user_attr("error", f"{type(e).__name__}: {e}")
                log(f"Trial {trial.number} failed: {type(e).__name__}: {e}")
                with open(log_path, "a") as f:
                    f.write(traceback.format_exc())
                raise
            finally:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                elif device.type == "mps":
                    torch.mps.empty_cache()
            m = json.loads(json.dumps(m, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
            trial.set_user_attr("metrics", m)
            trial.set_user_attr("wall_seconds", time.time() - t0)
            if img is not None:
                from .tasks import stretch
                import cv2
                ref = json.load(open(stretch_path)) if os.path.exists(stretch_path) else None
                im8, ref = stretch(img, ref)
                if not os.path.exists(stretch_path):
                    _write_json(stretch_path, ref)
                s = min(1.0, 900 / max(im8.shape[:2]))
                if s < 1:
                    im8 = cv2.resize(im8, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(study_dir, "trials", f"{trial.number}.jpg"), im8, [cv2.IMWRITE_JPEG_QUALITY, 90])
            vals = []
            for o in objectives:
                v = m.get(o["metric"])
                if v is None or not math.isfinite(float(v)):
                    raise RuntimeError(f"metric {o['metric']} is not available for this trial")
                vals.append(float(v))
            log(f"Trial {trial.number} done: " + ", ".join(f"{o['metric']} = {v:.5g}" for o, v in zip(objectives, vals)))
            return vals if len(vals) > 1 else vals[0]

        timeout = float(cfg.get("timeout_min") or 0) * 60 or None
        done = sum(1 for t in study.trials if t.state.is_finished())
        study.optimize(objective, n_trials=max(0, n_trials - done), timeout=timeout, catch=(Exception,),
                       gc_after_trial=True)
        status["state"] = "stopped" if cancel.is_set() else "done"
        log("Study stopped" if cancel.is_set() else "Study finished")
    except Exception as e:
        status["state"] = "error"
        status["error"] = f"{type(e).__name__}: {e}"
        with open(log_path, "a") as f:
            f.write(traceback.format_exc())
        log(f"Error: {type(e).__name__}: {e}")
    finally:
        status["ended"] = time.time()
        _write_json(status_path, status)


if __name__ == "__main__":
    main()
