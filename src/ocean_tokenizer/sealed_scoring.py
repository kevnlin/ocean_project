"""One-shot scoring of frozen checkpoints on a sealed test: registration, verification, metrics.

Shared by the CESM2 scorer (``109``) and the real-data scorer (``113``). A registration records
every checkpoint and data file by sha256 and is written once; scoring verifies every file first.

``metrics`` turns z-space predictions into physical anomaly units (z times the per-level training
spread; RMSE, bias and the Gaussian scores do not depend on the offset) and reports, for each
variable: RMSE, MAE, signed bias, R², Pearson r, and for the model the Gaussian NLL, CRPS, 68 / 95 %
coverage and mean predictive std (``reconstruction_metrics``); the same regression metrics by depth
band and the RMSE at every level; and the same for the persisted OI and the climatology (a zero
anomaly) on the same cells.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .point_baselines import BANDS, CH, band_of_levels
from .reconstruction_metrics import gaussian_metrics, regression_metrics

FAMILIES = ("grid", "reference")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def register(path, entries, data, meta):
    """Write the registration once: ``entries`` are (family, checkpoint) pairs, ``data`` named files,
    ``meta`` the test's description (dataset, years, leads)."""
    path = Path(path)
    if path.exists():
        raise SystemExit(f"{path} exists: a test is registered once")
    for family, _ in entries:
        if family not in FAMILIES:
            raise SystemExit(f"unknown family {family!r}; one of {FAMILIES}")
    record = {"created_unix": time.time(), **meta,
              "entries": [{"family": family, "checkpoint": str(ckpt), "sha256": sha256(ckpt)} for family, ckpt in entries],
              "data": {name: {"path": str(p), "sha256": sha256(p)} for name, p in data.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=1) + "\n")
    os.replace(temporary, path)
    return record


def verify(record):
    """Every recorded file must still have its recorded content."""
    for item in [*record["entries"], *record["data"].values()]:
        path = Path(item.get("checkpoint", item.get("path")))
        if not path.exists() or sha256(path) != item["sha256"]:
            raise SystemExit(f"{path} changed or vanished since the registration")


def _block(prediction, target, level, bands, n_levels, std=None):
    out = regression_metrics(prediction, target)
    if std is not None:
        out.update({k: v for k, v in gaussian_metrics(prediction, std, target).items() if k != "n"})
    out["by_band"] = {}
    for name in dict.fromkeys(bands):
        mask = bands == name
        if np.isfinite(target[mask]).any():
            out["by_band"][str(name)] = regression_metrics(prediction[mask], target[mask])
    out["by_level"] = []
    for i in range(n_levels):
        mask = (level == i) & np.isfinite(target)
        out["by_level"].append(float(np.sqrt(np.mean((prediction[mask] - target[mask]) ** 2))) if mask.any() else None)
    return out


def metrics(mean_z, std_z, base_z, target_z, level, norm, levels_m):
    """Physical metrics of the model, the persisted OI and the climatology on the same cells."""
    level = np.asarray(level, int)
    bands = band_of_levels(np.asarray(levels_m), BANDS)[level]
    result = {}
    for j, ch in enumerate(CH):
        scale = np.asarray(norm.std[ch], dtype=np.float64)[level]
        target = np.asarray(target_z, dtype=np.float64)[:, j] * scale
        keep = np.isfinite(target)
        physical = lambda z: np.asarray(z, dtype=np.float64)[:, j] * scale
        mean, base = physical(mean_z), physical(base_z)
        sigma = None if std_z is None else np.asarray(std_z, dtype=np.float64)[:, j] * scale
        cut = lambda a: None if a is None else a[keep]
        result[ch] = {
            "model": _block(mean[keep], target[keep], level[keep], bands[keep], len(levels_m), cut(sigma)),
            "persisted_oi": _block(base[keep], target[keep], level[keep], bands[keep], len(levels_m)),
            "climatology": _block(np.zeros(keep.sum()), target[keep], level[keep], bands[keep], len(levels_m))}
    return result


def collect(model, data, analysis, evs, context_profiles, amp, chunk=2048, seed=20260918):
    """Predictions of a model and the persisted OI at every cell of ``evs``, by lead (z units)."""
    import torch
    out = {}
    with torch.no_grad():
        for ev in evs:
            geometry = data.geometry(ev.source, ev.target[ev.profile], 32, ev.level)
            state, parts = None, {k: [] for k in ("mean", "std", "base", "target", "level", "month")}
            for start in range(0, len(ev.level), chunk):
                stop = min(start + chunk, len(ev.level))
                rows, lev = ev.target[ev.profile[start:stop]], ev.level[start:stop]
                piece = {k: v[start:stop] for k, v in geometry.items()}
                scene = data.scene(ev.source, rows, lev, analysis, context_profiles, [seed, ev.month], piece)
                with torch.autocast(device_type=data.device.type, dtype=torch.bfloat16,
                                    enabled=amp and data.device.type == "cuda"):
                    if state is None:
                        state = model.encode(scene)
                    prediction = model.decode(scene, state)
                mean = data.to_z(prediction["mean"].float(), rows, lev)
                if not torch.isfinite(mean).all():
                    raise FloatingPointError(f"nonfinite prediction in month {ev.month}, lead {ev.lead}")
                parts["mean"].append(mean.cpu().numpy())
                parts["std"].append(data.to_z(prediction["std"].float(), rows, lev).cpu().numpy())
                parts["base"].append(data.to_z(scene["baseline"], rows, lev).cpu().numpy())
                parts["target"].append(data.targets(rows, lev)[0].cpu().numpy())
                parts["level"].append(np.asarray(lev))
                parts["month"].append(np.full(len(lev), ev.month))
            bucket = out.setdefault(ev.lead, {k: [] for k in parts})
            for k, v in parts.items():
                bucket[k].append(np.concatenate(v))
    return {lead: {k: np.concatenate(v) for k, v in arrays.items()} for lead, arrays in out.items()}


def summarize(collected, norm, levels_m, years):
    """``metrics`` by lead, over all test years and for each year."""
    result = {}
    for lead, a in sorted(collected.items()):
        year = 2000 + a["month"] // 12
        result[str(lead)] = {"all": metrics(a["mean"], a["std"], a["base"], a["target"], a["level"], norm, levels_m),
                             "cells": int(len(a["level"]))}
        for y in years:
            keep = year == y
            if keep.any():
                result[str(lead)][str(y)] = metrics(a["mean"][keep], a["std"][keep], a["base"][keep],
                                                    a["target"][keep], a["level"][keep], norm, levels_m)
    return result
