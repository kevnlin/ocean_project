"""Damped persistence on the sealed tests, its factors fitted on the selection years only.

Added after both tests were opened (2026-10-20, at Zihao's request for more baselines); nothing in
it is chosen on test data. The forecast of the anomaly at lead l is alpha times the persisted OI
analysis of the input month, with one alpha per (lead, variable, depth band) fitted by least squares
in physical units on the selection years (CESM2 2004-2005, real 2021). The fit uses the persisted-OI
and target arrays a run saved on those years (they do not depend on the run's model). On a test,
cells, persisted OI and targets come from the scenes and the loop of 109 / 113, without a model; the
persisted-OI RMSE must equal the one 109 / 113 wrote, which checks that the cells are the same.

  .venv/bin/python experiments/synthetic/121_statistical_baselines.py --fit
  .venv/bin/python experiments/synthetic/121_statistical_baselines.py --score cesm2|real   # once per test
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ocean_tokenizer import calibration as CAL
from ocean_tokenizer import sealed_scoring as SS
from ocean_tokenizer.point_baselines import BANDS, CH, band_of_levels

OUTPUT = ROOT / "outputs/statistical_baselines_20261020"
ALPHAS = OUTPUT / "alphas.json"
#: the run whose saved selection-year arrays give the persisted OI and targets, and whose checkpoint
#: gives the scene settings (context profiles, amp) of the test loop
SOURCES = {"cesm2": Path("/DATA2/linke/ocean_project_ka/outputs/grid_ka_70k_20261019/gj_flow_s1234"),
           "real": Path("/DATA2/linke/ocean_project_ka_dev/outputs/real_global_20261019/scratch_s1234")}
SCORERS = {"cesm2": ROOT / "experiments/synthetic/109_sealed_cesm2_test.py",
           "real": ROOT / "experiments/real_data/113_sealed_real_test.py"}
SEALED_YEARS = {2006, 2007, 2022, 2023, 2024, 2025}


def fit_alphas(base, target, lead, level, levels_m):
    """``{lead: {variable: {band: alpha}}}``, the least-squares damping of each group's persisted OI."""
    bands = band_of_levels(np.asarray(levels_m), BANDS)[np.asarray(level, int)]
    out = {}
    for value in sorted(set(np.asarray(lead).tolist())):
        out[str(value)] = {}
        for j, ch in enumerate(CH):
            out[str(value)][ch] = {}
            for name in dict.fromkeys(band_of_levels(np.asarray(levels_m), BANDS)):
                keep = (lead == value) & (bands == name) & np.isfinite(target[:, j]) & np.isfinite(base[:, j])
                if keep.any():
                    x, y = base[keep, j].astype(np.float64), target[keep, j].astype(np.float64)
                    out[str(value)][ch][str(name)] = float((x * y).sum() / (x * x).sum())
    return out


def damped(base, lead, level, levels_m, alphas):
    """Each cell's persisted OI times its group's alpha (the factor product of ``calibration``)."""
    return CAL.apply_scales(base, lead, level, levels_m, alphas)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def fit():
    record = {"created_unix": time.time(), "method": __doc__.split("\n\n")[1], "tests": {}}
    for test, run in SOURCES.items():
        summary = json.loads(next(run.glob("summary_seed*.json")).read_text())
        if set(summary["val_years"]) & SEALED_YEARS:
            raise SystemExit(f"{run}: selection years include a sealed year")
        path = run / f"forecast_seed{summary['seed']}.npz"
        with np.load(path, allow_pickle=False) as z:
            if not set((2000 + z["month"] // 12).tolist()) <= set(summary["val_years"]):
                raise SystemExit(f"{path}: arrays outside the selection years")
            scale = z["normalization_scale"][z["level"]]
            alphas = fit_alphas(z["baseline"] * scale, z["target"] * scale, z["lead"].astype(int), z["level"],
                                summary["data"]["levels_m"])
        record["tests"][test] = {"source": str(path), "source_sha256": sha256(path), "fit_years": summary["val_years"],
                                 "levels_m": summary["data"]["levels_m"], "alphas": alphas}
        print(test, {l: round(alphas[l]["TEMP"]["0-100m"], 3) for l in alphas})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if ALPHAS.exists():
        raise SystemExit(f"{ALPHAS} exists: the factors are fitted once")
    ALPHAS.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {ALPHAS}")


def test_cells(test, device):
    """Persisted OI, targets, levels and months of every test cell, by lead, as 109 / 113 collect them."""
    import torch
    spec = importlib.util.spec_from_file_location(f"scorer_{test}", SCORERS[test])
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    seed = json.loads(next(SOURCES[test].glob("summary_seed*.json")).read_text())["seed"]
    training = torch.load(SOURCES[test] / f"best_seed{seed}.pt", map_location="cpu", weights_only=True)["training"]
    if test == "cesm2":
        data = scorer.scenes("grid", training, device)
    else:
        from ocean_tokenizer.real_forecast_scenes import RealGridForecastScenes
        data = RealGridForecastScenes(ROOT, device, caches=scorer.CACHES, oi_summary=scorer.DATA["oi_summary"],
                                      leads=scorer.LEADS)
    analysis, context = data.analysis(), training.get("eval_context_profiles") or training["context_profiles"]
    out = {}
    with torch.no_grad():
        for ev in data.forecast_months(list(scorer.TEST_YEARS), 0):
            geometry = data.geometry(ev.source, ev.target[ev.profile], 32, ev.level)
            bucket = out.setdefault(ev.lead, {k: [] for k in ("base", "target", "level", "month")})
            for start in range(0, len(ev.level), 2048):
                stop = min(start + 2048, len(ev.level))
                rows, lev = ev.target[ev.profile[start:stop]], ev.level[start:stop]
                scene = data.scene(ev.source, rows, lev, analysis, context, [20260918, ev.month],
                                   {k: v[start:stop] for k, v in geometry.items()})
                bucket["base"].append(data.to_z(scene["baseline"], rows, lev).cpu().numpy())
                bucket["target"].append(data.targets(rows, lev)[0].cpu().numpy())
                bucket["level"].append(np.asarray(lev))
                bucket["month"].append(np.full(stop - start, ev.month))
    return data, scorer, {lead: {k: np.concatenate(v) for k, v in a.items()} for lead, a in out.items()}


def score(test, device):
    target_path = Path(SCORERS[test]).parents[2] / ("outputs/cesm2_test_20261019" if test == "cesm2"
                                                    else "outputs/real_global_20261019/sealed")
    results_path, done = target_path / "results.json", target_path / "results_statistical.json"
    if done.exists():
        raise SystemExit(f"{done} exists: the statistical baselines are scored once")
    record = json.loads(ALPHAS.read_text())["tests"][test]
    data, scorer, cells = test_cells(test, device)
    reference = json.loads(results_path.read_text())["entries"][0]["by_lead"]
    out = {"alphas_sha256": sha256(ALPHAS), "method": "damped persistence", "by_lead": {}}
    for lead, a in sorted(cells.items()):
        # alphas are fitted in physical units; one factor per cell, so applying it in z units is the same
        mean = damped(a["base"], np.full(len(a["level"]), lead), a["level"], record["levels_m"], record["alphas"])
        block = {"all": SS.metrics(mean, None, a["base"], a["target"], a["level"], data.norm, data.levels),
                 "cells": int(len(a["level"]))}
        year = 2000 + a["month"] // 12
        for y in scorer.TEST_YEARS:
            keep = year == y
            block[str(y)] = SS.metrics(mean[keep], None, a["base"][keep], a["target"][keep], a["level"][keep],
                                       data.norm, data.levels)
        for ch in CH:
            mine, theirs = block["all"][ch]["persisted_oi"]["rmse"], reference[str(lead)]["all"][ch]["persisted_oi"]["rmse"]
            if abs(mine / theirs - 1) > 1e-4 or block["cells"] != reference[str(lead)]["cells"]:
                raise SystemExit(f"lead {lead} {ch}: persisted OI {mine} differs from results.json {theirs}; not the same cells")
        out["by_lead"][str(lead)] = block
        print(json.dumps({"lead": lead, **{ch: {"damped": round(block["all"][ch]["model"]["rmse"], 4),
                                               "persisted": round(block["all"][ch]["persisted_oi"]["rmse"], 4)} for ch in CH}}),
              flush=True)
    temporary = done.with_suffix(".tmp")
    temporary.write_text(json.dumps(out, indent=1, default=float) + "\n")
    os.replace(temporary, done)
    print(f"wrote {done}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--score", choices=["cesm2", "real"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.fit == bool(args.score):
        raise SystemExit("pass --fit or --score TEST")
    fit() if args.fit else score(args.score, args.device)


if __name__ == "__main__":
    main()
