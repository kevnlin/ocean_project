"""Checks of the 10-year training window: the targets stay the paper's.

Compares the data of the ``r_y10`` runs (``load_real(years=(2011, 2024), train_split=(2011, 2020))``,
the caches of ``162``) with the paper's (``load_real()``: 2016-2023, train 2016-2020, the caches of
``105`` in ``outputs/real_global_20261019``) and with a saved checkpoint of the paper's run:

  normalisation   the per-level mean and standard deviation of TEMP and SALT equal to the bit
                  (the new cohort's, the paper's cohort's and the checkpoint's "normalization")
  validation      the 2021 months, their 8,000-cell draws (profile row within the month, level),
                  the drawn profiles (float, position, month, day) and the targets are identical;
                  the input profiles of every validation month-lead are the same floats and positions
  splits          the training months are 2011-01..2020-12 (2016-01..2020-12 before); 2022-2023 stay
                  development, 2024 is outside every split (never trained or validated on)
  scenes          ``RealGridForecastScenes`` on the new caches gives the same 2021 forecast cells and
                  targets as on the paper's caches, the same OI selection, and the 2016-2024 rows of
                  the profile features are the registered ones

Writes ``outputs/real_y10_20261021/checks.json``; any failed check stops the script.

  .venv/bin/python experiments/real_data/164_real_y10_checks.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch

from ocean_tokenizer.point_baselines import CH
from ocean_tokenizer.real_forecast_scenes import RealGridForecastScenes
from ocean_tokenizer.real_global import EVAL_CELLS, REAL_SPLITS, load_real, raw_profiles, real_eval_month

OUTPUT = ROOT / "outputs/real_y10_20261021"
PAPER = ROOT / "outputs/real_global_20261019"
OI_SUMMARY = ROOT / "outputs/audit/real/fixed_baselines_real_global/summary.json"
CHECKPOINT = PAPER / "scratch_s1234/best_seed1234.pt"
OI_COHORT = "6366666ef8ad4ec33388fd8cc7d7f1666236d8f29158b28bee4818f16584e3a1"
NEW_YEARS, TRAIN_SPLIT = (2011, 2024), (2011, 2020)


def require(ok, what):
    print(("ok   " if ok else "FAIL ") + what, flush=True)
    if not ok:
        raise SystemExit(f"check failed: {what}")
    return bool(ok)


def main():
    started = time.time()
    report = {}
    old_c, old_norm, old_obs = load_real(ROOT)
    new_c, new_norm, new_obs = load_real(ROOT, years=NEW_YEARS, train_split=TRAIN_SPLIT)
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["normalization"]
    print(f"cohorts loaded: paper {len(old_c.lat)} rows, 10-year {len(new_c.lat)} rows ({time.time() - started:.0f}s)")

    # normalisation: bit-equal
    norm = {}
    for kind, old_d, new_d in (("mean", old_norm.mean, new_norm.mean), ("std", old_norm.std, new_norm.std)):
        for ch in CH:
            same_paper = np.array_equal(old_d[ch], new_d[ch])
            same_ckpt = np.array_equal(np.asarray(saved[kind][ch]), new_d[ch])
            norm[f"{kind}_{ch}"] = {"equals_paper_cohort": bool(same_paper), "equals_checkpoint": bool(same_ckpt),
                                    "levels": int(new_d[ch].size)}
            require(same_paper and same_ckpt, f"normalisation {kind} {ch} bit-equal to the paper's and the checkpoint's")
    report["normalization"] = norm

    # splits
    def months_of(c, split):
        return [int(m) for m in c.months_in(split)]
    old_train, new_train = months_of(old_c, "train"), months_of(new_c, "train")
    label = lambda m: f"{2000 + m // 12}-{m % 12 + 1:02d}"
    report["splits"] = {"paper_train_months": len(old_train), "paper_train_first_last": [label(old_train[0]), label(old_train[-1])],
                        "new_train_months": len(new_train), "new_train_first_last": [label(new_train[0]), label(new_train[-1])],
                        "new_train_rows": int(np.sum(new_c.year_split == "train")),
                        "paper_train_rows": int(np.sum(old_c.year_split == "train")),
                        "new_rows_by_split": {s: int(np.sum(new_c.year_split == s)) for s in np.unique(new_c.year_split)},
                        "years_2024_split": sorted(set(new_c.year_split[new_c.year == 2024].tolist())),
                        "years_2022_2023_split": sorted(set(new_c.year_split[(new_c.year >= 2022) & (new_c.year <= 2023)].tolist()))}
    require(new_train[0] == 12 * 11 and new_train[-1] == 12 * 20 + 11 and len(new_train) == 120, "training months 2011-01..2020-12")
    require(old_train[0] == 12 * 16 and old_train[-1] == 12 * 20 + 11 and len(old_train) == 60, "paper training months 2016-01..2020-12")
    require(new_train[-60:] == old_train, "the last 60 training months are the paper's")
    require(report["splits"]["years_2024_split"] == ["unassigned"], "2024 rows outside every split")
    require(report["splits"]["years_2022_2023_split"] == ["development"], "2022-2023 rows stay development")
    require(months_of(old_c, "validation") == months_of(new_c, "validation"), "the 2021 validation months")

    # the 2016-2023 rows are the same profiles in the same order
    n_pre = int(np.sum(new_c.year <= 2015))
    n_old = len(old_c.lat)
    same_rows = all(np.array_equal(getattr(old_c, k), getattr(new_c, k)[n_pre:n_pre + n_old]) for k in ("wmo", "lat", "lon", "month_index"))
    same_obs = all(np.array_equal(old_obs[ch], new_obs[ch][n_pre:n_pre + n_old], equal_nan=True) for ch in CH)
    report["rows"] = {"rows_2011_2015": n_pre, "rows_2016_2023": n_old, "rows_2024": int(len(new_c.lat) - n_pre - n_old),
                      "same_profiles_2016_2023": bool(same_rows), "same_z_scores_2016_2023": bool(same_obs)}
    require(same_rows and same_obs, "the 2016-2023 rows and their z-scored targets are the paper's")

    # validation cells and inputs at every lead
    _, _, old_day = raw_profiles(ROOT)
    _, _, new_day = raw_profiles(ROOT, NEW_YEARS)
    cells, inputs = 0, 0
    for m in months_of(old_c, "validation"):
        for lead in (0, 1, 3, 6):
            a = real_eval_month(old_c, old_obs, m, EVAL_CELLS, lead)
            b = real_eval_month(new_c, new_obs, m, EVAL_CELLS, lead)
            require(np.array_equal(a["prof"], b["prof"]) and np.array_equal(a["lev"], b["lev"]), f"{label(m)} lead {lead}: the cell draw")
            require(all(np.array_equal(a["target"][ch], b["target"][ch], equal_nan=True) for ch in CH), f"{label(m)} lead {lead}: the targets")
            for name, rows_a, rows_b in (("targets", a["tgt"], b["tgt"]), ("inputs", a["src"], b["src"])):
                same = all(np.array_equal(getattr(old_c, k)[rows_a], getattr(new_c, k)[rows_b]) for k in ("wmo", "lat", "lon", "month_index"))
                same = same and np.array_equal(old_day[rows_a], new_day[rows_b])
                require(same, f"{label(m)} lead {lead}: the {name} profiles (float, position, month, day)")
            cells += len(a["prof"])
            inputs += len(a["src"])
    report["validation"] = {"months": len(months_of(old_c, "validation")), "leads": [0, 1, 3, 6],
                            "cells_compared": int(cells), "input_profiles_compared": int(inputs), "identical": True}

    # scenes on both caches
    old = RealGridForecastScenes(ROOT, "cpu", caches=PAPER, oi_summary=OI_SUMMARY, leads=(0, 1, 3, 6))
    new = RealGridForecastScenes(ROOT, "cpu", caches=OUTPUT, oi_summary=OI_SUMMARY, leads=(0, 1, 3, 6),
                                 years=NEW_YEARS, train_split=TRAIN_SPLIT, oi_cohort_fingerprint=OI_COHORT)
    require(old.oi_selection == new.oi_selection, "the OI selection of 106 is reused unchanged")
    require(new.metadata["splits"] == {"train": [2011, 2020], "validation": [2021, 2021], "development": [2022, 2023]}
            and new.metadata["normalization_splits"] == {k: list(v) for k, v in REAL_SPLITS.items()}, "scene metadata records both split tables")
    require(len(new.months("train")) == 120 and len(old.months("train")) == 60, "scene training months 120 vs 60")
    old_val, new_val = old.forecast_months([2021], 8000), new.forecast_months([2021], 8000)
    require(len(old_val) == len(new_val), "the number of 2021 forecast month-leads")
    for a, b in zip(old_val, new_val):
        require(a.month == b.month and a.lead == b.lead and np.array_equal(a.profile, b.profile) and np.array_equal(a.level, b.level),
                f"scene {label(a.month)} lead {a.lead}: the cells")
        ta, _ = old.targets(a.target[a.profile], a.level)
        tb, _ = new.targets(b.target[b.profile], b.level)
        require(torch.equal(torch.nan_to_num(ta, nan=-999.), torch.nan_to_num(tb, nan=-999.)), f"scene {label(a.month)} lead {a.lead}: the targets")
        require(torch.equal(old.x[old.indices(a.target)], new.x[new.indices(b.target)]), f"scene {label(a.month)} lead {a.lead}: the query features")
        require(torch.equal(old.x[old.indices(a.source)], new.x[new.indices(b.source)]), f"scene {label(a.month)} lead {a.lead}: the input features")
    require(torch.equal(old.profile_mean, new.profile_mean) and torch.equal(old.profile_std, new.profile_std), "scene normalisation tensors")
    require(torch.equal(old.x, new.x[n_pre:n_pre + n_old]) and torch.equal(old.lagged, new.lagged[n_pre:n_pre + n_old]), "2016-2023 profile and lagged features")
    report["scenes"] = {"month_leads_2021": len(new_val), "cells": int(sum(len(e.level) for e in new_val)), "identical": True,
                        "new_cohort_fingerprint": new.metadata["cohort_fingerprint"], "paper_cohort_fingerprint": old.metadata["cohort_fingerprint"],
                        "new_feature_fingerprint": new.metadata["feature_fingerprint"], "paper_feature_fingerprint": old.metadata["feature_fingerprint"]}
    report["seconds"] = round(time.time() - started)
    path = OUTPUT / "checks.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))
    print(f"all checks passed, wrote {path} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
