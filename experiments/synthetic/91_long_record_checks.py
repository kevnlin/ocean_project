"""Checks that the 1964-2005 cohort leaves every 2000-2005 input unchanged.

The training-window experiment rebuilds, on ``cesm2_long_uniform``, what the
earlier CESM2 results used: the fixed baselines (``44_synth_argo_oi.py``) and
the local scale tables (``80_local_scale_tables.py``). Both are fitted on
2000-2005 only, so they must come out as before. This compares:

  cohort     every 2000-2005 profile of cesm2_long_uniform.nc with cesm2_uniform.nc
  baselines  OI settings per variable and depth band, and the climatology,
             nearest-profile and OI scores on 2004 and 2005
  tables     the four scale tables

and writes ``outputs/long_record_20261017/checks.json``; it exits non-zero on
any difference. (The surface cache is compared inside 88.)
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/long_record_20261017"
SHORT, LONG = ROOT / "data/synthetic_argo/cesm2_uniform.nc", ROOT / "data/synthetic_argo/cesm2_long_uniform.nc"
BASELINES = ROOT / "outputs/audit/synthetic/fixed_baselines/summary.json"
LONG_BASELINES = ROOT / "outputs/audit/synthetic/fixed_baselines_cesm2_long_uniform/summary.json"
TABLES = ROOT / "outputs/radius_scale_20261008/scale_tables.npz"
LONG_TABLES = OUTPUT / "scale_tables.npz"


def cohort():
    report = {}
    with xr.open_dataset(SHORT) as short, xr.open_dataset(LONG) as long:
        recent = np.asarray(long.month_index.values) >= 0
        if int(recent.sum()) != short.sizes["profile"]:
            return {"ok": False, "reason": "different number of 2000-2005 profiles"}
        for name in short.data_vars:
            a, b = np.asarray(short[name].values), np.asarray(long[name].values)[recent]
            same = (np.array_equal(a, b, equal_nan=True) if a.dtype.kind == "f"
                    else np.array_equal(a.astype(str), b.astype(str)))
            report[name] = bool(same)
        earlier = ~recent
        report["earlier_profiles"] = int(earlier.sum())
        report["earlier_year_split"] = sorted(set(np.asarray(long.year_split.values)[earlier].astype(str)))
        wmo = np.asarray(long.wmo.values).astype(str)
        report["unique_float_ids"] = bool(np.unique(wmo).size == wmo.size)
    report["ok"] = all(v for k, v in report.items() if isinstance(v, bool))
    return report


def scores_equal(a, b, path=""):
    """Every number of ``a`` equals ``b`` to 1e-9 relative; returns the differing paths."""
    if isinstance(a, dict):
        return [d for k in a for d in scores_equal(a[k], b.get(k) if isinstance(b, dict) else None, f"{path}/{k}")]
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        ok = isinstance(b, (int, float)) and np.isclose(a, b, rtol=1e-9, atol=0)
        return [] if ok else [path]
    return [] if a == b else [path]


def baselines():
    short, long = json.loads(BASELINES.read_text()), json.loads(LONG_BASELINES.read_text())
    differ = scores_equal(short["oi_selection"], long["oi_selection"], "oi_selection")
    for method in ("climatology", "nearest_profile", "oi"):
        for split in ("validation", "development"):
            differ += scores_equal(short["baselines"][method]["scores"][split],
                                   long["baselines"][method]["scores"][split], f"{method}/{split}")
    oi = {split: {ch: long["baselines"]["oi"]["scores"][split][ch]["rmse_physical"] for ch in ("TEMP", "SALT")}
          for split in ("validation", "development")}
    return {"ok": not differ, "differing": differ, "oi_rmse_physical": oi}


def tables():
    report = {}
    with np.load(TABLES, allow_pickle=False) as short, np.load(LONG_TABLES, allow_pickle=False) as long:
        for name in ("anom_cell", "anom_month", "res_cell", "res_month"):
            report[name] = {"equal": bool(np.array_equal(short[name], long[name])),
                            "largest_relative_difference": float(np.max(np.abs(long[name] / short[name] - 1)))}
    report["ok"] = all(v["largest_relative_difference"] < 1e-5 for v in report.values())
    return report


def main():
    result = {"cohort": cohort(), "baselines": baselines(), "tables": tables()}
    result["ok"] = all(part["ok"] for part in result.values())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
