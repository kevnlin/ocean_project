"""GLORYS12 and ARMOR3D scored at the paper's held-out real Argo cells: reference analyses.

Both products assimilate every Argo profile, including the held-out floats scored here, so
their rows are **reference analyses, never fair baselines** (they have seen the answer). The
script reads each product at the exact position, depth and month of every held-out cell of a
test period, with the paper's anomaly (WOA23 at the profile's position), the paper's z-scoring
(2016-2020 per-level statistics) and the paper's cells (``real_global.real_eval_month`` with
every held-out cell), and writes the RMSE, MAE and bias in physical units by variable, depth
band, level and overall for

  (a) the product's analysis of the target month        ("reference analysis", lead 0)
  (b) the analysis of the input month persisted to the target month, leads 1, 3, 6
      ("persisted analysis", the product's analogue of the paper's persisted-OI row): the
      product's ANOMALY of the input month (its field minus WOA23 of the input month's calendar
      month at the float's position) is carried to the target month, as the persisted OI carries
      an anomaly; persisting the absolute field would carry the seasonal cycle of the input month
      (reported as ``product_absolute_persisted`` for the record, not a row).

One global month is downloaded at a time (``copernicus_reference.fetch_month``), sampled at the
held-out profiles of every target month it serves and deleted; the sampled columns (a few MB)
are cached under ``--cache`` so a period is re-scored without re-downloading.

The 2024 period reads the sealed 2024 test's held-out values again: a declared extra pass on an
already-opened test; no registration or results file of a sealed test is touched.

  .venv/bin/python experiments/real_data/160_reference_products.py --product glorys12 --period 2021
  .venv/bin/python experiments/real_data/160_reference_products.py --product armor3d --period 2024
  .venv/bin/python experiments/real_data/160_reference_products.py --product glorys12 --period 2022-2023
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from ocean_tokenizer import copernicus_reference as CR
from ocean_tokenizer.audit_tools import woa_monthly
from ocean_tokenizer.point_baselines import CH
from ocean_tokenizer.real_global import load_real, raw_profiles, real_eval_month

OUTPUT = ROOT / "outputs/reference_products_20261021"
LEADS = (0, 1, 3, 6)
PERIODS = {"2021": (2021,), "2024": (2024,), "2022-2023": (2022, 2023)}
COHORT_YEARS = (2016, 2024)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def cells_of(c, obs, years):
    """``{(target month, lead): ev}`` of every held-out cell of ``years`` (the sealed tests' cells)."""
    months = sorted(set(int(m) for m in np.unique(c.month_index)))
    out = {}
    for m in months:
        if 2000 + m // 12 not in set(years) or not c.month(m, float_split="heldout_float").size:
            continue
        for lead in LEADS:
            if m - lead not in months:
                continue
            ev = real_eval_month(c, obs, m, 0, lead)
            if ev["src"].size:
                out[(m, lead)] = ev
    return out


_WOA = {}


def woa_columns(root, lat, lon, calendar_month, levels, chunk=20000):
    """WOA23 at (lat, lon) for ``calendar_month`` (1-12) on ``levels``: ``{channel: (R, L)}``, the
    interpolation of ``audit_tools.climatology_anomaly`` with the calendar month chosen freely."""
    import xarray as xr
    if not _WOA:
        w = woa_monthly(str(root))
        _WOA.update({ch: w[ch].load() for ch in CH})
    lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
    cm = np.full(lat.size, int(calendar_month))
    out = {}
    for ch in CH:
        da = _WOA[ch]
        clim = np.empty((lat.size, len(levels)), dtype="float32")
        for i in range(0, lat.size, chunk):
            sl = slice(i, i + chunk)
            clim[sl] = da.interp(time=xr.DataArray(cm[sl], dims="p"), lat=xr.DataArray(lat[sl], dims="p"),
                                 lon=xr.DataArray(lon[sl] % 360.0, dims="p"), depth=xr.DataArray(levels, dims="l"),
                                 method="linear").transpose("p", "l").values
        out[ch] = clim.astype(np.float64)
    return out


def sample(product, evs, c, levels, cache_path, download_dir, max_months=0):
    """Product columns at the held-out profiles of every target month, keyed by source month and
    target month; cached as an npz."""
    cache = {}
    if cache_path.exists():
        with np.load(cache_path) as z:
            cache = {k: z[k] for k in z.files}
    sources = sorted({m - lead for m, lead in evs})
    if max_months:
        sources = sources[:max_months]
    log = []
    for s in sources:
        targets = sorted({m for m, lead in evs if m - lead == s})
        keys = [f"{s}:{m}:{ch}" for m in targets for ch in CH]
        if all(k in cache for k in keys):
            continue
        record = CR.fetch_month(product, s, download_dir)
        try:
            field = CR.MonthField(record["file"], product)
            if field.time != CR.month_label(s):
                raise ValueError(f"asked for {CR.month_label(s)}, the file holds {field.time}")
            for m in targets:
                rows = c.month(m, float_split="heldout_float")
                cols = field.columns(c.lat[rows], c.lon[rows], levels)
                for ch in CH:
                    cache[f"{s}:{m}:{ch}"] = cols[ch].astype(np.float32)
            del field
        finally:
            os.remove(record["file"])
        record.pop("file")
        record["targets"] = [CR.month_label(m) for m in targets]
        log.append(record)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **cache)
        print(json.dumps({"downloaded": record["month"], "transfer_mb": round(record["transfer_mb"], 1),
                          "seconds": round(record["seconds"], 1), "targets": record["targets"]}), flush=True)
    return cache, log


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--product", required=True, choices=sorted(CR.PRODUCTS))
    p.add_argument("--period", required=True, choices=sorted(PERIODS))
    p.add_argument("--cache", default=None, help="directory of the sampled-column caches (default: OUTPUT/cache)")
    p.add_argument("--download-dir", default=None, help="where one month at a time is downloaded (default: OUTPUT/tmp)")
    p.add_argument("--max-months", type=int, default=0, help="smoke: only the first N source months")
    args = p.parse_args()
    started = time.time()
    years = PERIODS[args.period]
    cache_dir = Path(args.cache) if args.cache else OUTPUT / "cache"
    download_dir = Path(args.download_dir) if args.download_dir else OUTPUT / "tmp"
    OUTPUT.mkdir(parents=True, exist_ok=True)

    c, norm, obs = load_real(ROOT, years=COHORT_YEARS)
    levels = np.asarray(c.levels, np.float64)
    temp, salt, _ = raw_profiles(ROOT, COHORT_YEARS)
    climate = {"TEMP": temp - c.TEMP, "SALT": salt - c.SALT}       # WOA23 at the profile's position and month
    evs = cells_of(c, obs, years)
    cache, log = sample(args.product, evs, c, levels, cache_dir / f"{args.product}_{args.period}.npz", download_dir,
                        args.max_months)

    result = {"product": args.product, **{k: v for k, v in CR.PRODUCTS[args.product].items() if k != "variables"},
              "variables": CR.PRODUCTS[args.product]["variables"], "period": args.period, "years": list(years),
              "role": "reference analysis: the product assimilates every Argo profile, including the held-out "
                      "floats scored here, so it is a reference, never a fair baseline",
              "cells": "every held-out-float cell of the period (real_global.real_eval_month, max_cells=0), "
                       "scored where the target is finite; the sealed tests' cells",
              "target": "WOA23 anomaly at the profile's position, z-scored per level with the 2016-2020 "
                        "statistics, values beyond 10 standard deviations missing (real_global.load_real)",
              "product_anomaly": "product read at the profile's position (bilinear in lat/lon, linear in depth onto "
                                 "the 20 levels, no extrapolation) minus the same WOA23 climatology at the position "
                                 "(the calendar month of the product's month), z-scored with the same per-level "
                                 "statistics; errors in physical units are (z_product - z_target) * std[level], as "
                                 "sealed_scoring.metrics",
              "leads": {"0": "reference analysis of the target month",
                        "1,3,6": "the product's anomaly of the input month (target month minus lead), persisted to the "
                                 "target month, as the paper's persisted OI persists an anomaly"},
              "levels_m": levels.tolist(),
              "normalization": {kind: {ch: np.asarray(getattr(norm, kind)[ch]).tolist() for ch in CH} for kind in ("mean", "std")},
              "cohort": {"path": str(ROOT / "data/argo_cohort/global_global.nc"),
                         "sha256": sha256(ROOT / "data/argo_cohort/global_global.nc"),
                         "anomalies": str(ROOT / "data/argo_cohort/global_anomx.npz"),
                         "anomalies_sha256": sha256(ROOT / "data/argo_cohort/global_anomx.npz"), "years": list(COHORT_YEARS)},
              "downloads": log, "by_lead": {}, "by_lead_by_year": {}}
    if args.period == "2024":
        result["note"] = ("the 2024 held-out values were read again: a declared extra pass on an already-opened "
                          "test; no registration or results file of a sealed test was touched")
    woa_check = 0.
    for lead in LEADS:
        parts = {"pred": {ch: [] for ch in CH}, "clim": {ch: [] for ch in CH}, "target": {ch: [] for ch in CH},
                 "clim_target_month": {ch: [] for ch in CH}, "level": [], "month": []}
        months = []
        for (m, ld), ev in sorted(evs.items()):
            if ld != lead:
                continue
            keys = [f"{m - lead}:{m}:{ch}" for ch in CH]
            if not all(k in cache for k in keys):
                continue
            tgt, prof, lev = ev["tgt"], ev["prof"], ev["lev"]
            # WOA23 of the INPUT month's calendar month at the float's position: the product's anomaly
            # of the input month, persisted (lead 0: the target month's own climatology)
            woa = woa_columns(ROOT, c.lat[tgt], c.lon[tgt], (m - lead) % 12 + 1, levels)
            if lead == 0:
                for ch in CH:
                    k = np.isfinite(climate[ch][tgt])
                    woa_check = max(woa_check, float(np.max(np.abs(woa[ch][k] - climate[ch][tgt][k]))))
            for ch in CH:
                parts["pred"][ch].append(cache[f"{m - lead}:{m}:{ch}"].astype(np.float64)[prof, lev])
                parts["clim"][ch].append(woa[ch][prof, lev])
                parts["clim_target_month"][ch].append(climate[ch][tgt][prof, lev])
                parts["target"][ch].append(ev["target"][ch])
            parts["level"].append(lev)
            parts["month"].append(np.full(lev.size, m))
            months.append(m)
        if not months:
            continue
        cat = lambda d: {ch: np.concatenate(v) for ch, v in d.items()}
        pred, clim, target = cat(parts["pred"]), cat(parts["clim"]), cat(parts["target"])
        clim_target = cat(parts["clim_target_month"])
        level, month = np.concatenate(parts["level"]), np.concatenate(parts["month"])
        result["by_lead"][str(lead)] = {"cells": int(level.size), "target_months": [CR.month_label(m) for m in months],
                                        **CR.score_cells(pred, clim, target, norm, level, levels)}
        if lead:
            absolute = CR.score_cells(pred, clim_target, target, norm, level, levels)
            result["by_lead"][str(lead)]["product_absolute_persisted"] = {
                "note": "the absolute field of the input month persisted (carries its seasonal cycle); for the record only",
                **{ch: {k: absolute[ch]["product"][k] for k in ("n", "rmse", "mae", "mean_bias")} for ch in CH}}
        if len(years) > 1:
            for y in years:
                keep = 2000 + month // 12 == y
                result["by_lead_by_year"].setdefault(str(lead), {})[str(y)] = {
                    "cells": int(keep.sum()),
                    **CR.score_cells({ch: v[keep] for ch, v in pred.items()}, {ch: v[keep] for ch, v in clim.items()},
                                     {ch: v[keep] for ch, v in target.items()}, norm, level[keep], levels)}
        print(json.dumps({"lead": lead, "cells": int(level.size), "months": len(months),
                          **{ch: {"rmse": round(result["by_lead"][str(lead)][ch]["product"]["rmse"], 4),
                                  "n": result["by_lead"][str(lead)][ch]["product"]["n"],
                                  "dropped": result["by_lead"][str(lead)][ch]["cells_dropped_product_missing"],
                                  "clim_same": round(result["by_lead"][str(lead)][ch]["climatology_same_cells"]["rmse"], 4)}
                             for ch in CH}}), flush=True)
    result["woa_identity_check"] = {"max_abs_diff": woa_check,
                                    "meaning": "WOA23 read here at the position for the target month vs the cohort's own "
                                               "climatology (measured minus anomaly); lead 0 uses the identical values"}
    print(json.dumps({"woa_identity_check_max_abs_diff": woa_check}), flush=True)
    result["runtime_s"] = time.time() - started
    out = OUTPUT / f"{args.product}_{args.period}.json"
    out.write_text(json.dumps(result, indent=1, default=float) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
