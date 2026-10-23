"""The two test cohorts of the sealed CESM2 test v2 (2008-2010) and their caches.

Spec ``docs/superpowers/specs/2026-10-22-sealed-cesm2-test2.md``. For each training cohort of the
comparison (``cesm2_long_uniform`` of ``c_flow100``, ``cesm2_vlong_uniform`` of ``c_y154``):

  data/synthetic_argo/cesm2_long_test2.nc, data/synthetic_argo/cesm2_vlong_test2.nc
      every row of the training cohort unchanged and first, then 2006-01 ... 2010-12 (60 months,
      456,000 profiles) drawn exactly as ``87`` draws a month (generator ``[20261017, month + 10000]``,
      7,600 area-uniform wet positions, 20 % held-out floats, bilinear TEMP/SALT from the 2000-2010
      store of ``166``, anomalies against the same 2000-2003 climatology). The 2006-2007 months equal
      ``cesm2_long_test``'s rows value for value and are kept so that a forecast into 2008-01 ...
      2008-06 reads inputs from 2007-07 ... 2007-12. Year split "unassigned".
  outputs/cesm2_test2_20261022/<cohort>/{satellites,surface_grids,lagged_surface}.npz
      88's features, 100's grids, 93's lagged columns: the training cohort's rows and months copied
      from its registered caches value for value, 2006-2010 by the same recipes from the new store.
  outputs/audit/synthetic/fixed_baselines_<cohort>/summary.json
      ``44 --cohort <cohort>``: OI tuned on 2004, scored on 2005 (training rows); the selection, the
      per-month and the summary scores must equal the training cohort's exactly.

Checks before anything is kept: the climatology rebuilt from the store reproduces ``CLIM_POS`` of the
training cohort; the 2006-2007 draws equal ``cesm2_long_test``; the 2000-2003 normalisation equals the
training cache's; the grid, feature and lag recipes on the new store reproduce the training caches on
2005-07 ... 2005-12 and ``cesm2_test_20261019``'s on 2006-2007. No value of 2008-2010 is scored.
Resumable: an existing cohort file, cache set or OI summary is kept.

  .venv/bin/python experiments/synthetic/167_cesm2_test2_cohorts.py --workers 8
  .venv/bin/python experiments/synthetic/167_cesm2_test2_cohorts.py --cohorts cesm2_long_test2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import xarray as xr

import cohort_caches as H
from ocean_tokenizer import synth_argo_eval as S
from ocean_tokenizer.latent_experiment import array_fingerprint
from ocean_tokenizer.synthetic_latent_experiment import position_features

ROOT = H.ROOT
STORE = ROOT / "data/cesm2_long/cesm2_le_cohort_levels_2000_2010.zarr"
STORE_CHECKS = ROOT / "outputs/cesm2_test2_20261022/store_checks.json"
OUTPUT = ROOT / "outputs/cesm2_test2_20261022"
REFERENCE = ROOT / "data/synthetic_argo/cesm2_long_test.nc"           # the 2006-2007 draws, already made by 110
REFERENCE_CACHES = ROOT / "outputs/cesm2_test_20261019"
VLONG = ROOT / "outputs/cohort_cesm2_vlong_uniform"
#: test cohort -> its training cohort and that cohort's registered caches and OI summary
COHORTS = {
    "cesm2_long_test2": {"train": "cesm2_long_uniform", "satellites": H.OLD_SATELLITES, "grids": H.OLD_GRIDS,
                         "lagged": H.OLD_LAGGED, "oi": H.OLD_OI},
    "cesm2_vlong_test2": {"train": "cesm2_vlong_uniform", "satellites": VLONG / "satellites.npz",
                          "grids": VLONG / "surface_grids.npz", "lagged": VLONG / "lagged_surface.npz",
                          "oi": ROOT / "outputs/audit/synthetic/fixed_baselines_cesm2_vlong_uniform/summary.json"}}
KEPT_MONTHS = tuple(range(72, 96))           # 2006-01 ... 2007-12, equal to cesm2_long_test
TEST_MONTHS = tuple(range(96, 132))          # 2008-01 ... 2010-12
NEW_MONTHS = KEPT_MONTHS + TEST_MONTHS
INPUT_MONTHS = tuple(range(66, 132))         # fields needed: the new months and their inputs back to 2005-07
LEADS = H.LEADS
SEED = H.SEED


def run(script, options, log_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    with log_path.open("a") as log:
        subprocess.run([sys.executable, "-u", str(ROOT / script), *options], cwd=ROOT, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)


# ---------------------------------------------------------------- the cohort file
def build_cohort(name, spec, started):
    new_path = ROOT / f"data/synthetic_argo/{name}.nc"
    train_path = ROOT / f"data/synthetic_argo/{spec['train']}.nc"
    checks = {}
    store = H.Store(STORE)
    if not np.array_equal(store.months, np.arange(0, 132)):
        raise SystemExit(f"{STORE.name} must hold 2000-01 ... 2010-12")
    store.fit_climatology()
    train_ds = xr.open_dataset(train_path)
    if not np.allclose(store.levels, train_ds.level.values):
        raise SystemExit("store and cohort levels differ")
    store.check_climatology(train_ds)
    H.log(f"climatology from {STORE.name} reproduces CLIM_POS of {train_path.name}", started)
    checks["climatology_reproduces_training_clim_pos"] = True
    slices = H.month_rows(train_ds)
    if max(slices) != 71:
        raise SystemExit(f"{train_path.name} does not end at 2005-12")
    old = H.load_rows(train_ds)
    attrs = dict(train_ds.attrs)
    levels = train_ds.level.values
    train_ds.close()
    n_old = old["lat"].size
    H.log(f"{train_path.name}: {n_old:,} rows loaded", started)

    # the reference 2006-2007 rows of cesm2_long_test (110's draws from the 1964-2007 store)
    ref_ds = xr.open_dataset(REFERENCE)
    rows_before = json.loads(ref_ds.attrs["test_extension"])["rows_before"]
    ref_ds = ref_ds.isel(profile=slice(rows_before, None))
    ref = H.load_rows(ref_ds)
    ref_slices = H.month_rows(ref_ds)
    ref_ds.close()
    if sorted(ref_slices) != list(KEPT_MONTHS):
        raise SystemExit(f"{REFERENCE.name} does not hold exactly 2006-01 ... 2007-12 after its training rows")

    pieces = {k: [] for k in H.ALL_VARS}
    n_levels = levels.size
    for t in NEW_MONTHS:
        col, arr = store.draw_87(t)
        if t in ref_slices and not H.rows_equal(col, arr, ref, ref_slices[t]):
            raise SystemExit(f"87's draw of month {t} from {STORE.name} does not reproduce {REFERENCE.name}")
        for k in H.COLUMNS:
            pieces[k].append(col[k] if k in ("wmo", "float_split") else col[k].astype(old[k].dtype))
        pieces["year"].append(np.full(H.N_PER_MONTH, 2000 + t // 12, old["year"].dtype))
        pieces["year_split"].append(np.full(H.N_PER_MONTH, "unassigned"))
        for k in H.ARRAYS:
            pieces[k].append(arr[k])
        for k in ("TEMP_ERR", "SALT_ERR"):
            pieces[k].append(np.full((H.N_PER_MONTH, n_levels), np.nan, "float32"))
    store.close()
    del ref
    checks["draws_2006_2007_equal_cesm2_long_test"] = True
    checks["months_added"] = [NEW_MONTHS[0], NEW_MONTHS[-1]]
    H.log(f"drew {len(NEW_MONTHS)} months {NEW_MONTHS[0]} ... {NEW_MONTHS[-1]}; 2006-2007 equal {REFERENCE.name}", started)

    data = {}
    for k in H.ALL_VARS:
        data[k] = np.concatenate([old[k], *pieces[k]])          # training rows unchanged and first
        old[k] = None
        pieces[k] = None
    if (np.diff(data["month_index"]) < 0).any():
        raise SystemExit("rows are not sorted by month")
    if np.unique(data["wmo"]).size != data["wmo"].size:
        raise SystemExit("float ids are not unique")
    n_new = data["lat"].size - n_old
    if n_new != len(NEW_MONTHS) * H.N_PER_MONTH:
        raise SystemExit("wrong number of new rows")
    data_vars = {k: (("profile", "level") if data[k].ndim == 2 else "profile", data[k]) for k in H.ALL_VARS}
    attrs["test_extension"] = json.dumps({
        "months": "2006-01 to 2010-12", "sealed_months": "2008-01 to 2010-12", "seed": SEED,
        "per_month_generator": "numpy default_rng([seed, month_index + 10000]) (87's draw_month)",
        "store": str(STORE.relative_to(ROOT)), "rows_before": int(n_old), "rows_added": int(n_new),
        "training_cohort": spec["train"], "kept_2006_2007": f"equal to {REFERENCE.name} value for value",
        "climatology": "2000-2003, identical to cesm2_long_uniform.nc",
        "spec": "docs/superpowers/specs/2026-10-22-sealed-cesm2-test2.md",
        "script": "experiments/synthetic/167_cesm2_test2_cohorts.py"})
    n = H.write_cohort(new_path, data_vars, levels, attrs)
    checks["profiles"], checks["rows_copied"], checks["rows_added"] = int(n), int(n_old), int(n_new)
    H.log(f"{n:,} profiles ({n_old:,} copied + {n_new:,} added) -> {new_path}", started)
    return checks


# ---------------------------------------------------------------- the caches
def reference_rows(path):
    """{month: rows} of a cache's rows by month, in the cache's own order."""
    with np.load(path, allow_pickle=False) as cache:
        month = cache["month_index"]
    return {int(m): np.flatnonzero(month == m) for m in np.unique(month)}


def build_caches(name, spec, output, workers, started):
    checks = {}
    c, norm, obs = S.load(str(ROOT), cohort=name)
    fingerprint = array_fingerprint((c.month_index, c.lat, c.lon, c.wmo, c.levels,
                                     np.stack([obs[ch] for ch in ("TEMP", "SALT")], -1)))
    H.log(f"cohort {name} loaded: {len(c.lat):,} profiles", started)
    with np.load(spec["satellites"], allow_pickle=False) as cache:
        train_sat = {k: cache[k] for k in cache.files}
    train_meta = json.loads(str(train_sat["metadata"].item()))
    n_old = len(train_sat["month_index"])
    climate, mean, std = train_sat["surface_climatology"], train_sat["surface_mean"], train_sat["surface_std"]
    grid_lat, grid_lon = train_sat["grid_lat"], train_sat["grid_lon"]
    for k in ("lat", "lon", "month_index"):
        if not np.array_equal(train_sat[k], getattr(c, k)[:n_old]):
            raise SystemExit(f"{k} of the first {n_old:,} rows differs from {spec['satellites']}")
    with xr.open_dataset(ROOT / f"data/synthetic_argo/{spec['train']}.nc") as train_ds:
        if not np.array_equal(np.asarray(train_ds.wmo.values).astype(str), c.wmo[:n_old]):
            raise SystemExit(f"wmo of the first {n_old:,} rows differs from {spec['train']}.nc")
    if not np.array_equal(c.month_index[n_old:], np.repeat(np.array(NEW_MONTHS), H.N_PER_MONTH)):
        raise SystemExit("the new rows are not the 60 new months in order")
    new_norm = {"mean": np.stack([norm.mean[ch] for ch in ("TEMP", "SALT")], -1),
                "std": np.stack([norm.std[ch] for ch in ("TEMP", "SALT")], -1)}
    for k in ("mean", "std"):
        diff = float(np.abs(new_norm[k] - train_sat[f"normalization_{k}"]).max())
        checks[f"normalization_{k}_largest_difference"] = diff
        if diff > 1e-9:
            raise SystemExit(f"the 2000-2003 normalisation {k} differs from the training cache by {diff:.3g}")
    checks["training_rows_first_and_unchanged"] = int(n_old)
    H.log("training rows are first and unchanged; the 2000-2003 normalisation matches the training cache", started)

    fields = H.surface_fields(STORE, list(INPUT_MONTHS), c.levels, workers, started)
    z = {m: (f - climate[m % 12] - mean[:, None, None]) / std[:, None, None] for m, f in fields.items()}
    H.log(f"surface fields and z-scores of months {INPUT_MONTHS[0]} ... {INPUT_MONTHS[-1]} from {STORE.name}", started)

    # grids: the recipe reproduces the training cache on 2005-07 ... 2005-12 and 110's on 2006-2007
    with np.load(spec["grids"], allow_pickle=False) as cache:
        train_grids, train_grid_month = cache["grids"], cache["grid_month"]
    with np.load(REFERENCE_CACHES / "surface_grids.npz", allow_pickle=False) as cache:
        ref_grids, ref_grid_month = cache["grids"], cache["grid_month"]

    def grid_difference(stored, m):
        stored = stored.astype(float)
        if not np.array_equal(np.isfinite(stored), np.isfinite(z[m])):
            raise SystemExit(f"grid validity of month {m} differs")
        both = np.isfinite(stored)
        return float((np.abs(stored - z[m]) / np.maximum(1., np.abs(z[m])))[both].max())

    worst_train = max(grid_difference(train_grids[int(np.flatnonzero(train_grid_month == m)[0])], m) for m in range(66, 72))
    worst_ref = max(grid_difference(ref_grids[int(np.flatnonzero(ref_grid_month == m)[0])], m) for m in KEPT_MONTHS)
    checks["grids_largest_relative_difference_2005_07_12_vs_training_cache"] = worst_train
    checks["grids_largest_relative_difference_2006_2007_vs_cesm2_test_20261019"] = worst_ref
    if max(worst_train, worst_ref) > 1e-3:                       # float16 storage
        raise SystemExit("the grid recipe does not reproduce the existing caches")
    H.log(f"grid recipe reproduces the training cache on 2005-07..12 ({worst_train:.2g}) and "
          f"cesm2_test_20261019 on 2006-2007 ({worst_ref:.2g})", started)

    # features and lagged columns: new rows by the recipe, the copied rows from the training cache
    new_rows = np.arange(n_old, len(c.lat))
    features = np.zeros((len(c.lat), 56), dtype=np.float32)
    features[:n_old] = train_sat["features"]
    features[new_rows] = position_features(SimpleNamespace(lat=c.lat[new_rows], lon=c.lon[new_rows],
                                                           month_index=c.month_index[new_rows]))
    lagged = np.zeros((len(c.lat), len(LEADS), 7), dtype=np.float32)
    with np.load(spec["lagged"], allow_pickle=False) as cache:
        if [int(v) for v in cache["leads"]] != list(LEADS):
            raise SystemExit(f"{spec['lagged']} has other leads")
        if not np.array_equal(cache["month_index"], c.month_index[:n_old]):
            raise SystemExit(f"month_index of the copied rows differs from {spec['lagged']}")
        lagged[:n_old] = cache["features"]
    worst = 0.
    for m in range(66, 72):
        rows = c.month(m)
        worst = max(worst, float(np.abs(H.columns(z[m], fields[m], c.lat[rows], c.lon[rows], grid_lat, grid_lon)
                                        - train_sat["features"][rows, 44:51]).max()))
    checks["features_largest_difference_2005_07_12_vs_training_cache"] = worst
    if worst > 1e-5:
        raise SystemExit(f"features of 2005-07..12 differ from the training cache by {worst:.3g}")
    for m in NEW_MONTHS:
        rows = c.month(m)
        features[rows, 44:51] = H.columns(z[m], fields[m], c.lat[rows], c.lon[rows], grid_lat, grid_lon)
        for i, lead in enumerate(LEADS):
            lagged[rows, i] = H.columns(z[m - lead], fields[m - lead], c.lat[rows], c.lon[rows], grid_lat, grid_lon)
    ref_rows = reference_rows(REFERENCE_CACHES / "satellites.npz")
    with np.load(REFERENCE_CACHES / "satellites.npz", allow_pickle=False) as cache:
        ref_features, ref_lat, ref_lon = cache["features"], cache["lat"], cache["lon"]
    with np.load(REFERENCE_CACHES / "lagged_surface.npz", allow_pickle=False) as cache:
        ref_lagged = cache["features"]
    worst_f, worst_l = 0., 0.
    for m in KEPT_MONTHS:
        rows = c.month(m)
        if not (np.array_equal(c.lat[rows], ref_lat[ref_rows[m]]) and np.array_equal(c.lon[rows], ref_lon[ref_rows[m]])):
            raise SystemExit(f"month {m}: positions differ from cesm2_test_20261019")
        worst_f = max(worst_f, float(np.abs(features[rows] - ref_features[ref_rows[m]]).max()))
        worst_l = max(worst_l, float(np.abs(lagged[rows] - ref_lagged[ref_rows[m]]).max()))
    checks["features_largest_difference_2006_2007_vs_cesm2_test_20261019"] = worst_f
    checks["lagged_largest_difference_2006_2007_vs_cesm2_test_20261019"] = worst_l
    if max(worst_f, worst_l) > 1e-5:
        raise SystemExit("the 2006-2007 features or lagged columns differ from cesm2_test_20261019")
    H.log(f"feature and lag recipes reproduce the training cache on 2005-07..12 ({worst:.2g}) and "
          f"cesm2_test_20261019 on 2006-2007 (features {worst_f:.2g}, lagged {worst_l:.2g})", started)

    all_grids = np.concatenate([train_grids, np.stack([z[m] for m in NEW_MONTHS]).astype(np.float16)])
    all_months = np.r_[train_grid_month, np.array(NEW_MONTHS)]
    if len(np.unique(all_months)) != len(all_months):
        raise SystemExit("a month has two grids")
    checks["profiles"], checks["new_rows"], checks["grid_months"] = int(len(c.lat)), int(len(new_rows)), int(len(all_months))
    common = {"cohort_fingerprint": fingerprint, "cohort": name, "training_cohort": spec["train"],
              "training_cohort_fingerprint": train_meta["cohort_fingerprint"],
              "normalization_scope": "training years only", "normalization_years": [2000, 2003],
              "kept_months_2006_2007": [KEPT_MONTHS[0], KEPT_MONTHS[-1]], "test_months": [TEST_MONTHS[0], TEST_MONTHS[-1]],
              "store": str(STORE.relative_to(ROOT)),
              "copied_from": {k: str(Path(spec[k]).resolve()) for k in ("satellites", "grids", "lagged")},
              "copied_rows": int(n_old), "checks": checks,
              "spec": "docs/superpowers/specs/2026-10-22-sealed-cesm2-test2.md",
              "script": "experiments/synthetic/167_cesm2_test2_cohorts.py",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "helper_sha256": hashlib.sha256(Path(H.__file__).read_bytes()).hexdigest()}
    output.mkdir(parents=True, exist_ok=True)
    H.save(output / "satellites.npz", features=features, month_index=c.month_index, lat=c.lat, lon=c.lon,
           surface_climatology=climate, surface_mean=mean, surface_std=std, grid_month=all_months,
           grid_lat=grid_lat, grid_lon=grid_lon, normalization_mean=train_sat["normalization_mean"],
           normalization_std=train_sat["normalization_std"],
           metadata=json.dumps({**common, "feature_fingerprint": array_fingerprint((features,))}))
    H.save(output / "surface_grids.npz", grids=all_grids, grid_month=all_months, grid_lat=grid_lat, grid_lon=grid_lon,
           metadata=json.dumps(common))
    H.save(output / "lagged_surface.npz", features=lagged, leads=np.array(LEADS), month_index=c.month_index,
           lat=c.lat, lon=c.lon, metadata=json.dumps(common))
    H.log(f"caches written to {output}", started)
    return checks


# ---------------------------------------------------------------- the OI selection
def oi_equal(new_path, old_path):
    """(equal, detail): the OI settings, per-month and summary scores of ``new_path`` against ``old_path``."""
    new, old = json.loads(Path(new_path).read_text()), json.loads(Path(old_path).read_text())
    detail, equal = {}, True
    for key in ("oi_selection", "oi_single_selection"):
        detail[key] = new[key] == old[key]
        equal &= detail[key]
    scores = {}
    for method in old["baselines"]:
        for split in old["baselines"][method]["scores"]:
            a, b = new["baselines"][method]["scores"][split], old["baselines"][method]["scores"][split]
            scores[f"{method}/{split}"] = max(abs(a[ch][k] - b[ch][k]) for ch in ("TEMP", "SALT")
                                              for k in ("rmse_physical", "J") if isinstance(a[ch][k], (int, float)))
    detail["largest_score_difference"] = max(scores.values())
    detail["scores"] = scores
    detail["per_month_equal"] = new["per_month"] == old["per_month"]
    return equal and detail["per_month_equal"] and detail["largest_score_difference"] == 0., detail


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohorts", nargs="+", choices=list(COHORTS), default=list(COHORTS))
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    if not STORE_CHECKS.exists():
        raise SystemExit(f"{STORE_CHECKS} missing: run 166 first")
    for name in args.cohorts:
        started = time.time()
        spec = COHORTS[name]
        output = OUTPUT / name
        output.mkdir(parents=True, exist_ok=True)
        checks_path = output / "checks.json"
        checks = json.loads(checks_path.read_text()) if checks_path.exists() else {}
        new_path = ROOT / f"data/synthetic_argo/{name}.nc"
        if new_path.exists():
            print(f"{new_path} exists: keeping it", flush=True)
        else:
            checks["cohort"] = build_cohort(name, spec, started)
            checks_path.write_text(json.dumps(checks, indent=1) + "\n")
        caches = [output / f for f in ("satellites.npz", "lagged_surface.npz", "surface_grids.npz")]
        if all(f.exists() for f in caches):
            print("caches exist: keeping them", flush=True)
        else:
            checks["caches"] = build_caches(name, spec, output, args.workers, started)
            checks_path.write_text(json.dumps(checks, indent=1) + "\n")
        oi_summary = ROOT / f"outputs/audit/synthetic/fixed_baselines_{name}/summary.json"
        if not oi_summary.exists():
            run("experiments/synthetic/44_synth_argo_oi.py", ["--cohort", name, "--workers", str(args.workers)],
                output / "oi_baseline.log")
            H.log(f"OI baseline -> {oi_summary}", started)
        equal, detail = oi_equal(oi_summary, spec["oi"])
        checks["oi_selection_identical_to_training_cohort"] = equal
        checks["oi_detail"] = {"training_summary": str(spec["oi"].relative_to(ROOT)), **detail}
        checks_path.write_text(json.dumps(checks, indent=1) + "\n")
        print(json.dumps(checks, indent=1), flush=True)
        if not equal:
            raise SystemExit(f"the OI selection of {name} differs from {spec['train']}'s")
        H.log(f"{name} done", started)


if __name__ == "__main__":
    main()
