"""Shared pieces of ``152_cesm2_aug3_cohort.py`` and ``153_cesm2_vlong_cohort.py``.

Both scripts grow ``cesm2_long_uniform`` (``87_cesm2_long_cohort.py``) into a larger synthetic
cohort and build the caches ``150_wang_campaign.py`` waits for:

  outputs/cohort_<name>/satellites.npz       88's per-profile features (56 columns)
  outputs/cohort_<name>/lagged_surface.npz   93's lagged surface columns, leads 1/3/6
  outputs/cohort_<name>/surface_grids.npz    100's dense z-scored grids of every cohort month

The surface recipe is 88's (``month_fields``, ``sample_grid``), the normalisation is the 2000-2003
one stored in ``outputs/long_record_20261017/satellites.npz`` (read, never refitted), and every row
copied from ``cesm2_long_uniform`` keeps its existing cache values after the recipe has been shown
to reproduce them. The functions here are imported by both scripts rather than copied.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import warnings

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import xarray as xr

from ocean_tokenizer import synth_argo_eval as S
from ocean_tokenizer.latent_experiment import array_fingerprint
from ocean_tokenizer.synthetic_latent_experiment import position_features


def module(name, path):
    """A numbered experiment script as a module, registered so worker processes can unpickle it."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


COHORT_87 = module("long_cohort", "experiments/synthetic/87_cesm2_long_cohort.py")
SURFACE_88 = module("long_surface", "experiments/synthetic/88_cesm2_long_surface.py")

OLD = ROOT / "data/synthetic_argo/cesm2_long_uniform.nc"
OLD_STORE = ROOT / "data/cesm2_long/cesm2_le_cohort_levels.zarr"
OLD_SATELLITES = ROOT / "outputs/long_record_20261017/satellites.npz"
OLD_GRIDS = ROOT / "outputs/grid_ka_20261019/surface_grids.npz"
OLD_LAGGED = ROOT / "outputs/forecast_20261019/lagged_surface.npz"
OLD_OI = ROOT / "outputs/audit/synthetic/fixed_baselines_cesm2_long_uniform/summary.json"
OLD_FIRST_MONTH = -432                      # 1964-01
LEADS = (1, 3, 6)
SEED = COHORT_87.SEED
N_PER_MONTH = COHORT_87.N_PER_MONTH
ARRAYS = COHORT_87.ARRAYS
COLUMNS = ("lat", "lon", "month_index", "grid_y", "grid_x", "wmo", "float_split")
ALL_VARS = ("lat", "lon", "month_index", "grid_y", "grid_x", "year", "wmo", "year_split", "float_split",
            *ARRAYS, "TEMP_ERR", "SALT_ERR")


def log(message, started):
    print(f"{message} ({time.time() - started:.0f}s)", flush=True)


# ---------------------------------------------------------------- the level store and 87's climatology
class Store:
    """A 20-level store of ``53_cesm2_long_levels.py`` with 87's month reader and climatology."""

    def __init__(self, path):
        self.path = Path(path)
        self.ds = xr.open_zarr(self.path, chunks=None)
        self.months = np.asarray(self.ds.month_index.values, dtype=int)
        self.lat_c = self.ds.lat.values.astype(float)
        self.lon_c = self.ds.lon.values.astype(float)
        self.levels = np.asarray(self.ds.level.values)
        self.position = {int(m): i for i, m in enumerate(self.months)}
        self.clim = None
        self.wet0 = None

    def month(self, t):
        """87's ``month``: (raw, cleaned) TEMP/SALT of month ``t`` with the regridding fill removed."""
        if t not in self.position:
            raise SystemExit(f"month {t} not in {self.path.name}")
        i = self.position[t]
        raw = {v: np.asarray(self.ds[v][i].values, dtype="float32") for v in ("TEMP", "SALT")}
        fill = (raw["TEMP"] == 0) & (raw["SALT"] == 0)
        return raw, {v: np.where(fill, np.nan, a) for v, a in raw.items()}

    def fit_climatology(self):
        """87's 2000-2003 climatology (float32 nan-mean by calendar month) and its 5 m wet mask."""
        train = [t for t in range(72) if COHORT_87.TRAIN[0] <= 2000 + t // 12 <= COHORT_87.TRAIN[1]]
        clean_train = {v: np.stack([self.month(t)[1][v] for t in train]) for v in ("TEMP", "SALT")}
        cal_train = np.array(train) % 12
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN land columns
            self.clim = {v: np.stack([np.nanmean(a[cal_train == m], axis=0) for m in range(12)]).astype("float32")
                         for v, a in clean_train.items()}
        self.wet0 = np.isfinite(self.month(0)[1]["TEMP"][0])
        return self.clim

    def check_climatology(self, old, n=20_000, seed=1):
        """110's check: the climatology and stencil reproduce CLIM_POS of ``old`` (an open dataset)."""
        check = np.sort(np.random.default_rng(seed).choice(old.sizes["profile"], n, replace=False))
        lat_k, lon_k = old.lat.values[check], old.lon.values[check]
        cal = np.asarray(old.month_index.values)[check] % 12
        for v in ("TEMP", "SALT"):
            rebuilt = np.full((check.size, old.level.size), np.nan, "float32")
            for m in range(12):
                sel = cal == m
                idx, w = COHORT_87.stencil(lat_k[sel], lon_k[sel], self.lat_c, self.lon_c)
                rebuilt[sel] = COHORT_87.interp(self.clim[v][m], idx, w)
            stored = np.asarray(old[f"CLIM_POS_{v}"].values)[check]
            if not np.array_equal(rebuilt, stored, equal_nan=True):
                raise SystemExit(f"the climatology does not reproduce CLIM_POS_{v} of {OLD.name}: largest "
                                 f"difference {np.nanmax(np.abs(rebuilt - stored)):.3g}")

    def draw(self, t, generator_seed, wmo, month=None):
        """87's ``draw_month`` from the generator ``default_rng(generator_seed)``, float ids ``wmo(i)``.

        Mirrors ``87_cesm2_long_cohort.draw_month`` line for line (the same random stream, stencil,
        query choice and dtypes); only the generator's seed and the id scheme are parameters. 152
        checks against the stored rows that the default seed reproduces 87's rows exactly.
        ``month`` is ``self.month(t)`` when the caller already holds it.
        """
        raw, clean = self.month(t) if month is None else month
        rng = np.random.default_rng(generator_seed)
        lat_c, lon_c, wet0, clim = self.lat_c, self.lon_c, self.wet0, self.clim
        lat, lon = np.empty(0), np.empty(0)
        while lat.size < N_PER_MONTH:
            la = np.rad2deg(np.arcsin(rng.uniform(-1, 1, 4 * N_PER_MONTH)))
            lo = rng.uniform(0.0, 360.0, la.size)
            ok = np.abs(la) <= lat_c[-1]
            la, lo = la[ok], lo[ok]
            idx, _ = COHORT_87.stencil(la, lo, lat_c, lon_c)
            wet = np.logical_and.reduce([wet0[i, j] for i, j in idx])
            lat, lon = np.r_[lat, la[wet]], np.r_[lon, lo[wet]]
        lat, lon = lat[:N_PER_MONTH], lon[:N_PER_MONTH]
        idx, w = COHORT_87.stencil(lat, lon, lat_c, lon_c)
        iy = np.floor(lat + 90.0).astype(int)
        ix = np.floor(lon).astype(int) % lon_c.size
        cal = t % 12
        arrays = {}
        for v in ("TEMP", "SALT"):
            arrays[v] = COHORT_87.interp(clean[v], idx, w)
            arrays[f"{v}_RAW"] = COHORT_87.interp(raw[v], idx, w)
            arrays[f"CLIM_POS_{v}"] = COHORT_87.interp(clim[v][cal], idx, w)
            cc = clim[v][cal][:, iy, ix].T
            arrays[f"CLIM_CELL_{v}"] = np.where(np.isfinite(arrays[v]), cc, np.nan)
        query = np.zeros(N_PER_MONTH, bool)
        query[rng.choice(N_PER_MONTH, int(round(COHORT_87.QUERY_FRAC * N_PER_MONTH)), replace=False)] = True
        columns = {"lat": lat, "lon": lon, "month_index": np.full(N_PER_MONTH, t),
                   "grid_y": iy, "grid_x": ix,
                   "wmo": np.array([wmo(i) for i in range(N_PER_MONTH)]),
                   "float_split": np.where(query, "heldout_float", "cohort_float")}
        return columns, {k: a.astype("float32") for k, a in arrays.items()}

    def draw_87(self, t):
        """87's own ``draw_month`` on this store (its generator ``[SEED, t + 10000]`` and "L" ids)."""
        raw, clean = self.month(t)
        return COHORT_87.draw_month(t, raw, clean, self.clim, self.wet0, self.lat_c, self.lon_c)

    def close(self):
        self.ds.close()


def rows_equal(columns, arrays, old, rows):
    """Whether a drawn month equals the rows ``rows`` of ``old`` ({variable: array}) value for value."""
    for k in COLUMNS:
        if not np.array_equal(old[k][rows], columns[k]):
            return False
    return all(np.array_equal(old[k][rows], arrays[k], equal_nan=True) for k in ARRAYS)


def load_rows(old):
    """Every variable of the open dataset ``old`` in memory, strings as str."""
    out = {}
    for k in ALL_VARS:
        values = np.asarray(old[k].values)
        out[k] = values.astype(str) if values.dtype.kind in "OSU" else values
    return out


def month_rows(old):
    """{month: slice} of an open dataset whose rows are sorted by month (87's layout)."""
    months = np.asarray(old.month_index.values, dtype=int)
    if (np.diff(months) < 0).any():
        raise SystemExit(f"{OLD.name} rows are not sorted by month")
    uniq, start = np.unique(months, return_index=True)
    stop = np.r_[start[1:], months.size]
    return {int(m): slice(int(a), int(b)) for m, a, b in zip(uniq, start, stop)}


def write_cohort(path, data_vars, levels, attrs):
    """An xarray dataset in 87's layout, written atomically."""
    out = xr.Dataset(data_vars, coords={"level": levels}, attrs=attrs)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    out.to_netcdf(temporary)
    os.replace(temporary, path)
    return out.sizes["profile"]


# ---------------------------------------------------------------- 88's surface fields of a store
def store_fields(task):
    """88's surface fields of one store position; the task names the store (worker processes)."""
    position, levels, store = task
    SURFACE_88.STORE = Path(store)
    return SURFACE_88.month_fields((position, levels))


def surface_fields(store_path, months, levels, workers, started):
    """{month: (3, NY, NX)} SST/SSS/steric SSH of every month, 88's recipe on ``store_path``."""
    with xr.open_zarr(store_path, chunks=None) as ds:
        position = {int(m): i for i, m in enumerate(np.asarray(ds.month_index.values, dtype=int))}
    missing = [m for m in months if int(m) not in position]
    if missing:
        raise SystemExit(f"{Path(store_path).name} lacks months {missing[:5]}...")
    where = {position[int(m)]: int(m) for m in months}
    fields = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("fork")) as pool:
        tasks = ((position[int(m)], np.asarray(levels), str(store_path)) for m in months)
        for count, (i, f) in enumerate(pool.map(store_fields, tasks, chunksize=4), 1):
            fields[where[i]] = f
            if count % 240 == 0:
                log(f"surface fields {count}/{len(months)} months", started)
    return fields


def columns(z, absolute, lat, lon, grid_lat, grid_lon):
    """110's columns 44-50 from one month's z-scored fields and absolute SST."""
    sampled = SURFACE_88.sample_grid(z, lat, lon, grid_lat, grid_lon)
    sst = SURFACE_88.sample_grid(absolute[:1], lat, lon, grid_lat, grid_lon)
    out = np.zeros((len(lat), 7), dtype=np.float32)
    for k in range(3):
        out[:, 2 * k] = np.nan_to_num(sampled[:, k])
        out[:, 2 * k + 1] = np.isfinite(sampled[:, k])
    out[:, 6] = np.nan_to_num((sst[:, 0] - 14.) / 11.)
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, **arrays):
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def build_caches(name, store_path, old_at, output, script, workers, started, extra_metadata=None):
    """The three caches of cohort ``name`` under ``output`` and their checks.

    ``old_at`` gives, for each row of ``cesm2_long_uniform`` in its loaded (month-sorted) order, that
    row's position in the new cohort's loaded order. Surface fields come from ``store_path`` for every
    cohort month; the normalisation is the one stored in the existing satellite cache. Returns the
    check results.
    """
    checks = {}
    c, norm, obs = S.load(str(ROOT), cohort=name)
    fingerprint = array_fingerprint((c.month_index, c.lat, c.lon, c.wmo, c.levels,
                                     np.stack([obs[ch] for ch in ("TEMP", "SALT")], -1)))
    log(f"cohort {name} loaded: {len(c.lat):,} profiles", started)
    with np.load(OLD_SATELLITES, allow_pickle=False) as cache:
        old_sat = {k: cache[k] for k in cache.files}
    climate, mean, std = old_sat["surface_climatology"], old_sat["surface_mean"], old_sat["surface_std"]
    grid_lat, grid_lon = old_sat["grid_lat"], old_sat["grid_lon"]
    old_c = xr.open_dataset(OLD)
    old_order = np.argsort(np.asarray(old_c.month_index.values, int), kind="stable")
    for k in ("lat", "lon", "month_index"):
        if not np.array_equal(np.asarray(old_c[k].values)[old_order], getattr(c, k)[old_at]):
            raise SystemExit(f"{k} of the copied rows differs from {OLD.name}")
    if not np.array_equal(np.asarray(old_c.wmo.values).astype(str)[old_order], c.wmo[old_at]):
        raise SystemExit(f"wmo of the copied rows differs from {OLD.name}")
    for k in ("lat", "lon", "month_index"):
        if not np.array_equal(old_sat[k], getattr(c, k)[old_at]):
            raise SystemExit(f"{k} of the copied rows differs from {OLD_SATELLITES.name}")
    old_c.close()
    if len(np.unique(c.wmo)) != len(c.wmo):
        raise SystemExit("float ids are not unique")
    new_norm = {"mean": np.stack([norm.mean[ch] for ch in ("TEMP", "SALT")], -1),
                "std": np.stack([norm.std[ch] for ch in ("TEMP", "SALT")], -1)}
    for k in ("mean", "std"):
        diff = float(np.abs(new_norm[k] - old_sat[f"normalization_{k}"]).max())
        checks[f"normalization_{k}_largest_difference"] = diff
        if diff > 1e-9:
            raise SystemExit(f"the 2000-2003 normalisation {k} differs from {OLD_SATELLITES.name} by {diff:.3g}")
    log("copied rows and the 2000-2003 normalisation match the existing caches", started)

    months = [int(m) for m in np.unique(c.month_index)]
    fields = surface_fields(store_path, months, c.levels, workers, started)
    z = {m: (f - climate[m % 12] - mean[:, None, None]) / std[:, None, None] for m, f in fields.items()}
    log(f"surface fields and z-scores of {len(months)} months", started)

    # the grids of the old months reproduce 100's cache (float16 storage)
    with np.load(OLD_GRIDS, allow_pickle=False) as cache:
        old_grids, old_grid_month = cache["grids"], cache["grid_month"]
    worst = 0.
    for i, m in enumerate(old_grid_month):
        if int(m) not in z:
            continue
        stored = old_grids[i].astype(float)
        both = np.isfinite(stored) & np.isfinite(z[int(m)])
        if not np.array_equal(np.isfinite(stored), np.isfinite(z[int(m)])):
            raise SystemExit(f"grid validity of month {m} differs from {OLD_GRIDS.name}")
        worst = max(worst, float((np.abs(stored - z[int(m)]) / np.maximum(1., np.abs(z[int(m)])))[both].max()))
    checks["grids_largest_relative_difference_old_months"] = worst
    checks["grids_old_months_compared"] = int(len(old_grid_month))
    if worst > 1e-3:
        raise SystemExit(f"grids of the old months differ from {OLD_GRIDS.name} by up to {worst:.3g}")
    log(f"grids of {len(old_grid_month)} old months reproduce {OLD_GRIDS.name} (largest relative difference {worst:.2g})", started)

    # 88's features for every row, then the copied rows' existing values
    features = position_features(SimpleNamespace(lat=c.lat, lon=c.lon, month_index=c.month_index)).astype(np.float32)
    lagged = np.zeros((len(c.lat), len(LEADS), 7), dtype=np.float32)
    for m in months:
        rows = c.month(m)
        features[rows, 44:51] = columns(z[m], fields[m], c.lat[rows], c.lon[rows], grid_lat, grid_lon)
        for i, lead in enumerate(LEADS):
            if m - lead in z:
                lagged[rows, i] = columns(z[m - lead], fields[m - lead], c.lat[rows], c.lon[rows], grid_lat, grid_lon)
    worst = float(np.abs(features[old_at] - old_sat["features"]).max())
    checks["features_largest_difference_copied_rows"] = worst
    if worst > 1e-5:
        raise SystemExit(f"features of the copied rows differ from {OLD_SATELLITES.name} by up to {worst:.3g}")
    log(f"features of {len(old_at):,} copied rows reproduce {OLD_SATELLITES.name} (largest difference {worst:.2g})", started)
    with np.load(OLD_LAGGED, allow_pickle=False) as cache:
        if [int(v) for v in cache["leads"]] != list(LEADS):
            raise SystemExit(f"{OLD_LAGGED.name} has other leads")
        old_lagged = cache["features"]
        if not np.array_equal(cache["month_index"], c.month_index[old_at]):
            raise SystemExit(f"month_index of the copied rows differs from {OLD_LAGGED.name}")
    old_month = c.month_index[old_at]
    worst, filled = 0., {}
    for i, lead in enumerate(LEADS):
        had = old_month - lead >= OLD_FIRST_MONTH          # rows whose earlier month the old cohort held
        worst = max(worst, float(np.abs(lagged[old_at[had], i] - old_lagged[had, i]).max()))
        if (old_lagged[~had, i] != 0).any():
            raise SystemExit(f"{OLD_LAGGED.name} is not zero before 1964-01 at lead {lead}")
        filled[lead] = int((lagged[old_at[~had], i, 1] > 0).sum())
    checks["lagged_largest_difference_copied_rows"] = worst
    checks["lagged_copied_rows_newly_filled"] = filled
    if worst > 1e-6:
        raise SystemExit(f"lagged columns of the copied rows differ from {OLD_LAGGED.name} by up to {worst:.3g}")
    log(f"lagged columns of the copied rows reproduce {OLD_LAGGED.name} (largest difference {worst:.2g}; "
        f"rows newly filled before 1964: {filled})", started)
    # the copied rows keep their existing cache values exactly
    features[old_at] = old_sat["features"]
    for i, lead in enumerate(LEADS):
        had = old_month - lead >= OLD_FIRST_MONTH
        lagged[old_at[had], i] = old_lagged[had, i]
    missing = {lead: int((lagged[:, i, 1] == 0).sum()) for i, lead in enumerate(LEADS)}

    grid_month = np.array(months)
    grids = np.stack([z[m] for m in months]).astype(np.float16)
    common = {"cohort_fingerprint": fingerprint, "cohort": name, "normalization_scope": "training years only",
              "normalization_years": [2000, 2003], "surface_variables": ["SST", "SSS", "steric_SSH"],
              "surface_feature_offsets": [44, 46, 48], "months": [months[0], months[-1]],
              "store": str(Path(store_path).relative_to(ROOT)),
              "copied_from": {"satellites": str(OLD_SATELLITES.relative_to(ROOT)),
                              "grids": str(OLD_GRIDS.relative_to(ROOT)), "lagged": str(OLD_LAGGED.relative_to(ROOT))},
              "copied_rows": int(len(old_at)), "checks": checks, **(extra_metadata or {}),
              "script": str(Path(script).relative_to(ROOT)),
              "source_sha256": hashlib.sha256(Path(script).read_bytes()).hexdigest(),
              "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.mkdir(parents=True, exist_ok=True)
    save(output / "satellites.npz", features=features, month_index=c.month_index, lat=c.lat, lon=c.lon,
         surface_climatology=climate, surface_mean=mean, surface_std=std, grid_month=grid_month,
         grid_lat=grid_lat, grid_lon=grid_lon, normalization_mean=old_sat["normalization_mean"],
         normalization_std=old_sat["normalization_std"],
         metadata=json.dumps({**common, "feature_fingerprint": array_fingerprint((features,))}))
    save(output / "surface_grids.npz", grids=grids, grid_month=grid_month, grid_lat=grid_lat, grid_lon=grid_lon,
         metadata=json.dumps({**common, "units": "z-score of the anomaly against the 2000-2003 monthly climatology (88's constants)"}))
    save(output / "lagged_surface.npz", features=lagged, leads=np.array(LEADS), month_index=c.month_index,
         lat=c.lat, lon=c.lon, metadata=json.dumps({**common, "leads_months": list(LEADS),
                                                    "columns": ["SST_z", "SST_valid", "SSS_z", "SSS_valid", "SSH_z",
                                                                "SSH_valid", "SST_absolute_scaled"],
                                                    "replaces_feature_columns": [44, 51],
                                                    "profiles_without_earlier_month": missing}))
    checks["lagged_profiles_without_earlier_month"] = missing
    checks["profiles"] = int(len(c.lat))
    checks["months"] = len(months)
    log(f"caches written to {output}", started)
    return checks


def oi_selection_equal(summary_path):
    """(equal, detail): the OI settings and validation errors of ``summary_path`` against the old cohort's."""
    new, old = json.loads(Path(summary_path).read_text()), json.loads(OLD_OI.read_text())
    detail = {}
    equal = True
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
