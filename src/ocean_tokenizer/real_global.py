"""The real global Argo cohort as the gridded model reads it.

``global_global.nc`` with WOA23 subtracted at each profile's own position (the "exact" anomaly
of ``audit_tools.load_cohort``), cut to 2016-2023, the years of the 1-degree satellite store
``data/real_obs_1deg.zarr`` (``years`` extends the cut: 2024 for the sealed test, 2011 for the
10-year training window, whose statistics stay the 2016-2020 ones). 2025 is the sealed holdout of
the Real-Data v1 registration and is never read. Splits by year as the original real-data ``latent_experiment.SPLITS``; floats keep
the cohort's stored ``float_split``. A value more than ``QC_Z`` training standard deviations from
the training mean is treated as missing, as ``latent_experiment.OceanScenes`` does.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .argo_obs import ArgoCohort, ArgoNorm
from .audit_tools import load_cohort
from .latent_experiment import array_fingerprint
from .point_baselines import CH

REAL_SPLITS = {"train": (2016, 2020), "validation": (2021, 2021), "development": (2022, 2023)}
YEARS = (2016, 2023)
QC_Z = 10.
#: per-profile arrays of an ``ArgoCohort``, all cut together
PROFILE_FIELDS = ("month_index", "grid_y", "grid_x", "lat", "lon", "wmo", "year", "year_split",
                  "float_split", "TEMP", "SALT", "TEMP_ERR", "SALT_ERR", "ECCO_TEMP", "ECCO_SALT")


def subset_cohort(c: ArgoCohort, keep: np.ndarray) -> ArgoCohort:
    """The cohort restricted to the profiles where ``keep``, in their order, months re-indexed."""
    keep = np.asarray(keep, bool)
    fields = {k: (getattr(c, k)[keep] if getattr(c, k) is not None else None) for k in PROFILE_FIELDS}
    s = replace(c, **fields, _by_month={})
    if np.any(np.diff(s.month_index) < 0):
        raise ValueError("the cohort must be sorted by month")
    uniq, start = np.unique(s.month_index, return_index=True)
    stop = np.r_[start[1:], s.month_index.size]
    s._by_month = {int(m): (int(a), int(b)) for m, a, b in zip(uniq, start, stop)}
    return s


def quality_controlled(z: np.ndarray, limit: float = QC_Z) -> np.ndarray:
    """``z`` with every value beyond ``limit`` standard deviations set to NaN."""
    z = np.array(z, dtype=float, copy=True)
    with np.errstate(invalid="ignore"):
        z[np.abs(z) > limit] = np.nan
    return z


def training_splits(splits, train_split):
    """``splits`` with its training years replaced by ``train_split`` (a longer window, e.g.
    2011-2020): the other splits are kept and must not overlap it."""
    lo, hi = (int(y) for y in train_split)
    if lo > hi:
        raise ValueError(f"empty training split {train_split}")
    for name, (a, b) in splits.items():
        if name != "train" and not (hi < a or lo > b):
            raise ValueError(f"the training split {train_split} overlaps the {name} split {(a, b)}")
    return {**{k: tuple(v) for k, v in splits.items()}, "train": (lo, hi)}


def load_real(root, *, splits=REAL_SPLITS, years=YEARS, train_split=None):
    """``(cohort, norm, obs)`` like ``synth_argo_eval.load``: the 2016-2023 at-position anomaly
    cohort, its training-year normalisation and ``{channel: (P, L)}`` quality-controlled z-scores.
    The cohort's TEMP/SALT hold the same anomalies with the QC'd values set to NaN, so every
    method sees the same missing values.

    ``train_split`` (e.g. ``(2011, 2020)``) relabels the training years AFTER the normalisation
    and the quality control were fitted on ``splits``' training years: the per-level statistics,
    the targets and the validation cells stay those of ``splits``, only the months a model trains
    on change (``years`` must cover the window)."""
    c, _ = load_cohort(str(root), "global", splits, anomaly="exact")
    c = subset_cohort(c, (c.year >= years[0]) & (c.year <= years[1]))
    norm = ArgoNorm.fit(c, "train")
    obs = {ch: quality_controlled(norm.z(ch, getattr(c, ch))) for ch in CH}
    for ch in CH:
        getattr(c, ch)[np.isnan(obs[ch])] = np.nan
    if train_split is not None:
        table = training_splits(splits, train_split)
        if table["train"][0] < years[0] or table["train"][1] > years[1]:
            raise ValueError(f"the training split {train_split} is outside the loaded years {years}")
        c.apply_splits(table)
    return c, norm, obs


def cohort_fingerprint(c: ArgoCohort, obs) -> str:
    """The synthetic cohorts' fingerprint recipe: positions, months, floats, levels, z-scores."""
    return array_fingerprint((c.month_index, c.lat, c.lon, c.wmo, c.levels,
                              np.stack([obs[ch] for ch in CH], axis=-1)))


def surface_z(fields: np.ndarray, months: np.ndarray, train: np.ndarray):
    """``(z, climate, mean, spread)`` of monthly surface grids ``[M, F, H, W]``, the recipe of
    ``88_cesm2_long_surface.py``: the anomaly against the training months' climatology of the
    same calendar month, minus the training anomalies' mean, divided by their spread."""
    import warnings
    months, train = np.asarray(months), np.asarray(train, bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # land cells are all-NaN
        climate = np.stack([np.nanmean(fields[train & (months % 12 == m)], axis=0) for m in range(12)])
        anomalies = fields - climate[months % 12]
        mean = np.nanmean(anomalies[train], axis=(0, 2, 3))
        spread = np.nanstd(anomalies[train], axis=(0, 2, 3))
    spread = np.where(np.isfinite(spread) & (spread > 1e-6), spread, 1.)
    return (anomalies - mean[None, :, None, None]) / spread[None, :, None, None], climate, mean, spread


def _sample(field, lat, lon, grid_lat, grid_lon):
    """Periodic-longitude bilinear sampling of ``field [..., H, W]`` (``50``'s ``sample_grid``):
    ``[P, ...]``; any missing corner leaves the sample missing."""
    fy = (np.asarray(lat) - grid_lat[0]) / (grid_lat[1] - grid_lat[0])
    fx = ((np.asarray(lon) - grid_lon[0]) % 360) / (grid_lon[1] - grid_lon[0])
    i0 = np.clip(np.floor(fy).astype(int), 0, len(grid_lat) - 1)
    j0 = np.floor(fx).astype(int) % len(grid_lon)
    i1, j1 = np.minimum(i0 + 1, len(grid_lat) - 1), (j0 + 1) % len(grid_lon)
    wy, wx = fy - np.floor(fy), fx - np.floor(fx)
    return (field[..., i0, j0] * ((1 - wy) * (1 - wx)) + field[..., i0, j1] * ((1 - wy) * wx)
            + field[..., i1, j0] * (wy * (1 - wx)) + field[..., i1, j1] * (wy * wx)).T


def profile_surface_columns(z, absolute_sst, grid_months, grid_lat, grid_lon, lat, lon, month, lag=0):
    """Feature columns 44-50 of every profile from the grids of month ``month - lag``: SST, SSS
    and SSH z-scores with validity flags, then ``(SST - 14) / 11``. A month the grids do not
    cover gives zeros (all flags off)."""
    out = np.zeros((len(lat), 7))
    where = {int(m): i for i, m in enumerate(np.asarray(grid_months))}
    source = np.asarray(month) - lag
    for m in np.unique(source):
        if int(m) not in where:
            continue
        rows = np.flatnonzero(source == m)
        i = where[int(m)]
        sampled = _sample(z[i], lat[rows], lon[rows], grid_lat, grid_lon)          # [R, 3]
        for k in range(3):
            out[rows, 2 * k] = np.nan_to_num(sampled[:, k])
            out[rows, 2 * k + 1] = np.isfinite(sampled[:, k])
        sst = _sample(absolute_sst[i], lat[rows], lon[rows], grid_lat, grid_lon)
        out[rows, 6] = np.nan_to_num((sst - 14.) / 11.)
    return out


EVAL_CELLS = 8000          # validation cells a month, as the synthetic protocol
EVAL_SEED = 20260918       # the synthetic protocol's seed of the capped draw, [EVAL_SEED, month]


def real_eval_month(c: ArgoCohort, obs, m: int, max_cells: int = 0, lead: int = 0):
    """Scored cells of target month ``m``: every held-out profile at every level (or a draw of
    ``max_cells`` seeded by ``m``, so every lead and every method scores the same cells), and the
    inputs: every input-float profile of month ``m - lead``. Counts vary by month."""
    src = c.month(m - lead, float_split="cohort_float")
    tgt = c.month(m, float_split="heldout_float")
    rows, levels = tgt.size, c.levels.size
    prof, lev = np.repeat(np.arange(rows), levels), np.tile(np.arange(levels), rows)
    if max_cells and rows * levels > max_cells:
        pick = np.random.default_rng([EVAL_SEED, int(m)]).choice(rows * levels, max_cells, replace=False)
        prof, lev = prof[pick], lev[pick]
    if np.intersect1d(c.wmo[src], c.wmo[tgt]).size:
        raise ValueError(f"month {m}: input and held-out floats overlap")
    return dict(month=int(m), lead=int(lead), src=src, tgt=tgt, prof=prof, lev=lev,
                target={ch: obs[ch][tgt][prof, lev] for ch in CH})


def raw_profiles(root, years=YEARS):
    """Measured (absolute) TEMP/SALT and the day of every profile of ``load_real``'s cohort, in its
    row order (the file sorted stably by month, then cut to ``years``): ``(temp, salt, day)``,
    ``day`` counted from 2000-01-01."""
    import xarray as xr
    from .audit_tools import cohort_path
    with xr.open_dataset(cohort_path(str(root), "global")) as raw:
        order = np.argsort(np.asarray(raw.month_index.values, int), kind="stable")
        year = np.asarray(raw.year.values)[order]
        keep = order[(year >= years[0]) & (year <= years[1])]
        temp, salt = np.asarray(raw.TEMP.values)[keep], np.asarray(raw.SALT.values)[keep]
        day = (np.asarray(raw.time.values)[keep].astype("datetime64[D]") - np.datetime64("2000-01-01")).astype(float)
    return temp.astype(float), salt.astype(float), day
