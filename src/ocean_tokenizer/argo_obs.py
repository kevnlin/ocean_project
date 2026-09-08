"""Real Argo observations as model input, and held-out floats as the target.

This is the module that makes "run it on real data" mean what the cross-check
plan says it means.  The GODAS line samples a random **column of a reanalysis**
and scores against the same reanalysis; that is an OSSE wearing a real dataset's
clothes, and it is circular twice over — GODAS has already assimilated the very
Argo profiles a reconstruction is supposed to predict.

Here instead:

    input   real QC'd Argo profiles from the ``cohort_float`` WMOs, in the
            source month
    target  real QC'd Argo profiles from the ``heldout_float`` WMOs, at
            ``t_src + lead``
    score   RMSE at genuine measurement locations, clustered by the held-out
            float's WMO

No held-out float ever appears in the input, in any month of its life
(`44_build_argo_cohort.py` splits by WMO, not by profile), so the model cannot
have seen that instrument, its drift, or its trajectory.  That is what licenses
WMO as the inference unit the plan's P7 asks for.

Three things the real data supplies that the synthetic column could not
----------------------------------------------------------------------
**Provenance is the WMO.**  ``godas_obs`` numbered profile columns 0..23 and
used that as the provenance group — a synthetic label with no physical content.
A float is a real platform: its cycles share a sensor, a calibration history and
a drift, which is exactly the correlated-noise structure Phase 1b's
``whiten_provenance`` was built for.  Two profiles from one float genuinely are
correlated observations, and now they are labelled as such.

**Noise density is measured, not assumed.**  ``lambda_i = noise_density_i *
|support_i|`` has been driven by the pilot constant ``NOISE_DENSITY_POINT =
0.08`` since the beginning, and Phase 0 could only calibrate it by a sweep.
Argo delayed-mode files ship ``TEMP_ADJUSTED_ERROR`` / ``PSAL_ADJUSTED_ERROR``
per level.  When present (91% of profiles here) the error variance is the
reported one; the fallback is recorded per token in ``noise_is_measured`` so a
result can never quietly rest on the constant.

**Coverage is what the ocean actually gave us.**  Profiles per month vary by an
order of magnitude across 2000-2025 as the array was deployed.  That is a
property of the observing system, and the sparsity stress of P5 becomes a
question about real coverage rather than about a thinning knob.

Sampling and comparability
--------------------------
``n_profiles`` subsamples the month's real floats to keep the token budget
comparable with the registered GODAS rows (24 columns x 16 levels = 384 profile
tokens).  Subsampling is **by float, then by profile**, so a float contributing
40 cycles in a month cannot crowd out every other platform — that would silently
turn a spatial sample into a single-instrument sample.  ``n_profiles=0`` takes
everything.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch
import xarray as xr

from .batched_dfs import (NOISE_AREA_POINT_KM2, NOISE_AREA_PATCH_KM2,
                          profile_support_area_km2, patch_support_area_km2,
                          BOX_DEPTH_M, GODAS_LEVELS_M, level_thickness_m)
from .godas_obs import (MOD_PROFILE, MOD_SURF, MOD_SSH, N_MODALITIES,
                        N_CHANNELS, VARIABLE_GROUP, _patch_boxes)

CHANNELS = ("TEMP", "SALT")


# --------------------------------------------------------------------------
@dataclass
class ArgoCohort:
    """The QC'd cohort for one region, indexed for month-by-month access."""
    region: str
    levels: np.ndarray
    month_index: np.ndarray          # (P,) months since 2000-01
    grid_y: np.ndarray
    grid_x: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    wmo: np.ndarray                  # (P,) str — the provenance group AND the
    #                                  cluster unit for every CI in the plan
    year: np.ndarray
    year_split: np.ndarray
    float_split: np.ndarray
    TEMP: np.ndarray                 # (P, L)
    SALT: np.ndarray
    TEMP_ERR: np.ndarray
    SALT_ERR: np.ndarray
    grid: tuple[int, int] = (38, 26)
    _by_month: dict = field(default_factory=dict, repr=False)

    @staticmethod
    def load(path: str) -> "ArgoCohort":
        d = xr.open_dataset(path)
        c = ArgoCohort(
            region=str(d.attrs.get("region", "")),
            levels=np.asarray(d["level"].values, float),
            month_index=np.asarray(d["month_index"].values, int),
            grid_y=np.asarray(d["grid_y"].values, int),
            grid_x=np.asarray(d["grid_x"].values, int),
            lat=np.asarray(d["lat"].values, float),
            lon=np.asarray(d["lon"].values, float),
            wmo=np.asarray(d["wmo"].values).astype(str),
            year=np.asarray(d["year"].values, int),
            year_split=np.asarray(d["year_split"].values).astype(str),
            float_split=np.asarray(d["float_split"].values).astype(str),
            TEMP=np.asarray(d["TEMP"].values, float),
            SALT=np.asarray(d["SALT"].values, float),
            TEMP_ERR=np.asarray(d["TEMP_ERR"].values, float),
            SALT_ERR=np.asarray(d["SALT_ERR"].values, float),
            grid=tuple(int(x) for x in d.attrs.get("grid", (38, 26))))
        d.close()
        order = np.argsort(c.month_index, kind="stable")
        for k in ("month_index", "grid_y", "grid_x", "lat", "lon", "wmo",
                  "year", "year_split", "float_split", "TEMP", "SALT",
                  "TEMP_ERR", "SALT_ERR"):
            setattr(c, k, getattr(c, k)[order])
        uniq, start = np.unique(c.month_index, return_index=True)
        stop = np.r_[start[1:], c.month_index.size]
        c._by_month = {int(m): (int(a), int(b))
                       for m, a, b in zip(uniq, start, stop)}
        return c

    def month(self, m: int, float_split: str | None = None,
              year_split: str | None = None) -> np.ndarray:
        """Row indices for month ``m``, optionally restricted to a split."""
        a, b = self._by_month.get(int(m), (0, 0))
        idx = np.arange(a, b)
        if float_split is not None:
            idx = idx[self.float_split[idx] == float_split]
        if year_split is not None:
            idx = idx[self.year_split[idx] == year_split]
        return idx

    def apply_splits(self, table: dict) -> "ArgoCohort":
        """Recompute ``year_split`` in place from a named split table.

        The cohort file stores the main protocol's labels, but the split is a
        pure function of the profile's year, so a secondary protocol needs no
        rebuild -- and more importantly, it must NOT touch ``float_split``.
        The WMO-disjoint cohort is drawn from float ids alone and is deliberately
        independent of the year split; re-drawing it here would let the choice of
        evaluation era change which floats are held out, which is exactly the
        degree of freedom the registered cohort seed exists to remove.
        """
        lab = np.full(self.year.size, "unassigned", dtype=object)
        for name, (lo, hi) in table.items():
            lab[(self.year >= lo) & (self.year <= hi)] = name
        self.year_split = lab.astype(str)
        return self

    def months_in(self, split: str) -> np.ndarray:
        """Months whose profiles fall in a year split, sorted."""
        return np.unique(self.month_index[self.year_split == split])


# --------------------------------------------------------------------------
@dataclass
class ArgoNorm:
    """Per-level z-scores from TRAIN months only.

    Train-only is not a formality: statistics that see the validation or holdout
    era leak the target distribution into the input scaling, and the leak is
    invisible in every diagnostic because the model still "only" saw inputs.
    """
    mean: dict
    std: dict

    @staticmethod
    def fit(c: ArgoCohort, split: str = "train") -> "ArgoNorm":
        m = c.year_split == split
        if not m.any():
            raise ValueError(f"no profiles in split {split!r}")
        mean, std = {}, {}
        for ch, arr in (("TEMP", c.TEMP), ("SALT", c.SALT)):
            a = arr[m]
            mu = np.nanmean(a, axis=0)
            sd = np.nanstd(a, axis=0)
            # a level with no spread would divide by ~0 and manufacture huge
            # z-values from rounding; 1.0 leaves it in physical units instead
            sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
            mean[ch] = np.where(np.isfinite(mu), mu, 0.0)
            std[ch] = sd
        return ArgoNorm(mean, std)

    def z(self, ch: str, a: np.ndarray) -> np.ndarray:
        return (a - self.mean[ch]) / self.std[ch]

    def unz(self, ch: str, a: np.ndarray) -> np.ndarray:
        return a * self.std[ch] + self.mean[ch]


# --------------------------------------------------------------------------
@dataclass
class ArgoObsConfig:
    n_profiles: int = 24            # 0 = every profile the month actually has
    patch: int = 4
    context: int = 2
    n_queries: int = 512
    max_lead: int = 3
    modality_dropout: float = 0.2
    target_dropout: float = 0.2
    train: bool = False
    force_available: tuple | None = None
    use_measured_noise: bool = True
    #: variance floor for a reported error of 0, in the variable's own units.
    #: Argo occasionally reports 0.0 error; taken literally that is infinite
    #: evidence from one bottle, which would dominate the whole solve.
    min_error: float = 1e-3


def _select_profiles(c: ArgoCohort, idx: np.ndarray, n: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Subsample by float first, then by profile — see the module docstring."""
    if n <= 0 or idx.size <= n:
        return idx
    floats = np.unique(c.wmo[idx])
    rng.shuffle(floats)
    picked = []
    per = max(1, n // max(floats.size, 1))
    for w in floats:
        rows = idx[c.wmo[idx] == w]
        take = rows if rows.size <= per else rng.choice(rows, per, replace=False)
        picked.append(np.atleast_1d(take))
        if sum(p.size for p in picked) >= n:
            break
    out = np.concatenate(picked)
    return out[:n] if out.size > n else out


def training_rows(c: ArgoCohort, t_src: int, lead: int, n_profiles: int,
                  rng: np.random.Generator,
                  target_frac: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Input/target rows for a TRAINING sample, split by float within the cohort.

    The held-out floats may never be a training target -- that would spend the
    evaluation cohort on optimisation and make every WMO-clustered number
    meaningless.  So training draws both its input and its target from
    ``cohort_float``, and partitions them **by WMO**: a float is either an input
    or a target in a given sample, never both.

    Partitioning by profile instead would be the subtle version of the same
    leak.  A float's cycles within a month sit within a few hundred km of each
    other and share a sensor; predicting cycle 41 from cycle 40 of the same
    instrument is close to copying, and a model rewarded for it learns to lean
    on the input profile nearest the query rather than on the field.

    The partition is redrawn every sample, so over training the model sees every
    float in both roles -- just never simultaneously.
    """
    src = c.month(t_src, float_split="cohort_float")
    tgt_pool = c.month(t_src + lead, float_split="cohort_float")
    if src.size == 0 or tgt_pool.size == 0:
        return src[:0], tgt_pool[:0]
    floats = np.unique(np.r_[c.wmo[src], c.wmo[tgt_pool]])
    if floats.size < 2:
        return src[:0], tgt_pool[:0]
    n_tgt = max(1, min(floats.size - 1, int(round(target_frac * floats.size))))
    tgt_floats = set(rng.choice(floats, n_tgt, replace=False).tolist())
    src_rows = src[~np.isin(c.wmo[src], list(tgt_floats))]
    tgt_rows = tgt_pool[np.isin(c.wmo[tgt_pool], list(tgt_floats))]
    if src_rows.size == 0 or tgt_rows.size == 0:
        return src[:0], tgt_pool[:0]
    return _select_profiles(c, src_rows, n_profiles, rng), tgt_rows


def build_argo_sample(c: ArgoCohort, norm: ArgoNorm, t_src: int,
                      grid_fields: dict | None = None,
                      cfg: ArgoObsConfig | None = None,
                      rng: np.random.Generator | None = None,
                      lead: int = 0,
                      profile_rows: np.ndarray | None = None,
                      target_rows: np.ndarray | None = None) -> dict | None:
    """One real-observation sample.  Returns None if the month has no target.

    ``grid_fields`` optionally supplies the dense context streams (z-scored
    ``TEMP``/``SALT``/``SSH`` on the region grid, indexed by the same month
    axis).  Passing None runs profiles-only, which is a legitimate configuration
    and not a degraded one — it is the regime the sparsity stress explores.

    ``profile_rows`` / ``target_rows`` override the input and target selection.
    Every layout, redundancy and sparsity operator in `argo_experiments` works by
    handing this function a different row set, so the sample construction stays
    one code path and an experiment cannot accidentally change two things at once.
    """
    cfg = cfg or ArgoObsConfig()
    rng = rng or np.random.default_rng(0)
    if not 0 <= lead <= cfg.max_lead:
        raise ValueError(f"lead must be in 0..{cfg.max_lead}, got {lead}")
    L = c.levels.size
    dz_all = level_thickness_m(c.levels)
    z_norm = c.levels / BOX_DEPTH_M
    NY, NX = c.grid

    if target_rows is None:
        target_rows = c.month(t_src + lead, float_split="heldout_float")
    if target_rows.size == 0:
        return None                       # no held-out float reported: no target
    if profile_rows is None:
        profile_rows = _select_profiles(
            c, c.month(t_src, float_split="cohort_float"), cfg.n_profiles, rng)

    avail = np.ones(N_MODALITIES, dtype=bool)
    if cfg.force_available is not None:
        avail = np.asarray(cfg.force_available, dtype=bool)
    elif cfg.train and cfg.modality_dropout > 0:
        while True:
            avail = rng.random(N_MODALITIES) >= cfg.modality_dropout
            if avail.any():
                break
    if grid_fields is None:
        avail = avail & np.array([True, False, False])

    coord, value, vmask, modality, noise, support = [], [], [], [], [], []
    support_dz, prov, measured = [], [], []

    # ---- real profile tokens -------------------------------------------
    if avail[MOD_PROFILE] and profile_rows.size:
        wmos = c.wmo[profile_rows]
        # provenance group = platform. Two cycles of one float share a sensor,
        # so they share a noise group; this is the real version of what
        # godas_obs approximated with a column index.
        uniq_w = {w: i for i, w in enumerate(np.unique(wmos))}
        for r, w in zip(profile_rows, wmos):
            t = np.stack([norm.z("TEMP", c.TEMP[r]), norm.z("SALT", c.SALT[r])],
                         axis=-1)                                    # (L,2)
            coord.append(np.stack([
                np.full(L, c.grid_x[r] / max(NX - 1, 1)),
                np.full(L, c.grid_y[r] / max(NY - 1, 1)),
                z_norm, np.zeros(L)], axis=-1))
            value.append(t)
            vmask.append(np.isfinite(t))
            modality.append(np.full(L, MOD_PROFILE))
            support.append(np.full(L, profile_support_area_km2()))
            support_dz.append(dz_all.copy())
            prov.append(np.full(L, uniq_w[w]))
            # measured error variance, in z units, per level
            e_t = c.TEMP_ERR[r] / norm.std["TEMP"]
            e_s = c.SALT_ERR[r] / norm.std["SALT"]
            have = cfg.use_measured_noise and np.isfinite(e_t) & np.isfinite(e_s)
            var = np.maximum(np.nanmean(np.stack([e_t, e_s]), axis=0) ** 2,
                             cfg.min_error ** 2)
            noise.append(np.where(have, var * NOISE_AREA_POINT_KM2,
                                  NOISE_AREA_POINT_KM2))
            measured.append(have.astype(bool))

    # ---- dense gridded context, when supplied ---------------------------
    if grid_fields is not None:
        boxes = _patch_boxes(NY, NX, cfg.patch)
        base = int(prov[-1].max()) + 1 if prov else 0
        for mod, keys in ((MOD_SURF, ("TEMP", "SALT")), (MOD_SSH, ("SSH",))):
            if not avail[mod] or not all(k in grid_fields for k in keys):
                continue
            for back in range(cfg.context - 1, -1, -1):
                tt = max(t_src - back, 0)
                for (y0, x0, hy, hx) in boxes:
                    vals = []
                    for k in keys:
                        a = grid_fields[k]
                        box = (a[tt, 0, y0:y0 + hy, x0:x0 + hx] if a.ndim == 4
                               else a[tt, y0:y0 + hy, x0:x0 + hx])
                        with np.errstate(invalid="ignore"):
                            vals.append(np.nanmean(box) if np.isfinite(box).any()
                                        else np.nan)
                    v = np.array(vals + [0.0] * (N_CHANNELS - len(vals)))
                    m = np.array([np.isfinite(x) for x in vals]
                                 + [False] * (N_CHANNELS - len(vals)))
                    coord.append(np.array([[(x0 + hx / 2) / max(NX - 1, 1),
                                            (y0 + hy / 2) / max(NY - 1, 1),
                                            0.0, -float(back)]]))
                    value.append(v[None]); vmask.append(m[None])
                    modality.append(np.array([mod]))
                    noise.append(np.array([NOISE_AREA_PATCH_KM2]))
                    support.append(np.array([patch_support_area_km2(hy, hx, NY, NX)]))
                    support_dz.append(np.array([dz_all[0]]))
                    prov.append(np.array([base + mod * cfg.context + back]))
                    measured.append(np.array([False]))

    if coord:
        cat = lambda xs: np.concatenate(xs)
        coord, value, vmask = cat(coord), cat(value), cat(vmask)
        modality, noise, support = cat(modality), cat(noise), cat(support)
        support_dz, prov, measured = cat(support_dz), cat(prov), cat(measured)
    else:
        coord = np.zeros((0, 4)); value = np.zeros((0, N_CHANNELS))
        vmask = np.zeros((0, N_CHANNELS), bool); modality = np.zeros(0, int)
        noise = np.zeros(0); support = np.zeros(0); support_dz = np.zeros(0)
        prov = np.zeros(0, dtype=int); measured = np.zeros(0, bool)

    value = np.where(vmask, np.nan_to_num(value), 0.0)
    tok = torch.as_tensor(vmask.any(axis=-1))

    # ---- targets: the held-out floats' own measurements -----------------
    tgt_ch = np.ones(N_CHANNELS, dtype=bool)
    if cfg.train and cfg.target_dropout > 0 and rng.random() < cfg.target_dropout:
        tgt_ch[rng.integers(0, N_CHANNELS)] = False

    tz = np.stack([norm.z("TEMP", c.TEMP[target_rows]),
                   norm.z("SALT", c.SALT[target_rows])], axis=-1)    # (R,L,2)
    R = target_rows.size
    qy = np.repeat(c.grid_y[target_rows] / max(NY - 1, 1), L)
    qx = np.repeat(c.grid_x[target_rows] / max(NX - 1, 1), L)
    qz = np.tile(np.arange(L) / max(L - 1, 1), R)
    qcoord = np.stack([qx, qy, qz, np.full(R * L, float(lead))], axis=-1)
    target = tz.reshape(R * L, N_CHANNELS)
    tmask = np.isfinite(target) & tgt_ch[None, :]
    # the cluster label the plan's CIs need, carried with the target itself so
    # an aggregation cannot pair a residual with the wrong float
    tgt_wmo = np.repeat(c.wmo[target_rows], L)
    # position and level of every query, carried alongside.  A gridded
    # reference product (EN4, ECCO) is queried at real lat/lon/depth, not at
    # the normalised box coordinate, so it needs the physical position; keeping
    # it on the sample means the reference row and the model row are scored on
    # exactly the same queries rather than on two separately-built sets.
    tgt_lat = np.repeat(c.lat[target_rows], L)
    tgt_lon = np.repeat(c.lon[target_rows], L)
    tgt_lev = np.tile(c.levels, R)

    if cfg.n_queries and R * L > cfg.n_queries:
        pick = rng.choice(R * L, cfg.n_queries, replace=False)
        qcoord, target, tmask = qcoord[pick], target[pick], tmask[pick]
        tgt_wmo, tgt_lat = tgt_wmo[pick], tgt_lat[pick]
        tgt_lon, tgt_lev = tgt_lon[pick], tgt_lev[pick]

    return dict(
        coord=torch.as_tensor(coord, dtype=torch.float64),
        value=torch.as_tensor(value, dtype=torch.float32),
        value_mask=torch.as_tensor(vmask),
        mask=tok, support_mask=tok.clone(),
        modality=torch.as_tensor(modality, dtype=torch.long),
        variable_group=torch.as_tensor(
            np.vectorize(VARIABLE_GROUP.get)(modality).astype("int64")
            if modality.size else np.zeros(0, dtype="int64")),
        modality_available=torch.as_tensor(avail),
        noise_density=torch.as_tensor(noise, dtype=torch.float64),
        support_area=torch.as_tensor(support, dtype=torch.float64),
        support_dz=torch.as_tensor(support_dz, dtype=torch.float64),
        provenance=torch.as_tensor(np.asarray(prov, dtype="int64")),
        noise_is_measured=torch.as_tensor(measured),
        query=torch.as_tensor(qcoord, dtype=torch.float64),
        target=torch.as_tensor(np.nan_to_num(target), dtype=torch.float32),
        target_mask=torch.as_tensor(tmask),
        target_wmo=tgt_wmo, target_lat=tgt_lat, target_lon=tgt_lon,
        target_level=tgt_lev, target_month=int(t_src + lead),
        t_src=int(t_src), lead=int(lead),
        n_input_profiles=int(profile_rows.size),
        n_target_profiles=int(R),
    )
