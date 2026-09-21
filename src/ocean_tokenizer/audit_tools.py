"""Shared pieces for the pipeline audit (Wang, 2026-09-17 meeting).

Three things the audit needs that the registered pipeline does not provide:

* **Input QC that a real mapping system would apply.**  The cohort keeps every
  value whose Argo QC flag is good, but a handful of floats still deliver
  salinity hundreds of standard deviations from climatology (a drifting
  real-time conductivity cell, an "adjusted" profile with a bad adjustment).
  Fed to an encoder, one such value dominates a whole month's latent.
  :func:`robust_qc` flags them with train-only statistics.

* **The climatology at each profile's own position.**  The registered anomaly
  subtracts WOA23 at the centre of the 0.66 x 1.96 deg region cell, so a
  profile up to ~100 km from the centre inherits the climatological gradient
  across that distance as a spurious "anomaly".  :func:`exact_position_anomaly`
  rebuilds the target with the climatology interpolated to the profile itself.

* **How predictable the anomaly is at all.**  :func:`pair_correlation` and
  :func:`fit_cov` measure how fast the anomaly decorrelates between floats, and
  the nugget of the fit is the share of the variance no other float sees — the
  floor on skill at a held-out float.
"""
from __future__ import annotations

import dataclasses
import os

import numpy as np

from .argo_obs import ArgoCohort, ArgoNorm

R_EARTH_KM = 6371.0
CHANNELS = ("TEMP", "SALT")


# --------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance with numpy broadcasting."""
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dp = p2 - p1
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# --------------------------------------------------------------------------
@dataclasses.dataclass
class QCReport:
    k_sigma: float
    n_values_flagged: dict
    n_profiles_dropped: dict
    std_before: dict
    std_after: dict
    by_data_mode: dict


def robust_qc(c: ArgoCohort, split: str = "train", k_sigma: float = 8.0,
              n_iter: int = 3, profile_levels: int = 3,
              data_mode: np.ndarray | None = None) -> QCReport:
    """Flag gross outliers in place (set to NaN), statistics from ``split`` only.

    Per channel and level: the mean and standard deviation are fitted on the
    training years, values beyond ``k_sigma`` are excluded, and the fit is
    repeated ``n_iter`` times so the outliers cannot inflate the very scale used
    to find them (one float at z=719 moves the deep N. Pacific salinity std six
    times).  A profile with ``profile_levels`` or more flagged levels in a
    channel loses that whole channel: a drifting sensor corrupts every level,
    not only the ones that happen to cross the threshold.

    8 sigma is deliberately loose.  The Gulf Stream anomaly is heavy-tailed for
    physical reasons (warm- and cold-core rings sit several degrees off
    climatology), and QC must not remove the signal the model is asked for.
    """
    tr = c.year_split == split
    flagged, dropped, sd0, sd1, modes = {}, {}, {}, {}, {}
    for ch in CHANNELS:
        a = getattr(c, ch)
        bad = ~np.isfinite(a)
        sd0[ch] = np.nanstd(a[tr], axis=0)
        for _ in range(n_iter):
            ref = np.where(bad, np.nan, a)
            mu = np.nanmean(ref[tr], axis=0)
            sd = np.nanstd(ref[tr], axis=0)
            sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, np.inf)
            with np.errstate(invalid="ignore"):
                bad = bad | (np.abs(a - mu) > k_sigma * sd)
        new = bad & np.isfinite(a)
        whole = new.sum(axis=1) >= profile_levels
        new = new | (whole[:, None] & np.isfinite(a))
        flagged[ch] = int(new.sum())
        dropped[ch] = int(whole.sum())
        if data_mode is not None:
            u, n = np.unique(data_mode[new.any(axis=1)], return_counts=True)
            modes[ch] = {str(k): int(v) for k, v in zip(u, n)}
        a = np.where(new, np.nan, a)
        setattr(c, ch, a)
        sd1[ch] = np.nanstd(a[tr], axis=0)
    return QCReport(k_sigma, flagged, dropped,
                    {k: v.tolist() for k, v in sd0.items()},
                    {k: v.tolist() for k, v in sd1.items()}, modes)


# --------------------------------------------------------------------------
def woa_monthly(root: str):
    import xarray as xr
    w = xr.open_zarr(os.path.join(root, "data", "woa23_standard.zarr"))
    return w.assign_coords(lon=(w.lon % 360.0)).sortby("lon")


def climatology_anomaly(raw: ArgoCohort, root: str, lat: np.ndarray,
                        lon: np.ndarray, chunk: int = 20000) -> dict[str, np.ndarray]:
    """TEMP/SALT minus WOA23 read at (``lat``, ``lon``) in each profile's month.

    Passing the profiles' own positions gives the exact-position anomaly; passing
    their grid-cell centres reproduces how the registered cohorts were built.
    Arrays are aligned with ``raw`` (month-sorted by ``ArgoCohort.load``).
    """
    import xarray as xr
    w = woa_monthly(root)
    out = {}
    cm = raw.month_index % 12 + 1
    for ch in CHANNELS:
        da = w[ch].load()
        clim = np.empty(getattr(raw, ch).shape, dtype="float32")
        for i in range(0, cm.size, chunk):
            sl = slice(i, i + chunk)
            clim[sl] = da.interp(
                time=xr.DataArray(cm[sl], dims="p"),
                lat=xr.DataArray(lat[sl], dims="p"),
                lon=xr.DataArray(lon[sl] % 360.0, dims="p"),
                depth=xr.DataArray(raw.levels, dims="l"),
                method="linear").transpose("p", "l").values
        out[ch] = getattr(raw, ch) - clim
    return out


def exact_position_anomaly(raw: ArgoCohort, root: str,
                           chunk: int = 20000) -> dict[str, np.ndarray]:
    """The anomaly with the climatology at each profile's own lat/lon."""
    return climatology_anomaly(raw, root, raw.lat, raw.lon, chunk)


def cohort_path(root: str, region: str) -> str:
    """The RAW cohort file: ``global_global.nc`` for the global run, else
    ``<region>_ext.nc`` (the extended 23-level regional cohorts)."""
    base = os.path.join(root, "data", "argo_cohort")
    return os.path.join(base, "global_global.nc" if region == "global"
                        else f"{region}_ext.nc")


def load_cohort(root: str, region: str, split_table: dict | None = None,
                anomaly: str = "cell", qc: bool = False,
                k_sigma: float = 8.0) -> tuple[ArgoCohort, QCReport | None]:
    """An anomaly cohort under the audit's switches.

    ``anomaly="cell"`` subtracts WOA23 at the centre of the profile's grid cell
    (how the registered regional cohorts were built); ``"exact"`` at the
    profile's own position. Regional cells come from the shipped
    ``<region>_ext_anom.nc``; the global cohort ships raw values, so both
    variants are computed from it and cached. ``qc=True`` applies
    :func:`robust_qc` after the split table is set, so its statistics come from
    the training years of THAT protocol.
    """
    base = os.path.join(root, "data", "argo_cohort")
    raw_path = cohort_path(root, region)
    if anomaly not in ("cell", "exact"):
        raise ValueError(anomaly)
    if anomaly == "cell" and region != "global":
        c = ArgoCohort.load(os.path.join(base, f"{region}_ext_anom.nc"))
    else:
        c = ArgoCohort.load(raw_path)
        tag = "anomx" if anomaly == "exact" else "anomcell"
        cache = os.path.join(base, (f"{region}_{tag}.npz" if region == "global"
                                    else f"{region}_ext_{tag}.npz"))
        if os.path.exists(cache):
            z = np.load(cache)
            assert z["n"] == c.TEMP.shape[0], "stale anomaly cache"
            c.TEMP, c.SALT = z["TEMP"].astype(float), z["SALT"].astype(float)
        else:
            if anomaly == "exact":
                a = exact_position_anomaly(c, root)
            else:
                (la0, la1), (lo0, lo1) = ((-90.0, 90.0), (0.0, 360.0))
                NY, NX = c.grid
                a = climatology_anomaly(
                    c, root, la0 + (c.grid_y + 0.5) * (la1 - la0) / NY,
                    lo0 + (c.grid_x + 0.5) * (lo1 - lo0) / NX)
            np.savez(cache, TEMP=a["TEMP"], SALT=a["SALT"], n=c.TEMP.shape[0])
            c.TEMP, c.SALT = a["TEMP"].astype(float), a["SALT"].astype(float)
    if split_table is not None:
        c.apply_splits(split_table)
    rep = None
    if qc:
        import xarray as xr
        d = xr.open_dataset(raw_path)
        order = np.argsort(np.asarray(d["month_index"].values, int), kind="stable")
        dm = np.asarray(d["data_mode"].values).astype(str)[order]
        rep = robust_qc(c, k_sigma=k_sigma, data_mode=dm)
    return c, rep


BINS_KM = np.array([0, 10, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1200])


def pair_correlation(c: ArgoCohort, norm: ArgoNorm, ch: str, level: int,
                     months: np.ndarray, bins=BINS_KM, max_months: int = 200,
                     seed: int = 0, max_profiles: int = 1500,
                     rows: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Same-month correlation of z-anomalies between DIFFERENT floats, by distance.

    Returns (correlation per bin, pair count per bin).  Pairs from one float are
    excluded: two cycles of one instrument share its calibration and would
    report instrument persistence as ocean covariance.
    """
    rng = np.random.default_rng(seed)
    if months.size > max_months:
        months = rng.choice(months, max_months, replace=False)
    nb = len(bins) - 1
    num, den, cnt = np.zeros(nb), np.zeros(nb), np.zeros(nb)
    in_rows = None if rows is None else np.zeros(c.TEMP.shape[0], bool)
    if in_rows is not None:
        in_rows[rows] = True
    for m in months:
        idx = c.month(int(m))
        if in_rows is not None:
            idx = idx[in_rows[idx]]
        # a global month holds ~7 600 profiles; the pair count is quadratic, and
        # a random subset estimates the same binned correlation
        if idx.size > max_profiles:
            idx = np.sort(rng.choice(idx, max_profiles, replace=False))
        v = (getattr(c, ch)[idx, level] - norm.mean[ch][level]) / norm.std[ch][level]
        ok = np.isfinite(v)
        idx, v = idx[ok], v[ok]
        if idx.size < 3:
            continue
        d = haversine_km(c.lat[idx][:, None], c.lon[idx][:, None],
                         c.lat[idx][None], c.lon[idx][None])
        iu = np.triu_indices(idx.size, 1)
        keep = (c.wmo[idx][:, None] != c.wmo[idx][None])[iu]
        dd = d[iu][keep]
        prod = (v[:, None] * v[None])[iu][keep]
        sq = ((v[:, None] ** 2 + v[None] ** 2) / 2)[iu][keep]
        b = np.digitize(dd, bins) - 1
        ok = (b >= 0) & (b < nb)
        num += np.bincount(b[ok], weights=prod[ok], minlength=nb)
        den += np.bincount(b[ok], weights=sq[ok], minlength=nb)
        cnt += np.bincount(b[ok], minlength=nb)
    return num / np.maximum(den, 1e-12), cnt


# --------------------------------------------------------------------------
@dataclasses.dataclass
class CovModel:
    """rho(d) = a0 + a1 g(d; L1) + a2 g(d; L2); nugget = 1 - a0 - a1 - a2."""
    a0: float
    a1: float
    L1: float
    a2: float
    L2: float

    @property
    def nugget(self) -> float:
        return max(1.0 - self.a0 - self.a1 - self.a2, 0.02)

    def __call__(self, d: np.ndarray) -> np.ndarray:
        return (self.a0 + self.a1 * np.exp(-0.5 * (d / self.L1) ** 2)
                + self.a2 * np.exp(-0.5 * (d / self.L2) ** 2))


def fit_cov(corr: np.ndarray, cnt: np.ndarray, bins=BINS_KM) -> CovModel:
    """Weighted least squares over binned correlations, grid over (L1, L2)."""
    from scipy.optimize import nnls
    mid = 0.5 * (bins[:-1] + bins[1:])
    ok = cnt > 50
    x, y, w = mid[ok], corr[ok], np.sqrt(np.minimum(cnt[ok], 2e5))
    best = None
    for L1 in (10, 15, 20, 30, 40, 50, 60, 75, 90, 110, 140, 180, 240):
        for L2 in (300, 450, 650, 900, 1300, 2000):
            A = np.stack([np.ones_like(x), np.exp(-0.5 * (x / L1) ** 2),
                          np.exp(-0.5 * (x / L2) ** 2)], -1)
            coef, _ = nnls(A * w[:, None], y * w)
            s = coef.sum()
            if s > 0.98:
                coef = coef * 0.98 / s
            sse = float((((A @ coef) - y) * w) @ (((A @ coef) - y) * w))
            if best is None or sse < best[0]:
                best = (sse, CovModel(coef[0], coef[1], L1, coef[2], L2))
    return best[1]


#: absolute-latitude bands for the global covariance fit: a front-dominated
#: mid-latitude ocean and the tropical wave guide do not share one covariance
LAT_BANDS = ((0.0, 20.0), (20.0, 45.0), (45.0, 90.1))


def band_of_lat(lat: np.ndarray, bands=LAT_BANDS) -> np.ndarray:
    a = np.abs(np.asarray(lat))
    out = np.zeros(a.shape, int)
    for i, (lo, hi) in enumerate(bands):
        out[(a >= lo) & (a < hi)] = i
    return out
