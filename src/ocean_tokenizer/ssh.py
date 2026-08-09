"""Pseudo-SSH modality: steric (dynamic) height from the T/S truth fields.

Why a *pseudo* SSH
------------------
The advisor's tokenization list includes the satellite altimeter, but the
standardized stores hold only TEMP/SALT/SST/SSS — there is no SSH variable
([config.py](config.py) `ZARR`).  Downloading and regridding true CESM2-LE SSH
is route B (external data, and the original 1 deg regrid recipe would have to be
reproduced exactly).  Route A, implemented here, derives the **steric /
baroclinic** component of sea-surface height from the T/S fields we already
have — the dominant part of what an altimeter sees over the open ocean at
monthly, 1 deg resolution.

Honest limitations, to be repeated in any report that uses this field:

* It is the **baroclinic component only**, referenced to ~990 dbar.  There is no
  barotropic (bottom-pressure) contribution and no mass/freshwater term, so this
  is not comparable to an altimeter product in absolute terms.
* Because it is computed *from* the same TEMP/SALT that the model is asked to
  reconstruct, it is a **derived** observation, not an independent one.  Within
  the OSSE that is internally consistent — it tells us whether a vertically
  integrated surface constraint helps — but it cannot answer "does real
  altimetry help", which is a Phase-5 question.
* Columns that do not reach the reference level (shelves, marginal seas) are
  undefined and returned as NaN, exactly as they are in real dynamic-height
  maps.

Method (TEOS-10, via ``gsw``)
-----------------------------
For each ocean column::

    p   = p_from_z(-z, lat)                  # dbar
    SA  = SA_from_SP(SP, p, lon, lat)        # absolute salinity
    CT  = CT_from_t(SA, t, p)                # conservative temperature
    Phi = geo_strf_dyn_height(SA, CT, p, p_ref)   # m^2/s^2, dynamic height anomaly
    eta = Phi[surface] / g(lat)              # metres of steric height

``p_ref = 990 dbar`` sits below the deepest analysis level (985 m -> 992.8 dbar
at the equator, 998.1 dbar at the pole), so one fixed reference is valid at
every latitude.
"""
from __future__ import annotations

import numpy as np

#: reference pressure (dbar); below the 985 m level at every latitude
P_REF_DBAR = 990.0


def steric_height_columns(temp, salt, depth, lat, lon, p_ref=P_REF_DBAR):
    """Steric height in metres for a set of full columns.

    Parameters
    ----------
    temp, salt : (D, N)
        In-situ temperature (degC) and practical salinity (PSU) profiles.
        Columns must be finite at every level; screen them first.
    depth : (D,)
        Positive depths in metres, increasing.
    lat, lon : (N,)
        Column coordinates in degrees (lon in either convention).

    Returns
    -------
    (N,) steric height in metres, relative to ``p_ref``.
    """
    import gsw

    depth = np.asarray(depth, dtype="float64")
    lat = np.asarray(lat, dtype="float64")
    lon = np.asarray(lon, dtype="float64")
    p = gsw.p_from_z(-depth[:, None], lat[None, :])            # (D,N) dbar
    SA = gsw.SA_from_SP(np.asarray(salt, dtype="float64"), p,
                        lon[None, :], lat[None, :])
    CT = gsw.CT_from_t(SA, np.asarray(temp, dtype="float64"), p)
    phi = gsw.geo_strf_dyn_height(SA, CT, p, p_ref=p_ref, axis=0)   # (D,N)
    g = gsw.grav(lat, 0.0)                                      # (N,) m/s^2
    return phi[0] / g


def steric_height_field(temp3d, salt3d, grid, p_ref=P_REF_DBAR):
    """One monthly (H, W) steric-height field in metres; NaN where undefined.

    ``temp3d`` / ``salt3d`` are (D, H, W) fields on the analysis grid.  A column
    is used only where every analysis level is finite in both variables, i.e.
    where the water column actually reaches the reference level.
    """
    D, H, W = temp3d.shape
    ok = (np.isfinite(temp3d).all(axis=0) & np.isfinite(salt3d).all(axis=0)
          & grid.ocean)                                          # (H,W)
    out = np.full((H, W), np.nan, dtype="float32")
    ii, jj = np.where(ok)
    if ii.size == 0:
        return out
    lat2d, lon2d = np.meshgrid(grid.lat, grid.lon, indexing="ij")
    eta = steric_height_columns(temp3d[:, ii, jj], salt3d[:, ii, jj],
                                grid.depth, lat2d[ii, jj], lon2d[ii, jj],
                                p_ref=p_ref)
    out[ii, jj] = eta.astype("float32")
    return out


class SSHAnom:
    """Train-only monthly climatology + z-score for the pseudo-SSH field.

    Mirrors the discipline of :class:`anomaly.AnomNorm` (from which it is kept
    separate so that adding SSH cannot perturb the TEMP/SALT/SST/SSS statistics
    the certified checkpoints were trained with): the climatology and the
    scalar mean/std come from **training months only**, and a calendar month
    absent from the training split falls back to the all-month mean.

    The z-scored field keeps NaN where the steric height is undefined (columns
    that do not reach the reference level); the consumer turns that into
    value 0 + a finite flag.
    """

    def __init__(self, train_ssh, train_months):
        x = np.asarray(train_ssh, dtype="float32")           # (T,H,W)
        m = np.asarray(train_months, dtype=int)
        # all-NaN columns (never-defined cells) are expected; the resulting NaN
        # climatology is the correct answer there, so silence the slice warning
        # exactly as anomaly.Climatology does.
        with np.errstate(invalid="ignore"):
            glob = np.nanmean(x, axis=0)
            self.clim = np.empty((12,) + x.shape[1:], dtype="float32")
            for cm in range(1, 13):
                sel = m == cm
                self.clim[cm - 1] = np.nanmean(x[sel], axis=0) if sel.any() else glob
        a = x - self.clim[m - 1]
        self.mean = float(np.nan_to_num(np.nanmean(a)))
        sd = float(np.nanstd(a))
        # Same degenerate-variance guard as anomaly.AnomNorm (anomaly.py:96).
        # The threshold has to sit well above float32 rounding dust: a field
        # whose anomaly is genuinely zero still leaves a residual of ~1e-8 m
        # from the climatology subtraction, and dividing that by its own std
        # would manufacture O(1) z-scores out of nothing.
        self.std = sd if sd > 1e-6 else 1.0

    def z(self, field, month):
        """(H,W) steric height in metres -> z-scored anomaly (NaN preserved)."""
        return (np.asarray(field, dtype="float32") - self.clim[month - 1]
                - self.mean) / self.std


def load_ssh_cache(path):
    """Load the cached pseudo-SSH archive written by ``28_make_ssh.py``.

    Returns ``(ssh, time_index)`` where ``ssh`` is (T, H, W) float32 and
    ``time_index`` the (T,) zarr time indices, so a caller can align the field
    with any split by index lookup.
    """
    z = np.load(path)
    return z["ssh"], z["time_index"]


def ssh_for_indices(ssh, time_index, wanted):
    """Select the cached SSH rows matching ``wanted`` zarr time indices.

    Raises if any requested month is missing from the cache rather than
    silently returning a misaligned field.
    """
    pos = {int(t): i for i, t in enumerate(np.asarray(time_index))}
    missing = [int(t) for t in np.asarray(wanted) if int(t) not in pos]
    if missing:
        raise KeyError(f"pseudo-SSH cache is missing time indices {missing[:8]}"
                       f"{' ...' if len(missing) > 8 else ''}; regenerate with "
                       f"experiments/28_make_ssh.py --years <range>")
    return ssh[[pos[int(t)] for t in np.asarray(wanted)]]
