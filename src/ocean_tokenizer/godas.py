"""GODAS loading, unit conversion and train-only normalisation (mentor §4).

Source: NOAA PSL GODAS monthly, https://psl.noaa.gov/data/gridded/data.godas.html
Downloaded by ``experiments/13_download_godas.py``.

The doc calls for "checked GODAS loading": the checks here are not decoration.
A silent unit error (Kelvin read as Celsius) or a missing month shifts every
anomaly without ever raising, and produces numbers that look entirely
reasonable.  Every conversion therefore keys off the file's declared units and
**raises on anything it does not recognise**, rather than inferring from
magnitude — inference is precisely how such a bug survives.

Measured on the downloaded subset (2000-2025, 25-50 N, 280-331 E):
``pottmp`` is K (270.5-302.9), ``salt`` is kg/kg (0.0306-0.0375), ``sshg`` is
m.  Fill values already decode to NaN via xarray.  Note the SSH land mask
differs slightly from T/S (14.5 % vs 16.5 % NaN), so masks are per-variable.
"""
from __future__ import annotations

import os
import glob
import warnings

import numpy as np

# unit spellings seen in GODAS / accepted without conversion
_KELVIN = {"K", "kelvin", "degK", "deg_K"}
_CELSIUS = {"degC", "celsius", "C", "deg_C"}
_KG_PER_KG = {"kg/kg", "kg kg-1", "kg/kg**1"}
_G_PER_KG = {"g/kg", "psu", "PSU", "1e-3"}

KELVIN_OFFSET = 273.15


def check_contiguous_months(times) -> None:
    """Raise unless ``times`` is sorted, unique and has no missing month.

    A gap silently changes which calendar month each index refers to, which
    corrupts a monthly climatology without any other symptom.
    """
    t = np.asarray(times).astype("datetime64[M]")
    if t.size == 0:
        raise ValueError("no months supplied")
    if np.any(np.diff(t.astype("int64")) < 0):
        raise ValueError("months are not sorted in ascending order")
    uniq, counts = np.unique(t, return_counts=True)
    if np.any(counts > 1):
        bad = ", ".join(str(x) for x in uniq[counts > 1][:5])
        raise ValueError(f"duplicate month(s): {bad}")
    step = np.diff(t.astype("int64"))
    if np.any(step != 1):
        where = t[:-1][step != 1]
        raise ValueError(
            f"gap in the month series: not contiguous after "
            f"{', '.join(str(x) for x in where[:5])}. A monthly climatology "
            f"cannot be fitted on a series with missing months.")


def to_celsius(arr: np.ndarray, units: str | None) -> np.ndarray:
    """Temperature -> degC, keyed off the declared units only."""
    if units in _CELSIUS:
        return arr
    if units in _KELVIN:
        return arr - KELVIN_OFFSET
    raise ValueError(
        f"unrecognised temperature unit {units!r}; refusing to guess from "
        f"magnitude. Accepted: {sorted(_KELVIN | _CELSIUS)}")


def to_g_per_kg(arr: np.ndarray, units: str | None) -> np.ndarray:
    """Salinity -> g/kg, keyed off the declared units only."""
    if units in _G_PER_KG:
        return arr
    if units in _KG_PER_KG:
        return arr * 1000.0
    raise ValueError(
        f"unrecognised salinity unit {units!r}; refusing to guess from "
        f"magnitude. Accepted: {sorted(_KG_PER_KG | _G_PER_KG)}")


def check_alignment(axes: dict[str, dict]) -> None:
    """Raise unless every variable shares one time axis and one grid.

    ``axes`` maps variable name -> {"time":, "lat":, "lon":}.  Coordinates are
    compared by value, not just by length: a grid shifted by one cell has the
    same shape and would otherwise pair the wrong water with the wrong cell.
    """
    if not axes:
        raise ValueError("no variables supplied")
    names = list(axes)
    ref_name, ref = names[0], axes[names[0]]
    for name in names[1:]:
        cur = axes[name]
        rt = np.asarray(ref["time"]).astype("datetime64[M]")
        ct = np.asarray(cur["time"]).astype("datetime64[M]")
        if rt.shape != ct.shape or np.any(rt != ct):
            raise ValueError(
                f"time axis of {name!r} does not match {ref_name!r} "
                f"({ct.size} vs {rt.size} months)")
        for ax in ("lat", "lon"):
            r, c = np.asarray(ref[ax]), np.asarray(cur[ax])
            if r.shape != c.shape:
                raise ValueError(
                    f"{ax} grid of {name!r} has shape {c.shape}, "
                    f"{ref_name!r} has {r.shape}")
            if not np.allclose(r, c):
                raise ValueError(
                    f"{ax} grid of {name!r} has the same shape as {ref_name!r} "
                    f"but different coordinates (max offset "
                    f"{float(np.max(np.abs(r - c))):.4f})")


VARS3D = ("TEMP", "SALT")
VARS2D = ("SSH",)


class GodasNorm:
    """Monthly climatology and anomaly scales, fitted on TRAINING months only.

    Doc §4: "Monthly spatial climatology plus per-depth T/S anomaly scale and a
    global SSH anomaly scale are fit on training months only."

    ``clim[v]`` is (12, ...) — one spatial field per calendar month, so the
    seasonal cycle is removed per cell rather than globally.  ``scale[v]`` is
    per-depth for the 3-D variables (the thermocline's anomaly amplitude is an
    order of magnitude above the deep ocean's, so one global scale would let
    the surface dominate every loss) and a single scalar for SSH.

    ``train`` selects the training months of ``fields``; the held-out months
    must never enter these statistics.  ``check_contiguous_months`` runs on the
    training slice, because a monthly climatology fitted over a gapped series
    silently mis-assigns calendar months.
    """

    def __init__(self, fields: dict, train, eps: float = 1e-8):
        months = np.asarray(fields["months"])
        tr_months = months[train]
        check_contiguous_months(tr_months)
        cal = np.array([int(str(np.datetime64(m, "M"))[5:7]) for m in tr_months])
        self.eps = float(eps)
        self.clim, self.scale = {}, {}
        for v in VARS3D + VARS2D:
            if v not in fields:
                continue
            x = np.asarray(fields[v])[train]
            per_month = [x[cal == m] for m in range(1, 13)]
            # Land cells are NaN in every month by construction, so nanmean
            # over them is legitimately an empty slice. Expected, not a fault.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "Mean of empty slice")
                warnings.filterwarnings("ignore", "Degrees of freedom <= 0")
                allm = np.nanmean(x, axis=0)
                self.clim[v] = np.stack(
                    [np.nanmean(g, axis=0) if len(g) else allm
                     for g in per_month])
            anom = x - self.clim[v][cal - 1]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "Degrees of freedom <= 0")
                warnings.filterwarnings("ignore", "Mean of empty slice")
                if v in VARS3D:                   # per depth level
                    ax = tuple(i for i in range(anom.ndim) if i != 1)
                    self.scale[v] = np.nanstd(anom, axis=ax)
                else:                             # one scalar
                    self.scale[v] = np.nanstd(anom)
            self.scale[v] = np.maximum(self.scale[v], self.eps)

    def _cal(self, months) -> np.ndarray:
        return np.array([int(str(np.datetime64(m, "M"))[5:7])
                         for m in np.asarray(months)])

    def _bcast(self, v: str, ref: np.ndarray) -> np.ndarray:
        s = self.scale[v]
        if np.ndim(s) == 0:
            return s
        shape = [1] * ref.ndim
        shape[1] = s.size                          # depth axis
        return s.reshape(shape)

    def z(self, v: str, x: np.ndarray, months) -> np.ndarray:
        """Physical field -> anomaly z-score. NaN in, NaN out."""
        x = np.asarray(x)
        return (x - self.clim[v][self._cal(months) - 1]) / self._bcast(v, x)

    def unz(self, v: str, z: np.ndarray, months) -> np.ndarray:
        """Anomaly z-score -> physical field."""
        z = np.asarray(z)
        return z * self._bcast(v, z) + self.clim[v][self._cal(months) - 1]


# GODAS file variable -> our name, with the unit converter each one needs
_SRC = {"pottmp": ("TEMP", to_celsius), "salt": ("SALT", to_g_per_kg),
        "sshg": ("SSH", None)}
SPACE_STRIDE = 2


def load_godas(directory: str, years: tuple[int, int] | None = None,
               stride: int = SPACE_STRIDE) -> dict:
    """Load the regional subset written by experiments/13_download_godas.py.

    Returns ``{months, TEMP (T,Z,Y,X), SALT, SSH (T,Y,X), lat, lon, depth}``
    in degC / g/kg / m, with land left as NaN.

    Everything the doc §4 calls for is checked rather than assumed: the three
    variables must share one time axis and one grid, the month series must be
    contiguous and unique, and units must be declared (not inferred).
    ``stride`` subsamples space, giving the doc's 38 x 26 experiment grid.
    """
    import xarray as xr

    out, axes, units = {}, {}, {}
    for src, (name, conv) in _SRC.items():
        paths = sorted(glob.glob(os.path.join(directory, f"{src}.*.nc")))
        if years is not None:
            lo, hi = years
            paths = [p for p in paths
                     if lo <= int(os.path.basename(p).split(".")[1]) <= hi]
        if not paths:
            raise FileNotFoundError(
                f"no {src}.*.nc in {directory}"
                + (f" for years {years}" if years else "")
                + " — run experiments/13_download_godas.py")
        ds = xr.open_mfdataset(paths, combine="by_coords") if len(paths) > 1 \
            else xr.open_dataset(paths[0])
        arr = ds[src]
        axes[src] = dict(time=ds.time.values, lat=ds.lat.values, lon=ds.lon.values)
        units[src] = arr.attrs.get("units")
        sl = (slice(None),) * (arr.ndim - 2) + (slice(None, None, stride),
                                                slice(None, None, stride))
        vals = np.asarray(arr.values, dtype="float64")[sl]
        out[name] = conv(vals, units[src]) if conv is not None else vals
        if name == "TEMP":
            out["lat"] = ds.lat.values[::stride]
            out["lon"] = ds.lon.values[::stride]
            out["depth"] = ds.level.values
            out["months"] = np.asarray(ds.time.values).astype("datetime64[M]")
        ds.close()

    check_alignment(axes)
    check_contiguous_months(out["months"])
    return out
