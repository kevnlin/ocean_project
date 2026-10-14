"""Readers for the iHESP CESM1.3 historical ocean fields kept on this server.

Two sources, one interface (``lat``, ``lon``, ``area``, ``kmt``,
``full_level_index`` and ``field(variable, t)`` for month ``t`` counted from
2000-01), at the 20 depth levels of the synthetic Argo cohorts:

  HighResolution   the 0.1-degree run, from the per-month subset files of
                   ``experiments/data/51_ihesp_hr_subset.py``
  LowResolution    the 1-degree run, from the 2000-2005 decade files

Two traps of these files are handled here, once. Salinity is stored in g/g
with ``scale_factor = 1000`` and is returned in g/kg. Fill values are returned
as stored (far above ``FILL``); in the 0.1-degree files a third of the grid lies
in land blocks the model eliminated, where every field reads -1, so land must be
masked with ``kmt``, never with the fill value alone.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

#: the synthetic cohorts' levels, metres (41_synth_argo_cohort.py)
LEVELS = np.array([5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 85.0, 105.0, 125.0,
                   145.0, 165.1, 186.3, 222.6, 267.7, 326.9, 408.8, 527.7,
                   707.6, 984.7])
N_MONTHS = 72
FILL = 1e30


def year_month(t):
    """Calendar year and month (1-12) of a month index counted from 2000-01."""
    return 2000 + t // 12, t % 12 + 1


def stored_values(variable_handle, name, index):
    """Values of a netCDF variable in physical units, fill values left as they are stored.

    POP writes salinity in g/g with ``scale_factor = 1000``. The library's own
    unpacking would turn the fill value into a mask, so the factor is applied
    here; the scaled fill is still far above ``FILL``. Salinity without the
    factor would come out in g/g and is refused.
    """
    variable_handle.set_auto_maskandscale(False)
    values = variable_handle[index]
    scale = float(getattr(variable_handle, "scale_factor", 1.0))
    if name == "SALT" and scale != 1000.0:
        raise ValueError("SALT carries no scale_factor of 1000: the values would not be in g/kg")
    return values if scale == 1.0 else values * np.float32(scale)


class HighResolution:
    """Per-month files of 51_ihesp_hr_subset.py, already at the cohort's levels."""
    name, describe = "ihesp_hr_uniform", "iHESP CESM1.3 ne120_t12 (0.1-degree ocean), member 002"

    def __init__(self, root):
        import netCDF4
        self.folder = Path(root) / "data/ihesp/hr_cohort_levels"
        with netCDF4.Dataset(self.folder / "grid.nc") as g:
            for key in g.variables:
                g[key].set_auto_maskandscale(False)
            self.lat, self.lon = g["TLAT"][:], g["TLONG"][:]
            self.area, self.kmt = g["TAREA"][:], g["KMT"][:]
        with netCDF4.Dataset(self.folder / "TEMP.200001.nc") as nc:
            self.full_level_index = [int(k) for k in np.atleast_1d(nc.source_level_index)]
            depth = nc["z_t"][:] / 100.
        if not np.allclose(depth, LEVELS, atol=0.2):
            raise ValueError("the subset's levels are not the cohort's")

    def field(self, variable, t):
        import netCDF4
        year, month = year_month(t)
        with netCDF4.Dataset(self.folder / f"{variable}.{year}{month:02d}.nc") as nc:
            return stored_values(nc[variable], variable, 0)


class LowResolution:
    """The 2000-2005 decade files of the 1-degree run, 60 levels."""
    name, describe = "ihesp_lr_uniform", "iHESP CESM1.3 ne30_g16 (1-degree ocean), member 002"

    def __init__(self, root):
        import netCDF4
        self.files = {}
        for variable in ("TEMP", "SALT"):
            found = glob.glob(str(Path(root) / f"data/ihesp/lr/*.pop.h.{variable}.200001-200512.nc"))
            if len(found) != 1:
                raise FileNotFoundError(f"expected one 2000-2005 {variable} file, found {len(found)}")
            self.files[variable] = found[0]
        with netCDF4.Dataset(self.files["TEMP"]) as nc:
            for key in ("TLAT", "TLONG", "TAREA", "KMT", "z_t"):
                nc[key].set_auto_maskandscale(False)
            self.lat, self.lon = nc["TLAT"][:], nc["TLONG"][:]
            self.area, self.kmt = nc["TAREA"][:], nc["KMT"][:]
            depth = nc["z_t"][:] / 100.
            if nc["TEMP"].shape[0] != N_MONTHS:
                raise ValueError("the decade file does not hold 72 months")
        self.full_level_index = [int(np.abs(depth - d).argmin()) for d in LEVELS]
        if not np.allclose(depth[self.full_level_index], LEVELS, atol=0.2):
            raise ValueError("the cohort's levels are not levels of this grid")

    def field(self, variable, t):
        import netCDF4
        with netCDF4.Dataset(self.files[variable]) as nc:
            return stored_values(nc[variable], variable, (t, self.full_level_index))


SOURCES = {"hr": HighResolution, "lr": LowResolution}

