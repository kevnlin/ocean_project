"""The GLORYS12 reanalysis as a third OSSE simulator (spec ``2026-10-21-glorys-osse.md``).

Copernicus ``GLOBAL_MULTIYEAR_PHY_001_030``, dataset ``cmems_mod_glo_phy_my_0.083deg_P1M-m``:
monthly ``thetao``, ``so`` and ``zos`` on a regular 1/12-degree grid (2041 latitudes -80..90,
4320 longitudes -180..179.917), int16 with a scale and an offset, fill -32767. The first 36
of its 50 levels (0.49-1062.4 m) are downloaded; the cohort's 20 levels (5-984.7 m) are
linear interpolations of each column between its two bracketing GLORYS levels, missing where
either bracket is missing (at and below the sea floor).

Calendar. The runner and the scene classes count month index 0 as 2000-01 with training up
to 2003-12 and selection in 2004-2005, so GLORYS years are relabelled by -13: GLORYS 1997-2016
= cohort 1984-2003 (month indices -192..47, training), 2017-2018 = 2004-2005 (48..71,
selection), 2019-2020 = 2006-2007 (72..95, the sealed test).

Per processed month only three products stay on disk (``data/glorys``): the 1-degree box
means (``boxes``), the 20-level columns at the union of the cells drawn for that calendar
month in every cohort year (``columns``: the climatology of a drawn cell is the mean of its
training-year values in these files), and the grid/mask file written from the first month.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

#: the synthetic cohorts' levels, metres (41_synth_argo_cohort.py)
LEVELS = np.array([5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 85.0, 105.0, 125.0,
                   145.0, 165.1, 186.3, 222.6, 267.7, 326.9, 408.8, 527.7,
                   707.6, 984.7])
DATASET = "cmems_mod_glo_phy_my_0.083deg_P1M-m"
VARIABLES = ("thetao", "so", "zos")
DESCRIBE = "Copernicus GLORYS12V1 reanalysis (GLOBAL_MULTIYEAR_PHY_001_030, 1/12-degree, monthly means)"
N_LAT, N_LON = 2041, 4320
STEP = 1.0 / 12.0
FILL_INT16 = -32767
SEED = 20261021
N_PER_MONTH = 7600
QUERY_FRAC = 0.2
#: a profile lies within this fraction of the cell's width from its centre (85's IN_CELL)
IN_CELL = 0.45
#: cohort year = GLORYS year - OFFSET_YEARS
OFFSET_YEARS = 13
TRAIN_MONTHS = tuple(range(-192, 48))          # GLORYS 1997-01 .. 2016-12
SELECTION_MONTHS = tuple(range(48, 72))        # GLORYS 2017-01 .. 2018-12
TEST_MONTHS = tuple(range(72, 96))             # GLORYS 2019-01 .. 2020-12
COHORT_MONTHS = TRAIN_MONTHS + SELECTION_MONTHS
ALL_MONTHS = COHORT_MONTHS + TEST_MONTHS
SPLITS = {"train": (1984, 2003), "validation": (2004, 2004), "development": (2005, 2005)}
N_BOXES = 180 * 360


def year_month(t):
    """Cohort calendar year and month (1-12) of a month index counted from 2000-01."""
    return 2000 + t // 12, t % 12 + 1


def glorys_year_month(t):
    """GLORYS calendar year and month of a cohort month index."""
    year, month = year_month(int(t))
    return year + OFFSET_YEARS, month


def month_index(glorys_year, glorys_month):
    return 12 * (int(glorys_year) - 2000 - OFFSET_YEARS) + int(glorys_month) - 1


def stamp(t):
    """``YYYYMM`` of a cohort month in the GLORYS calendar: the file names."""
    year, month = glorys_year_month(t)
    return f"{year}{month:02d}"


class Folders:
    def __init__(self, root):
        self.root = Path(root)
        self.base = self.root / "data/glorys"
        self.raw, self.boxes, self.columns = self.base / "raw", self.base / "boxes", self.base / "columns"
        self.grid, self.draws = self.base / "grid.npz", self.base / "draws.npz"

    def raw_file(self, t):
        return self.raw / f"glorys_{stamp(t)}.nc"

    def boxes_file(self, t):
        return self.boxes / f"boxes_{stamp(t)}.npz"

    def columns_file(self, t):
        return self.columns / f"columns_{stamp(t)}.npz"


# ----------------------------------------------------------------------------- the raw month
def unpack(variable, index=0):
    """A packed int16 variable as float32 with NaN at the fill, without the library's float64 masking."""
    variable.set_auto_maskandscale(False)
    raw = np.asarray(variable[index])
    fill = int(getattr(variable, "_FillValue", FILL_INT16))
    out = raw.astype(np.float32) * np.float32(getattr(variable, "scale_factor", 1.0)) + np.float32(getattr(variable, "add_offset", 0.0))
    out[raw == fill] = np.nan
    return out


def interpolation_weights(depth, levels=LEVELS):
    """For each cohort level the index ``k`` of the GLORYS level above it and the weight of ``k + 1``."""
    depth = np.asarray(depth, dtype=np.float64)
    k = np.searchsorted(depth, levels, side="right") - 1
    if (k < 0).any() or (k + 1 >= len(depth)).any():
        raise ValueError("a cohort level is not bracketed by two GLORYS levels")
    w = (levels - depth[k]) / (depth[k + 1] - depth[k])
    return k, w.astype(np.float32)


def to_cohort_levels(field, depth, levels=LEVELS):
    """``(36, ny, nx)`` -> ``(20, ny, nx)`` linear interpolation in depth; NaN where a bracket is NaN."""
    k, w = interpolation_weights(depth, levels)
    out = np.empty((len(levels),) + field.shape[1:], np.float32)
    for i in range(len(levels)):
        out[i] = field[k[i]] * (np.float32(1) - w[i]) + field[k[i] + 1] * w[i]
    return out


def read_month(path):
    """``(thetao, so)`` at the cohort's 20 levels, ``zos``, and the grid of one raw month file."""
    import netCDF4
    with netCDF4.Dataset(path) as nc:
        if nc["thetao"].shape[0] != 1:
            raise ValueError(f"{path}: not one month")
        depth = np.asarray(nc["depth"][:], dtype=np.float64)
        lat, lon = np.asarray(nc["latitude"][:], np.float64), np.asarray(nc["longitude"][:], np.float64)
        if lat.shape != (N_LAT,) or lon.shape != (N_LON,):
            raise ValueError(f"{path}: grid {lat.shape} x {lon.shape} is not the 1/12-degree grid")
        if nc["so"].units not in ("1e-3", "psu", "PSU"):
            raise ValueError(f"{path}: salinity units {nc['so'].units}")
        temp = to_cohort_levels(unpack(nc["thetao"]), depth)
        salt = to_cohort_levels(unpack(nc["so"]), depth)
        zos = unpack(nc["zos"])
        hours = float(nc["time"][0])
    return temp, salt, zos, {"depth": depth, "lat": lat, "lon": lon, "time_hours_since_1950": hours}


# ----------------------------------------------------------------------------- the grid
def cell_area(lat):
    """Relative area of each cell of the regular grid, ``(N_LAT, N_LON)``: cos(latitude)."""
    return np.repeat(np.maximum(np.cos(np.deg2rad(np.asarray(lat, dtype=np.float64))), 0.0)[:, None], N_LON, axis=1)


def write_grid(folders, temp, salt, grid):
    """The grid file from the first processed month: lat, lon, area and the 20-level wet mask."""
    wet = np.isfinite(temp) & np.isfinite(salt)
    ocean = wet[0]
    payload = dict(lat=grid["lat"], lon=grid["lon"], depth=grid["depth"], levels=LEVELS, wet=np.packbits(wet, axis=-1),
                   wet_shape=np.array(wet.shape), ocean=ocean, area=cell_area(grid["lat"]).astype(np.float32),
                   metadata=json.dumps({"source": DESCRIBE, "ocean": "cells with a value at 5 m (both brackets wet)",
                                        "wet": "packed bits, the finite mask of the 20 interpolated levels"}))
    folders.grid.parent.mkdir(parents=True, exist_ok=True)
    temporary = folders.grid.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, folders.grid)


class Grid:
    def __init__(self, folders):
        with np.load(folders.grid, allow_pickle=False) as g:
            self.lat, self.lon, self.depth = g["lat"], g["lon"], g["depth"]
            shape = tuple(int(v) for v in g["wet_shape"])
            self.wet = np.unpackbits(g["wet"], axis=-1, count=shape[-1]).astype(bool).reshape(shape)
            self.ocean, self.area = g["ocean"], g["area"].astype(np.float64)
        self.shape = self.ocean.shape
        self.lat2d = np.repeat(self.lat[:, None], N_LON, axis=1)
        self.lon2d = np.repeat(self.lon[None, :], N_LAT, axis=0)

    def check(self, temp, salt, stamp_text):
        wet = np.isfinite(temp) & np.isfinite(salt)
        if not np.array_equal(wet, self.wet):
            raise ValueError(f"{stamp_text}: the wet mask differs from the grid file's ({int((wet != self.wet).sum())} cells)")


# ----------------------------------------------------------------------------- 85's draws
def draw_cells(area, ocean, n, rng):
    """Flat indices of ``n`` distinct ocean cells, drawn with probability ~ area."""
    candidates = np.flatnonzero(np.asarray(ocean).ravel())
    if n > candidates.size:
        raise ValueError(f"{n} profiles asked of {candidates.size} ocean cells")
    weight = np.asarray(area, dtype=np.float64).ravel()[candidates]
    return np.sort(rng.choice(candidates, n, replace=False, p=weight / weight.sum()))


def draw_queries(n, fraction, rng):
    held = np.zeros(n, dtype=bool)
    held[rng.choice(n, int(round(fraction * n)), replace=False)] = True
    return held


def draw_month(rng, grid):
    cells = draw_cells(grid.area, grid.ocean, N_PER_MONTH, rng)
    held = draw_queries(N_PER_MONTH, QUERY_FRAC, rng)
    offset = rng.uniform(-IN_CELL, IN_CELL, (N_PER_MONTH, 2))
    return cells, held, offset


def compute_draws(grid):
    """Cells, queries and offsets of every cohort and test month.

    Cohort months (-192..71) come from one generator consumed in month order, as 85; each test
    month from its own ``default_rng([SEED, t + 10000])``, as 138.
    """
    out = {}
    rng = np.random.default_rng(SEED)
    for t in COHORT_MONTHS:
        out[t] = draw_month(rng, grid)
    for t in TEST_MONTHS:
        out[t] = draw_month(np.random.default_rng([SEED, t + 10_000]), grid)
    return out


def draws(folders, grid=None):
    """``compute_draws`` once, cached in ``draws.npz`` and re-verified on a sample when read."""
    if folders.draws.exists():
        with np.load(folders.draws, allow_pickle=False) as d:
            if int(d["seed"]) != SEED:
                raise ValueError("draws.npz was made with another seed")
            months, cells, held, offset = d["months"], d["cells"], d["held"], d["offset"]
            out = {int(t): (cells[i], held[i], offset[i]) for i, t in enumerate(months)}
        if grid is not None:          # the first cohort month and one test month must come back
            for t in (COHORT_MONTHS[0], TEST_MONTHS[0]):
                rng = np.random.default_rng(SEED) if t == COHORT_MONTHS[0] else np.random.default_rng([SEED, t + 10_000])
                cells, held, offset = draw_month(rng, grid)
                if not (np.array_equal(cells, out[t][0]) and np.array_equal(held, out[t][1]) and np.allclose(offset, out[t][2])):
                    raise ValueError("draws.npz does not reproduce the generator's draws")
        return out
    if grid is None:
        grid = Grid(folders)
    out = compute_draws(grid)
    months = np.array(ALL_MONTHS)
    temporary = folders.draws.with_suffix(".tmp.npz")
    np.savez(temporary, seed=SEED, months=months, cells=np.stack([out[t][0] for t in months]).astype(np.int64),
             held=np.stack([out[t][1] for t in months]), offset=np.stack([out[t][2] for t in months]),
             metadata=json.dumps({"cohort_months": "one default_rng(SEED) in month order -192..71 (85)",
                                  "test_months": "default_rng([SEED, t + 10000]) per month (138)",
                                  "per_month": [N_PER_MONTH, "cells ~ area without replacement", "20 % queries",
                                                "uniform offset in (-0.45, 0.45) cell widths"]}))
    os.replace(temporary, folders.draws)
    return out


def union_cells(all_draws, calendar_month):
    """Sorted union of the cells drawn in every year of one calendar month (0-11)."""
    return np.unique(np.concatenate([all_draws[t][0] for t in ALL_MONTHS if t % 12 == calendar_month]))


def position_in_cell(lat_centre, lon_centre, offset):
    """The centre moved by ``offset`` cell widths north and east; longitude in 0-360."""
    lat = np.clip(np.asarray(lat_centre, np.float64) + offset[:, 0] * STEP, -90.0, 90.0)
    lon = (np.asarray(lon_centre, np.float64) + offset[:, 1] * STEP) % 360.0
    return lat, lon


def one_degree_cell(lat, lon):
    """Row and column of the 1-degree cell, the cohorts' ``grid_y`` / ``grid_x``."""
    iy = np.clip(np.floor(np.asarray(lat) + 90.0).astype(int), 0, 179)
    ix = np.floor(np.asarray(lon) % 360.0).astype(int) % 360
    return iy, ix


def one_degree_box(lat, lon):
    iy, ix = one_degree_cell(lat, lon)
    return iy * 360 + ix


# ----------------------------------------------------------------------------- 86's box means
def box_mean(values, weight, box, n_boxes=N_BOXES):
    """``(L, n)`` cell values -> ``(L, n_boxes)`` area-weighted means; NaN cells skipped, empty boxes NaN."""
    out = np.full((values.shape[0], n_boxes), np.nan)
    for level, v in enumerate(values):
        ok = np.isfinite(v)
        total = np.bincount(box[ok], weights=weight[ok] * v[ok], minlength=n_boxes)
        area = np.bincount(box[ok], weights=weight[ok], minlength=n_boxes)
        out[level] = np.where(area > 0, total / np.maximum(area, 1e-300), np.nan)
    return out


def train_mean(values):
    """``(years, cells, L)`` -> ``(cells, L)`` mean over the years, NaN where none is finite."""
    count = np.isfinite(values).sum(0)
    total = np.where(np.isfinite(values), values, 0.).sum(0, dtype=np.float64)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan).astype(np.float32)


def climatology_at(folders, calendar_month, cells):
    """1997-2016 monthly climatology of T and S at ``cells`` from the union files of a calendar month."""
    months = [t for t in TRAIN_MONTHS if t % 12 == calendar_month]
    temp, salt = [], []
    for t in months:
        with np.load(folders.columns_file(t), allow_pickle=False) as z:
            where = np.searchsorted(z["cells"], cells)
            if (where >= len(z["cells"])).any() or not np.array_equal(z["cells"][np.minimum(where, len(z["cells"]) - 1)], cells):
                raise ValueError(f"{folders.columns_file(t)}: a wanted cell is not in the union")
            temp.append(z["TEMP"][where])
            salt.append(z["SALT"][where])
    return train_mean(np.stack(temp)), train_mean(np.stack(salt))


def columns_at(folders, t, cells):
    """T and S columns of cohort month ``t`` at ``cells`` from its union file."""
    with np.load(folders.columns_file(t), allow_pickle=False) as z:
        where = np.searchsorted(z["cells"], cells)
        if (where >= len(z["cells"])).any() or not np.array_equal(z["cells"][np.minimum(where, len(z["cells"]) - 1)], cells):
            raise ValueError(f"{folders.columns_file(t)}: a wanted cell is not in the union")
        return z["TEMP"][where], z["SALT"][where]
