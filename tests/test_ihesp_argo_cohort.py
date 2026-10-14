"""Sampling a synthetic Argo cohort from a model on its own grid; tiny fields, no iHESP data."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/synthetic/85_ihesp_argo_cohort.py"
SPEC = importlib.util.spec_from_file_location("ihesp_argo_cohort", SCRIPT)
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


def test_cells_are_distinct_ocean_cells_drawn_by_area():
    rng = np.random.default_rng(0)
    area = np.ones((40, 50))
    area[:, 25:] = 3.                       # the eastern half is three times as large
    ocean = np.ones((40, 50), bool)
    ocean[:5] = False
    cells = C.draw_cells(area, ocean, 600, rng)
    assert cells.shape == (600,) and len(np.unique(cells)) == 600
    assert ocean.ravel()[cells].all()
    east = (cells % 50 >= 25).mean()
    assert .68 < east < .82                 # 3/4 expected


def test_more_profiles_than_ocean_cells_is_refused():
    with pytest.raises(ValueError, match="ocean cells"):
        C.draw_cells(np.ones((3, 3)), np.ones((3, 3), bool), 10, np.random.default_rng(0))


def test_a_fifth_of_a_month_is_held_out_as_queries():
    held = C.draw_queries(7600, .2, np.random.default_rng(1))
    assert held.dtype == bool and held.sum() == 1520


def test_columns_are_the_models_own_values_with_land_and_sea_floor_missing():
    field = np.arange(3 * 4 * 5, dtype="float32").reshape(3, 4, 5)       # (levels, ny, nx)
    field[1, 2, 3] = 9.96921e36                                           # fill inside the ocean
    kmt = np.full((4, 5), 62)
    kmt[1, 1] = 5                                                         # sea floor above full-grid level 5
    cells = np.array([0, 1 * 5 + 1, 2 * 5 + 3])
    out = C.columns(field, kmt, full_level_index=[0, 4, 9], cells=cells)
    assert out.shape == (3, 3) and out.dtype == np.float32
    assert np.array_equal(out[0], field[:, 0, 0])
    assert np.array_equal(out[1][:2], field[:2, 1, 1]) and np.isnan(out[1][2])   # level 9 >= KMT 5
    assert np.isnan(out[2][1]) and out[2][0] == field[0, 2, 3]


def test_climatology_is_the_training_years_mean_of_the_same_cell_and_calendar_month():
    values = np.array([[[1.], [10.]], [[3.], [np.nan]], [[100.], [20.]]])   # (years, cells, levels)
    clim = C.train_mean(values, train=np.array([True, True, False]))
    assert clim.shape == (2, 1)
    assert clim[0, 0] == 2. and clim[1, 0] == 10.
    all_missing = C.train_mean(np.full((2, 1, 1), np.nan), train=np.array([True, True]))
    assert np.isnan(all_missing).all()


def test_month_of_a_time_index():
    assert C.year_month(0) == (2000, 1) and C.year_month(59) == (2004, 12) and C.year_month(71) == (2005, 12)


def test_one_degree_cell_of_a_position():
    iy, ix = C.one_degree_cell(np.array([-78.4, 0., 89.97]), np.array([0.05, 359.95, 180.]))
    assert iy.tolist() == [11, 90, 179] and ix.tolist() == [0, 359, 180]


def test_a_profile_sits_at_a_random_point_inside_its_cell_not_on_the_lattice():
    rng = np.random.default_rng(2)
    n = 5000
    lat_c = rng.uniform(-75, 85, n)
    lon_c = rng.uniform(0, 360, n)
    side_km = 11.
    area = np.full(n, (side_km * 1e5) ** 2)                     # POP areas are in cm^2
    lat, lon = C.position_in_cell(lat_c, lon_c, area, rng.uniform(-.45, .45, (n, 2)))
    assert ((lon >= 0) & (lon < 360)).all() and (np.abs(lat) <= 90).all()
    north_km = (lat - lat_c) * 111.19
    dlon = (lon - lon_c + 180) % 360 - 180
    east_km = dlon * 111.19 * np.cos(np.deg2rad(lat_c))
    assert np.abs(north_km).max() <= .45 * side_km * 1.001 and np.abs(east_km).max() <= .45 * side_km * 1.001
    assert np.abs(north_km).std() > .1 * side_km                # really spread over the cell
    # the centre is returned for a zero offset, and offsets are in units of the cell's own size
    same = C.position_in_cell(lat_c, lon_c, area, np.zeros((n, 2)))
    assert np.allclose(same[0], lat_c) and np.allclose(same[1], lon_c % 360)
    wide = C.position_in_cell(lat_c, lon_c, 4 * area, np.full((n, 2), .45))
    assert np.allclose((wide[0] - lat_c) * 111.19, 2 * .45 * side_km, rtol=1e-3)


def write_subset(folder, salt_scale):
    """A tiny copy of the layout 51_ihesp_hr_subset.py writes."""
    import netCDF4
    folder.mkdir(parents=True)
    with netCDF4.Dataset(folder / "grid.nc", "w") as g:
        g.createDimension("nlat", 3); g.createDimension("nlon", 4)
        for name, value in (("TLAT", 10.), ("TLONG", 200.), ("TAREA", 1e12)):
            g.createVariable(name, "f8", ("nlat", "nlon"))[:] = value
        g.createVariable("KMT", "i4", ("nlat", "nlon"))[:] = 62
    for variable, stored, scale in (("TEMP", 12.5, None), ("SALT", 0.0345, salt_scale)):
        with netCDF4.Dataset(folder / f"{variable}.200001.nc", "w") as nc:
            nc.source_level_index = np.arange(20)
            for dim, n in (("time", 1), ("z_t", 20), ("nlat", 3), ("nlon", 4)):
                nc.createDimension(dim, n)
            nc.createVariable("z_t", "f4", ("z_t",))[:] = C.LEVELS * 100.
            v = nc.createVariable(variable, "f4", ("time", "z_t", "nlat", "nlon"), fill_value=np.float32(9.96921e33))
            if scale is not None:
                v.scale_factor = np.float32(scale)
            v.set_auto_maskandscale(False)
            data = np.full((1, 20, 3, 4), stored, "float32")
            data[0, :, 0, 0] = 9.96921e33                                  # land
            v[:] = data


def test_salinity_is_read_in_grams_per_kilogram(tmp_path):
    # POP stores salinity in g/g with scale_factor 1000; the cohort must hold g/kg
    write_subset(tmp_path / "data/ihesp/hr_cohort_levels", salt_scale=1000.)
    source = C.HighResolution(tmp_path)
    salt, temp = source.field("SALT", 0), source.field("TEMP", 0)
    assert salt.shape == (20, 3, 4) and salt.dtype == np.float32
    assert salt[0, 1, 1] == pytest.approx(34.5, rel=1e-6) and temp[0, 1, 1] == pytest.approx(12.5)
    cells = np.array([0, 5])
    out = C.columns(salt, source.kmt, source.full_level_index, cells)
    assert np.isnan(out[0]).all() and out[1, 0] == pytest.approx(34.5, rel=1e-6)   # land stays missing


def test_a_salinity_file_without_its_scale_is_refused(tmp_path):
    write_subset(tmp_path / "data/ihesp/hr_cohort_levels", salt_scale=None)
    with pytest.raises(ValueError, match="g/kg"):
        C.HighResolution(tmp_path).field("SALT", 0)
