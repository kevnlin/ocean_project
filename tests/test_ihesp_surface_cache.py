"""Surface inputs for the iHESP cohorts: the arithmetic, on tiny arrays."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/synthetic/86_ihesp_surface_cache.py"
SPEC = importlib.util.spec_from_file_location("ihesp_surface_cache", SCRIPT)
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)


def test_box_mean_weights_by_area_and_skips_missing_cells():
    values = np.array([[1., 3., np.nan, 10.], [2., 2., 2., np.nan]])       # (levels, cells)
    area = np.array([1., 3., 5., 2.])
    box = np.array([0, 0, 0, 2])
    out = S.box_mean(values, area, box, n_boxes=3)
    assert out.shape == (2, 3)
    assert out[0, 0] == pytest.approx((1 * 1 + 3 * 3) / 4) and out[0, 2] == 10.
    assert out[1, 0] == pytest.approx(2.) and np.isnan(out[1, 2])
    assert np.isnan(out[:, 1]).all()                                       # a box with no cell


def test_one_degree_box_of_a_position():
    box = S.one_degree_box(np.array([-89.6, 0.2, 89.99]), np.array([0.4, 359.7, -0.5]))
    assert box.tolist() == [0 * 360 + 0, 90 * 360 + 359, 179 * 360 + 359]


def fields(months=24, cells=5, seed=0):
    rng = np.random.default_rng(seed)
    season = np.sin(2 * np.pi * (np.arange(months) % 12) / 12)[:, None, None]
    return 10. + 3. * season + rng.normal(size=(months, 3, cells))


def test_anomalies_use_the_training_months_climatology_and_spread_only():
    f = fields()
    month_of_year = np.arange(24) % 12
    train = np.arange(24) < 12                      # the first year trains, the second is held out
    z, clim, mean, std = S.anomaly_z(f, month_of_year, train)
    assert z.shape == f.shape and clim.shape == (12, 3, 5)
    assert np.allclose(clim, f[:12])                # one training year: its own values
    changed = f.copy(); changed[12:] += 100.
    z2, clim2, mean2, std2 = S.anomaly_z(changed, month_of_year, train)
    assert np.array_equal(clim, clim2) and np.array_equal(mean, mean2) and np.array_equal(std, std2)
    assert np.array_equal(z[:12], z2[:12]) and not np.allclose(z[12:], z2[12:])


def test_z_scores_have_unit_spread_over_the_training_months():
    f = fields(months=48, cells=400, seed=1)
    month_of_year = np.arange(48) % 12
    train = np.arange(48) < 36
    z, _, _, _ = S.anomaly_z(f, month_of_year, train)
    assert np.allclose(z[train].std(axis=(0, 2)), 1., atol=1e-6)
    assert np.allclose(z[train].mean(axis=(0, 2)), 0., atol=1e-6)


def test_a_cell_missing_in_training_stays_missing():
    f = fields()
    f[:12, 2, 3] = np.nan
    z, _, _, _ = S.anomaly_z(f, np.arange(24) % 12, np.arange(24) < 12)
    assert np.isnan(z[:, 2, 3]).all() and np.isfinite(z[:, 2, 4]).all()


def test_surface_columns_sit_where_the_model_expects_them():
    position = np.zeros((4, 56), dtype=np.float32)
    position[:, :44] = 7.
    z = np.array([[1., 2., 3.], [np.nan, 5., 6.], [7., np.nan, 9.], [1., 1., np.nan]])
    sst = np.array([14., np.nan, 25., 3.])
    x = S.surface_features(position, z, sst)
    assert x.shape == (4, 56) and x.dtype == np.float32
    assert np.array_equal(x[:, :44], position[:, :44])
    assert x[0, 44:50].tolist() == [1., 1., 2., 1., 3., 1.]          # value, present, per field
    assert x[1, 44:46].tolist() == [0., 0.] and x[2, 46:48].tolist() == [0., 0.] and x[3, 48:50].tolist() == [0., 0.]
    assert x[0, 50] == 0. and x[2, 50] == pytest.approx(1.) and x[1, 50] == 0.     # (SST - 14) / 11, 0 when missing
    assert np.array_equal(x[:, 51:], np.zeros((4, 5), dtype=np.float32))
    assert np.isfinite(x).all()


def test_a_months_fields_are_computed_once_and_then_read_back(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return np.arange(6, dtype=np.float32).reshape(3, 2)

    first = S.cached(tmp_path / "t00.npy", compute)
    again = S.cached(tmp_path / "t00.npy", compute)
    assert len(calls) == 1 and np.array_equal(first, again) and again.dtype == np.float32
    assert [p.name for p in tmp_path.iterdir()] == ["t00.npy"]       # no half-written file left behind
