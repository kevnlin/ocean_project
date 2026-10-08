"""Check satellite sampling at arbitrary Argo positions and longitude seams."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/synthetic/50_prepare_matched_satellites.py"
SPEC = importlib.util.spec_from_file_location("matched_satellites", SCRIPT)
SATELLITES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SATELLITES)


def test_bilinear_profile_positions_reproduce_affine_field():
    lat, lon = np.array([-1., 0., 1.]), np.array([0., 90., 180., 270.])
    field = (2 * lat[:, None] + 3 * lon[None, :])[None]
    query_lat, query_lon = np.array([-.75, .25]), np.array([22.5, 135.])
    actual = SATELLITES.sample_grid(field, query_lat, query_lon, lat, lon)
    np.testing.assert_allclose(actual[:, 0], 2 * query_lat + 3 * query_lon)


def test_longitude_wrap_interpolates_across_the_dateline():
    lat, lon = np.array([-1., 1.]), np.array([45., 135., 225., 315.])
    field = np.broadcast_to(np.array([10., 20., 30., 40.]), (2, 4))[None]
    actual = SATELLITES.sample_grid(field, np.zeros(3), np.array([0., 360., -360.]), lat, lon)
    np.testing.assert_allclose(actual[:, 0], 25.)


def test_missing_corners_do_not_create_surface_observations():
    lat, lon = np.array([-1., 1.]), np.array([0., 180.])
    field = np.array([[[2., np.nan], [4., 6.]]])
    assert np.isnan(SATELLITES.sample_grid(field, np.array([0.]), np.array([90.]), lat, lon)).all()


def test_outside_latitude_grid_is_rejected():
    field = np.ones((3, 2, 2))
    with pytest.raises(ValueError, match="outside"):
        SATELLITES.sample_grid(field, np.array([-2.]), np.array([90.]),
                               np.array([-1., 1.]), np.array([0., 180.]))
