"""Profiles onto the 1-degree grid, and a grid back onto points."""
import numpy as np

from ocean_tokenizer.gridded_obs import bin_profiles, sample_bilinear


# --------------------------------------------------------------------------
# bin_profiles
# --------------------------------------------------------------------------
def test_bin_profiles_averages_per_cell_and_level():
    gy = np.array([2, 2, 0]); gx = np.array([3, 3, 1])
    v = np.array([[1.0, 10.0], [3.0, np.nan], [5.0, 6.0]])      # (n=3, L=2)
    g = bin_profiles(gy, gx, v, (4, 5))
    assert g.shape == (2, 4, 5)
    assert g[0, 2, 3] == 2.0                # mean of 1 and 3
    assert g[1, 2, 3] == 10.0               # the NaN level of the second profile is skipped
    assert g[0, 0, 1] == 5.0 and g[1, 0, 1] == 6.0
    assert np.isnan(g).sum() == 2 * 4 * 5 - 4      # every other cell is empty


def test_bin_profiles_leaves_a_cell_empty_when_all_its_values_are_missing():
    g = bin_profiles(np.array([1]), np.array([1]), np.array([[np.nan, 2.0]]), (3, 3))
    assert np.isnan(g[0, 1, 1]) and g[1, 1, 1] == 2.0


def test_bin_profiles_without_profiles_is_all_empty():
    g = bin_profiles(np.array([], dtype=int), np.array([], dtype=int), np.zeros((0, 3)), (2, 2))
    assert g.shape == (3, 2, 2) and np.isnan(g).all()


# --------------------------------------------------------------------------
# sample_bilinear
# --------------------------------------------------------------------------
def _field():
    lat = np.arange(180) - 89.5
    lon = np.arange(360) + 0.5
    # two channels: one linear in latitude, one linear in longitude
    return np.stack([np.repeat(lat[:, None], 360, axis=1),
                     np.repeat(lon[None, :], 180, axis=0)])


def test_sample_bilinear_is_exact_at_cell_centres():
    out = sample_bilinear(_field(), np.array([-0.5, 10.5]), np.array([0.5, 200.5]))
    assert out.shape == (2, 2)
    assert np.allclose(out, [[-0.5, 0.5], [10.5, 200.5]])


def test_sample_bilinear_interpolates_between_centres():
    out = sample_bilinear(_field(), np.array([0.25]), np.array([10.9]))
    assert np.allclose(out, [[0.25, 10.9]])


def test_sample_bilinear_is_periodic_in_longitude():
    f = np.zeros((1, 180, 360)); f[0, :, 0] = 2.0; f[0, :, 359] = 4.0
    # 0.0 E lies midway between the centres at 359.5 E and 0.5 E
    assert np.allclose(sample_bilinear(f, np.array([0.5]), np.array([0.0])), [[3.0]])
    assert np.allclose(sample_bilinear(f, np.array([0.5]), np.array([360.0])), [[3.0]])
    assert np.allclose(sample_bilinear(f, np.array([0.5]), np.array([359.75])), [[3.5]])


def test_sample_bilinear_clamps_at_the_poles():
    out = sample_bilinear(_field(), np.array([89.9, -89.9]), np.array([50.5, 50.5]))
    assert np.allclose(out[:, 0], [89.5, -89.5])


def test_binning_then_sampling_returns_a_profile_at_its_cell_centre():
    gy, gx = np.array([100]), np.array([40])
    g = np.nan_to_num(bin_profiles(gy, gx, np.array([[7.0, -3.0]]), (180, 360)))
    assert np.allclose(sample_bilinear(g, np.array([10.5]), np.array([40.5])), [[7.0, -3.0]])
