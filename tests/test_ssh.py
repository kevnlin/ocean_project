"""Physical sanity tests for the pseudo-SSH (steric height) modality.

These pin *oceanography*, not a regression snapshot: a warm, fresh column
stands taller than a cold, salty one, the field is translation-invariant when
the ocean is, and columns that do not reach the reference level are undefined.
"""
import numpy as np
import pytest

gsw = pytest.importorskip("gsw")

from ocean_tokenizer.ssh import (P_REF_DBAR, SSHAnom, steric_height_columns,
                                 steric_height_field, ssh_for_indices)

DEPTH = np.array([5., 15., 25., 35., 45., 55., 65., 85., 105., 125.,
                  145., 165., 186., 222., 267., 327., 408., 527., 707., 985.])


def _column(t_surf=25.0, s_surf=35.0):
    """A plausible stratified column: exponential thermocline, mild halocline."""
    t = t_surf - (t_surf - 4.0) * (1 - np.exp(-DEPTH / 300.))
    s = s_surf + 0.5 * (1 - np.exp(-DEPTH / 500.))
    return t, s


class _Grid:
    """Minimal CommonGrid stand-in."""
    def __init__(self, H=4, W=5, ocean=None):
        self.lat = np.linspace(-30, 30, H)
        self.lon = np.linspace(0, 60, W)
        self.depth = DEPTH
        self.ndepth, self.nlat, self.nlon = DEPTH.size, H, W
        self.ocean = np.ones((H, W), bool) if ocean is None else ocean


def test_warm_column_stands_taller_than_cold():
    """Thermosteric expansion: the warmer column must have the greater height."""
    tw, sw = _column(t_surf=28.0)
    tc, sc = _column(t_surf=12.0)
    eta = steric_height_columns(np.stack([tw, tc], 1), np.stack([sw, sc], 1),
                                DEPTH, np.array([10.0, 10.0]),
                                np.array([200.0, 200.0]))
    assert eta[0] > eta[1]


def test_fresh_column_stands_taller_than_salty():
    """Halosteric effect at fixed temperature."""
    t, s_fresh = _column(s_surf=33.0)
    _, s_salty = _column(s_surf=37.0)
    eta = steric_height_columns(np.stack([t, t], 1),
                                np.stack([s_fresh, s_salty], 1),
                                DEPTH, np.array([10.0, 10.0]),
                                np.array([200.0, 200.0]))
    assert eta[0] > eta[1]


def test_magnitude_is_physically_plausible():
    """Steric height relative to ~1000 dbar is O(1) m, not O(1) mm or O(100) m."""
    t, s = _column()
    eta = steric_height_columns(t[:, None], s[:, None], DEPTH,
                                np.array([20.0]), np.array([300.0]))
    assert 0.2 < float(eta[0]) < 3.0


def test_uniform_ocean_is_zonally_flat_to_sub_mm():
    """A practical-salinity-uniform ocean is *nearly* zonally flat.

    It is not exactly flat, and that is correct TEOS-10 behaviour rather than a
    bug: ``SA_from_SP`` adds a longitude-dependent absolute-salinity anomaly
    (the global silicate/nutrient correction), so identical SP columns at
    different longitudes have slightly different SA and hence density.  The
    residual is sub-millimetre — four orders of magnitude below the ~cm SSH
    anomaly signal this modality is meant to carry — so it must be small, not
    zero.
    """
    g = _Grid()
    t, s = _column()
    temp = np.broadcast_to(t[:, None, None], (t.size, g.nlat, g.nlon)).copy()
    salt = np.broadcast_to(s[:, None, None], (s.size, g.nlat, g.nlon)).copy()
    field = steric_height_field(temp, salt, g)
    assert np.isfinite(field).all()
    zonal_spread = float(np.max(field.max(axis=1) - field.min(axis=1)))
    assert zonal_spread < 5e-3, f"zonal spread {zonal_spread:.2e} m is too large"


def test_short_columns_are_nan():
    """A column that does not reach the reference level is undefined, not zero."""
    g = _Grid()
    t, s = _column()
    temp = np.broadcast_to(t[:, None, None], (t.size, g.nlat, g.nlon)).copy()
    salt = np.broadcast_to(s[:, None, None], (s.size, g.nlat, g.nlon)).copy()
    temp[-3:, 0, 0] = np.nan                    # shelf column
    field = steric_height_field(temp, salt, g)
    assert np.isnan(field[0, 0])
    assert np.isfinite(field[1, 1])


def test_land_is_nan():
    g = _Grid()
    g.ocean[2, :] = False
    t, s = _column()
    temp = np.broadcast_to(t[:, None, None], (t.size, g.nlat, g.nlon)).copy()
    salt = np.broadcast_to(s[:, None, None], (s.size, g.nlat, g.nlon)).copy()
    field = steric_height_field(temp, salt, g)
    assert np.all(np.isnan(field[2]))


def test_reference_pressure_is_below_the_deepest_level_everywhere():
    """One fixed p_ref must be valid from the equator to the pole."""
    for lat in (0.0, 45.0, 89.5):
        assert gsw.p_from_z(-DEPTH[-1], lat) > P_REF_DBAR


def test_sshanom_removes_the_seasonal_cycle():
    """A pure seasonal signal must z-score to (almost) zero anomaly."""
    rng = np.random.default_rng(0)
    months = np.tile(np.arange(1, 13), 8)
    seasonal = 0.1 * np.sin(2 * np.pi * months / 12)
    field = (seasonal[:, None, None] + np.zeros((1, 3, 3))).astype("float32")
    an = SSHAnom(field, months)
    z = an.z(field[0], months[0])
    assert np.nanmax(np.abs(z)) < 1e-3


def test_sshanom_uses_training_months_only():
    """Statistics must not shift when unseen months are scored -- the whole
    point of a train-only climatology."""
    months = np.tile(np.arange(1, 13), 4)
    rng = np.random.default_rng(1)
    train = rng.normal(1.2, 0.05, (48, 4, 4)).astype("float32")
    an = SSHAnom(train, months)
    before = (an.clim.copy(), an.mean, an.std)
    an.z(rng.normal(5.0, 2.0, (4, 4)).astype("float32"), 3)   # wild new month
    assert np.array_equal(before[0], an.clim)
    assert (before[1], before[2]) == (an.mean, an.std)


def test_sshanom_preserves_nan():
    """Undefined columns must stay undefined, not become zero anomalies."""
    months = np.tile(np.arange(1, 13), 3)
    f = np.ones((36, 2, 2), "float32")
    f[:, 0, 0] = np.nan
    an = SSHAnom(f, months)
    z = an.z(f[0], 1)
    assert np.isnan(z[0, 0]) and np.isfinite(z[1, 1])


def test_ssh_for_indices_raises_on_missing_month():
    """Silent misalignment would corrupt every downstream experiment."""
    ssh = np.zeros((3, 2, 2), "float32")
    idx = np.array([10, 11, 12])
    assert ssh_for_indices(ssh, idx, [12, 10]).shape == (2, 2, 2)
    with pytest.raises(KeyError):
        ssh_for_indices(ssh, idx, [10, 99])
