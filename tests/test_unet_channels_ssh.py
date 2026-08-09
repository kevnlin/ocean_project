"""The `ssh` cfg token must be strictly additive.

The certified U-Net checkpoints (`audit_depthwise_e40`) were trained with
c_in = 10 on `profiles_woa_surf`.  If adding SSH support shifted the channel
count or the channel *order* of any pre-existing config, those checkpoints
would silently load against mismatched inputs and every historical number would
become unreproducible.  These tests pin that it cannot happen.
"""
import numpy as np
import pytest

from ocean_tokenizer import baselines as B


class _Grid:
    def __init__(self, D=4, H=8, W=10):
        self.lat = np.linspace(-40, 40, H)
        self.lon = np.linspace(0, 350, W)
        self.depth = np.linspace(5, 900, D)
        self.ndepth, self.nlat, self.nlon = D, H, W
        self.ocean = np.ones((H, W), bool)
        self.ocean[0, 0] = False


class _Norm:
    """Identity normaliser with the AnomNorm interface."""
    def z3d(self, v, arr, month=None):
        return np.asarray(arr, dtype="float32")

    def zsurf(self, v, arr, month=None):
        return np.asarray(arr, dtype="float32")


def _sample(grid, with_ssh=False):
    D, H, W = grid.ndepth, grid.nlat, grid.nlon
    rng = np.random.default_rng(0)
    obs = np.full((D, H, W), np.nan, "float32")
    obs[:, 2, 3] = 1.0
    s = {
        "month": 5,
        "gt": {v: rng.normal(size=(D, H, W)).astype("float32") for v in B.VARS},
        "woa": {v: rng.normal(size=(D, H, W)).astype("float32") for v in B.VARS},
        "obs": {v: obs.copy() for v in B.VARS},
        "surf": {sv: rng.normal(size=(H, W)).astype("float32")
                 for sv in ("SST", "SSS")},
    }
    if with_ssh:
        z = rng.normal(size=(H, W)).astype("float32")
        z[0, 1] = np.nan                     # an undefined (shelf) column
        s["ssh_z"] = z
    return s


CFGS = {
    "profiles_only": ("profiles",),
    "woa_only": ("woa",),
    "profiles_woa": ("profiles", "woa"),
    "profiles_woa_surf": ("profiles", "woa", "surf"),
}


@pytest.mark.parametrize("name,cfg", list(CFGS.items()))
def test_existing_configs_are_bit_identical_with_ssh_support(name, cfg):
    """A sample that carries an ssh_z field must not change any cfg that does
    not ask for 'ssh'."""
    grid = _Grid()
    norm = _Norm()
    without = B._unet_channels(_sample(grid, with_ssh=False), grid, norm, cfg)
    withssh = B._unet_channels(_sample(grid, with_ssh=True), grid, norm, cfg)
    assert without.shape == withssh.shape
    assert np.array_equal(without, withssh)


def test_profiles_woa_surf_still_has_ten_channels():
    """The width the certified checkpoint was trained at."""
    grid = _Grid()
    X = B._unet_channels(_sample(grid), grid, _Norm(), ("profiles", "woa", "surf"))
    assert X.shape[1] == 10


def test_ssh_adds_exactly_two_channels_appended_at_the_end():
    grid = _Grid()
    norm = _Norm()
    s = _sample(grid, with_ssh=True)
    base = B._unet_channels(s, grid, norm, ("profiles", "woa", "surf"))
    ext = B._unet_channels(s, grid, norm, ("profiles", "woa", "surf", "ssh"))
    assert ext.shape[1] == base.shape[1] + 2
    # every pre-existing channel keeps its index, except the ocean-mask channel
    # which stays last in both
    assert np.array_equal(base[:, :-1], ext[:, :-3])
    assert np.array_equal(base[:, -1], ext[:, -1])


def test_missing_ssh_is_a_zero_channel_not_a_crash():
    """A month with no cached SSH must degrade gracefully, like a missing
    surface field, rather than raise."""
    grid = _Grid()
    X = B._unet_channels(_sample(grid, with_ssh=False), grid, _Norm(),
                         ("profiles", "woa", "surf", "ssh"))
    assert X.shape[1] == 12
    assert np.all(X[:, -3] == 0.0) and np.all(X[:, -2] == 0.0)


def test_undefined_ssh_cells_become_zero_with_a_zero_finite_flag():
    """NaN means missing, never an observed anomaly of zero."""
    grid = _Grid()
    s = _sample(grid, with_ssh=True)
    X = B._unet_channels(s, grid, _Norm(), ("profiles", "woa", "surf", "ssh"))
    val, flag = X[:, -3], X[:, -2]
    assert np.all(val[:, 0, 1] == 0.0)
    assert np.all(flag[:, 0, 1] == 0.0)
    assert np.all(flag[:, 1, 1] == 1.0)
