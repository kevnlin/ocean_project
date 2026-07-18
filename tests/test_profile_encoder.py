"""Task-5 tests: physical-depth-band ProfileEncoder.

Required coverage (Monday brief):
  20-level input · 23-level input · irregular depth samples · missing shallow
  levels · missing deep levels · completely missing segment · one-profile
  batch · zero-profile batch · gradient flow — plus support-mass sanity.
"""
import numpy as np
import pytest
import torch

from ocean_tokenizer.token_api import (
    ProfileEncoder, TokenBatch, default_depth_bands,
)

D_MODEL = 32
DEPTHS_20 = np.array([5, 15, 25, 35, 45, 55, 65, 85, 105, 125, 145, 165,
                      186, 222, 267, 327, 408, 527, 707, 985], dtype="float32")
DEPTHS_23 = np.concatenate([DEPTHS_20, [1106, 1245, 1400]]).astype("float32")


def obs(depths, P=3, B=1, seed=0, month=3):
    rng = np.random.default_rng(seed)
    D = len(depths)
    return dict(
        prof=torch.tensor(rng.normal(size=(B, P, 2, D)).astype("float32")),
        lat=torch.tensor(rng.uniform(-80, 80, (B, P)).astype("float32")),
        lon=torch.tensor(rng.uniform(0, 360, (B, P)).astype("float32")),
        month=torch.full((B,), month))


# --------------------------------------------------------------- band derivation
def test_default_bands_20_level():
    bands = default_depth_bands(float(DEPTHS_20.max()))
    assert bands == [(0.0, 50.0), (50.0, 200.0), (200.0, 500.0), (500.0, 985.0)]


def test_default_bands_23_level():
    bands = default_depth_bands(float(DEPTHS_23.max()))
    assert bands == [(0.0, 50.0), (50.0, 200.0), (200.0, 500.0),
                     (500.0, 1000.0), (1000.0, 1400.0)]


# ------------------------------------------------------------- 20 vs 23 levels
def test_20_level_input():
    enc = ProfileEncoder(DEPTHS_20, d_model=D_MODEL)
    tb = enc(**obs(DEPTHS_20, P=5))
    assert enc.n_bands == 4
    assert tb.emb.shape == (1, 5 * 4, D_MODEL)
    assert tb.mask.all() and torch.isfinite(tb.emb).all()


def test_23_level_input_same_encoder_weights():
    """One encoder (built with 23-level bands) ingests both grids by passing
    ``depths`` at forward time — no divisibility constraint anywhere."""
    enc = ProfileEncoder(DEPTHS_23, d_model=D_MODEL)
    tb23 = enc(**obs(DEPTHS_23, P=4))
    assert tb23.emb.shape == (1, 4 * 5, D_MODEL) and tb23.mask.all()
    # same weights, 20-level input: deepest band (1000-1400) must mask out
    tb20 = enc(**obs(DEPTHS_20, P=4), depths=torch.tensor(DEPTHS_20))
    mask = tb20.mask.reshape(4, 5)
    assert mask[:, :4].all() and not mask[:, 4].any()


@pytest.mark.parametrize("D", [7, 11, 20, 23, 31])
def test_arbitrary_level_counts(D):
    """No dependence on D being divisible by anything."""
    depths = np.sort(np.random.default_rng(D).uniform(2, 1400, D)).astype("float32")
    enc = ProfileEncoder(depths, d_model=D_MODEL)
    tb = enc(**obs(depths, P=2))
    assert tb.emb.shape[1] == 2 * enc.n_bands
    assert torch.isfinite(tb.emb).all()


# ------------------------------------------------------------ ragged / missing
def test_irregular_depth_samples_per_profile():
    """Per-profile (B,P,D) depths: two profiles sampling different depths."""
    enc = ProfileEncoder(DEPTHS_20, d_model=D_MODEL)
    o = obs(DEPTHS_20, P=2)
    d = torch.tensor(np.stack([DEPTHS_20,
                               DEPTHS_20 * 0.9 + 3.0]).astype("float32"))[None]
    tb = enc(**o, depths=d)
    assert tb.mask.all() and torch.isfinite(tb.emb).all()


def test_missing_shallow_levels():
    enc = ProfileEncoder(DEPTHS_20, d_model=D_MODEL)
    o = obs(DEPTHS_20, P=1)
    o["prof"][0, 0, :, :5] = float("nan")          # depths 5-45 gone -> band 0 empty
    tb = enc(**o)
    mask = tb.mask.reshape(1, enc.n_bands)
    assert not mask[0, 0] and mask[0, 1:].all()


def test_missing_deep_levels():
    enc = ProfileEncoder(DEPTHS_20, d_model=D_MODEL)
    o = obs(DEPTHS_20, P=1)
    o["prof"][0, 0, :, 17:] = float("nan")         # 527/707/985 gone -> band 3 empty
    tb = enc(**o)
    mask = tb.mask.reshape(1, enc.n_bands)
    assert not mask[0, -1] and mask[0, :-1].all()


def test_completely_missing_segment_zeroed():
    enc = ProfileEncoder(DEPTHS_20, d_model=D_MODEL)
    o = obs(DEPTHS_20, P=2)
    o["prof"][0, 1, :, 13:17] = float("nan")       # 222-408 -> band 2 of profile 1
    tb = enc(**o)
    mask = tb.mask.reshape(2, enc.n_bands)
    assert not mask[1, 2]
    assert torch.equal(tb.emb.reshape(2, enc.n_bands, -1)[1, 2],
                       torch.zeros(D_MODEL))
    assert tb.support_mass.reshape(2, enc.n_bands)[1, 2] == 0.0


def test_ragged_nan_depths_ignored():
    """NaN depth = absent level (ragged padding) and never contributes."""
    enc = ProfileEncoder(DEPTHS_20, d_model=D_MODEL)
    o = obs(DEPTHS_20, P=1)
    d = torch.tensor(DEPTHS_20)[None, None].clone()
    d[0, 0, 10:] = float("nan")                    # only the top 10 levels exist
    tb = enc(**o, depths=d)
    mask = tb.mask.reshape(1, enc.n_bands)
    assert mask[0, 0] and mask[0, 1]               # 0-50, 50-200 covered
    assert not mask[0, 2] and not mask[0, 3]       # deeper bands absent


# ------------------------------------------------------------------ batch sizes
def test_one_profile_batch():
    enc = ProfileEncoder(DEPTHS_20, d_model=D_MODEL)
    tb = enc(**obs(DEPTHS_20, P=1))
    assert tb.emb.shape == (1, enc.n_bands, D_MODEL) and tb.mask.all()


def test_zero_profile_batch():
    enc = ProfileEncoder(DEPTHS_20, d_model=D_MODEL)
    tb = enc(**obs(DEPTHS_20, P=0))
    assert tb.emb.shape == (1, 0, D_MODEL)
    assert tb.support_mass is not None and tb.support_mass.shape == (1, 0)


# -------------------------------------------------------------------- gradients
def test_gradient_flow():
    enc = ProfileEncoder(DEPTHS_20, d_model=D_MODEL)
    tb = enc(**obs(DEPTHS_20, P=6))
    tb.emb.pow(2).mean().backward()
    for p in (enc.level_mlp[0].weight, enc.band_proj.weight,
              enc.coord_proj.weight):
        assert p.grad is not None and torch.isfinite(p.grad).all()
        assert p.grad.abs().sum() > 0


# ----------------------------------------------------------------- support mass
def test_support_mass_full_column_covers_bands():
    """A complete 20-level profile's band masses roughly tile the bands and
    sum to ~the full represented column span."""
    enc = ProfileEncoder(DEPTHS_20, d_model=D_MODEL)
    tb = enc(**obs(DEPTHS_20, P=1))
    mass = tb.support_mass.reshape(enc.n_bands).numpy()
    assert (mass > 0).all()
    widths = np.array([hi - lo for lo, hi in enc.bands])
    # each band's valid span cannot exceed its width by more than the
    # half-interval spill of the boundary levels
    assert (mass <= widths * 1.6).all()
    total = mass.sum()
    assert 900 < total < 1200          # ~full column (5..985 m + edge halves)


def test_support_mass_halves_when_levels_halved():
    """Dropping every other level should roughly halve nothing — mass tracks
    *represented span*, not level count: it stays ~the same (intervals widen).
    This is exactly the property MBCA needs (mass != token/level count)."""
    enc = ProfileEncoder(DEPTHS_20, d_model=D_MODEL)
    full = enc(**obs(DEPTHS_20, P=1))
    sparse_depths = DEPTHS_20[::2]
    o = obs(DEPTHS_20, P=1)
    o["prof"] = o["prof"][:, :, :, ::2]
    sparse = enc(**o, depths=torch.tensor(sparse_depths))
    m_full = full.support_mass.sum()
    m_sparse = sparse.support_mass.sum()
    assert 0.7 < float(m_sparse / m_full) < 1.3
