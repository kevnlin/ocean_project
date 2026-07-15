"""Unit tests for the Week-2 shared-latent token API.

Contract points under test (ICLR-plan Week 2):
  * variable profile counts (incl. zero)
  * missing modalities (any subset, incl. none)
  * coordinate consistency between encoders and the query side
  * masks: padding- and NaN-invariance of the fused output
  * normalisation / NaN handling end-to-end via the pipeline bridge
plus permutation invariance and gradient flow.
"""
import itertools

import numpy as np
import pytest
import torch

from ocean_tokenizer.token_api import (
    MODALITIES, N_COORD_FEATS, TokenBatch, coord_features,
    GridPatchEncoder, ProfileEncoder, PointEncoder,
    SharedLatentStub, sample_to_obs, make_query_coords, build_stub,
)

D_MODEL = 32
DEPTHS = np.array([5, 15, 25, 35, 45, 55, 65, 85, 105, 125, 145, 165,
                   186, 222, 267, 327, 408, 527, 707, 985], dtype="float32")
D = len(DEPTHS)


def make_stub(seed=0, patch=(4, 6)):
    torch.manual_seed(seed)
    return SharedLatentStub({
        "profiles": ProfileEncoder(DEPTHS, c_vars=2, d_model=D_MODEL),
        "surf": GridPatchEncoder(2, d_model=D_MODEL, patch=patch,
                                 modality="surf_grid"),
        "woa": GridPatchEncoder(2, d_model=D_MODEL, patch=patch,
                                modality="woa_grid"),
        "points": PointEncoder(d_model=D_MODEL),
    }, d_model=D_MODEL)


def profile_obs(P, B=1, seed=0, month=3):
    rng = np.random.default_rng(seed)
    return dict(
        prof=torch.tensor(rng.normal(size=(B, P, 2, D)).astype("float32")),
        lat=torch.tensor(rng.uniform(-80, 80, (B, P)).astype("float32")),
        lon=torch.tensor(rng.uniform(0, 360, (B, P)).astype("float32")),
        month=torch.full((B,), month))


def surf_obs(B=1, H=12, W=24, seed=1, month=3):
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(B, 2, H, W)).astype("float32")
    f[:, :, :2, :3] = np.nan                       # a land corner
    return dict(field=torch.tensor(f),
                lat=torch.linspace(-80, 80, H),
                lon=torch.linspace(0, 345, W),
                month=torch.full((B,), month))


def queries(Q=17, B=1, seed=2):
    rng = np.random.default_rng(seed)
    q = np.stack([rng.uniform(-80, 80, Q), rng.uniform(0, 360, Q),
                  rng.uniform(0, 985, Q), np.full(Q, 3.0)],
                 axis=-1).astype("float32")
    return torch.tensor(np.repeat(q[None], B, axis=0))


# ---------------------------------------------------------------- coordinates
def test_coord_features_shape_and_determinism():
    q = queries(Q=50)
    f1, f2 = coord_features(q), coord_features(q)
    assert f1.shape == (1, 50, N_COORD_FEATS)
    assert torch.equal(f1, f2)
    assert torch.isfinite(f1).all()


def test_coordinate_consistency_across_encoders():
    """The same physical location yields identical coord features regardless
    of which modality carries it, and TokenBatch.coord stores it faithfully."""
    lat, lon, month = 12.5, 200.0, 7
    prof_enc = ProfileEncoder(DEPTHS, d_model=D_MODEL, n_segments=4)
    pt_enc = PointEncoder(d_model=D_MODEL)

    tbp = prof_enc(**dict(prof=torch.zeros(1, 1, 2, D),
                          lat=torch.tensor([[lat]]), lon=torch.tensor([[lon]]),
                          month=torch.tensor([month])))
    seg_depth = DEPTHS.reshape(4, -1).mean(1)
    expect = np.stack([np.full(4, lat), np.full(4, lon), seg_depth,
                       np.full(4, month)], -1)
    np.testing.assert_allclose(tbp.coord[0].numpy(), expect, rtol=1e-5)

    depth = float(seg_depth[0])
    tbq = pt_enc(values=torch.zeros(1, 1), var_id=torch.zeros(1, 1, dtype=torch.long),
                 lat=torch.tensor([[lat]]), lon=torch.tensor([[lon]]),
                 depth=torch.tensor([[depth]]), month=torch.tensor([month]))
    # identical (lat, lon, depth, month) -> identical coordinate features
    assert torch.allclose(coord_features(tbp.coord[0, 0]),
                          coord_features(tbq.coord[0, 0]))


# ---------------------------------------------------------- variable P counts
@pytest.mark.parametrize("P", [0, 1, 7, 130])
def test_variable_profile_counts(P):
    enc = ProfileEncoder(DEPTHS, d_model=D_MODEL, n_segments=4)
    tb = enc(**profile_obs(P))
    assert tb.emb.shape == (1, P * 4, D_MODEL)
    assert tb.mask.shape == (1, P * 4)
    if P:
        assert tb.mask.all()
        assert torch.isfinite(tb.emb).all()


def test_stub_forward_any_profile_count():
    model = make_stub()
    q = queries()
    outs = [model({"profiles": profile_obs(P)}, q) for P in (0, 1, 1500)]
    for o in outs:
        assert o.shape == (1, 17, 2) and torch.isfinite(o).all()


# ------------------------------------------------------------------ modalities
def test_missing_modalities_any_subset():
    model = make_stub()
    q = queries()
    full = {"profiles": profile_obs(40), "surf": surf_obs(),
            "points": dict(values=torch.randn(1, 9),
                           var_id=torch.randint(0, 4, (1, 9)),
                           lat=torch.rand(1, 9) * 100 - 50,
                           lon=torch.rand(1, 9) * 360,
                           depth=torch.rand(1, 9) * 900,
                           month=torch.tensor([3]))}
    for r in range(len(full) + 1):
        for keys in itertools.combinations(full, r):
            out = model({k: full[k] for k in keys}, q)
            assert out.shape == (1, 17, 2) and torch.isfinite(out).all(), keys


def test_modality_ids_stamped():
    model = make_stub()
    tb = model.encode({"profiles": profile_obs(3), "surf": surf_obs()},
                      batch=1)
    ids = set(tb.modality.unique().tolist())
    assert ids == {MODALITIES["profile"], MODALITIES["surf_grid"]}


# ----------------------------------------------------------------------- masks
def test_padding_invariance():
    """Padded (invalid) profile slots must not change the fused output."""
    model = make_stub()
    q = queries()
    obs5 = profile_obs(5, seed=7)
    out5 = model({"profiles": obs5}, q)

    pad = 3
    obs8 = {
        "prof": torch.cat([obs5["prof"],
                           torch.full((1, pad, 2, D), float("nan"))], 1),
        "lat": torch.cat([obs5["lat"], torch.full((1, pad), float("nan"))], 1),
        "lon": torch.cat([obs5["lon"], torch.full((1, pad), float("nan"))], 1),
        "month": obs5["month"],
        "valid": torch.tensor([[True] * 5 + [False] * pad]),
    }
    out8 = model({"profiles": obs8}, q)
    assert torch.allclose(out5, out8, atol=1e-6)


def test_permutation_invariance():
    model = make_stub()
    q = queries()
    obs = profile_obs(23, seed=11)
    perm = torch.randperm(23)
    shuffled = dict(prof=obs["prof"][:, perm], lat=obs["lat"][:, perm],
                    lon=obs["lon"][:, perm], month=obs["month"])
    assert torch.allclose(model({"profiles": obs}, q),
                          model({"profiles": shuffled}, q), atol=1e-6)


def test_nan_segment_masked():
    obs = profile_obs(4)
    obs["prof"][0, 2, :, 5:10] = float("nan")     # kill segment 1 of profile 2
    enc = ProfileEncoder(DEPTHS, d_model=D_MODEL, n_segments=4)
    tb = enc(**obs)
    mask = tb.mask.reshape(4, 4)                   # (P, segments)
    assert not mask[2, 1]
    assert mask.sum() == 15
    # masked token embeddings are zeroed
    assert torch.equal(tb.emb.reshape(4, 4, -1)[2, 1],
                       torch.zeros(D_MODEL))


def test_all_nan_grid_patch_masked():
    enc = GridPatchEncoder(2, d_model=D_MODEL, patch=(4, 6))
    o = surf_obs(H=8, W=12)
    o["field"][:, :, :4, :6] = float("nan")        # entire first patch
    tb = enc(**o)
    assert tb.emb.shape[1] == (8 // 4) * (12 // 6)
    assert not tb.mask[0, 0]
    assert tb.mask[0, 1:].all()


def test_grid_patch_non_divisible_padding():
    enc = GridPatchEncoder(2, d_model=D_MODEL, patch=(5, 7))
    o = surf_obs(H=12, W=24)                       # 12/5 -> 3, 24/7 -> 4
    tb = enc(**o)
    assert tb.emb.shape[1] == 3 * 4
    assert torch.isfinite(tb.emb).all()


def test_volume_grid_tokens():
    enc = GridPatchEncoder(2, d_model=D_MODEL, patch=(4, 6),
                           modality="woa_grid")
    B, H, W = 2, 8, 12
    field = torch.randn(B, 2, D, H, W)
    tb = enc(field=field, lat=torch.linspace(-80, 80, H),
             lon=torch.linspace(0, 345, W), month=torch.tensor([1, 6]),
             depth=torch.tensor(DEPTHS))
    n_patch = (H // 4) * (W // 6)
    assert tb.emb.shape == (B, D * n_patch, D_MODEL)
    # depth coordinate cycles through the level grid
    got = tb.coord[0, :, 2].reshape(D, n_patch)[:, 0].numpy()
    np.testing.assert_allclose(got, DEPTHS, rtol=1e-6)
    # month is per batch item
    assert (tb.coord[0, :, 3] == 1).all() and (tb.coord[1, :, 3] == 6).all()


# ------------------------------------------------------------------- gradients
def test_gradient_flow():
    model = make_stub()
    q = queries()
    out = model({"profiles": profile_obs(10), "surf": surf_obs()}, q)
    out.pow(2).mean().backward()
    for name in ("profiles", "surf"):
        g = model.encoders[name].val_proj.weight.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    assert model.head[0].weight.grad is not None


def test_empty_observation_set():
    model = make_stub()
    out = model({}, queries())
    assert out.shape == (1, 17, 2) and torch.isfinite(out).all()


# ------------------------------------------------- pipeline bridge / normalise
class FakeGrid:
    nlat, nlon = 12, 24
    lat = np.linspace(-80.0, 80.0, nlat)
    lon = np.linspace(0.0, 345.0, nlon)
    depth = DEPTHS.astype("float64")
    ndepth = D
    ocean = np.ones((nlat, nlon), bool)


class IdentityNorm:
    """Stands in for anomaly.AnomNorm; NaN passthrough checks masking."""
    def z3d(self, v, arr, month):
        return arr
    def zsurf(self, v, arr, month):
        return arr


def fake_sample(P=6, seed=3):
    rng = np.random.default_rng(seed)
    g = FakeGrid()
    ii = rng.integers(0, g.nlat, P)
    jj = rng.integers(0, g.nlon, P)
    obs = {v: np.full((D, g.nlat, g.nlon), np.nan, "float32")
           for v in ("TEMP", "SALT")}
    for v in obs:
        obs[v][:, ii, jj] = rng.normal(size=(D, P)).astype("float32")
    return {
        "month": 4,
        "prof": {"ij": np.stack([ii, jj], 1), "lat": g.lat[ii], "lon": g.lon[jj]},
        "obs": obs,
        "woa": {v: rng.normal(size=(D, g.nlat, g.nlon)).astype("float32")
                for v in ("TEMP", "SALT")},
        "surf": {v: rng.normal(size=(g.nlat, g.nlon)).astype("float32")
                 for v in ("SST", "SSS")},
    }


def test_bridge_end_to_end():
    g = FakeGrid()
    obs = sample_to_obs(fake_sample(), g, IdentityNorm())
    assert set(obs) == {"profiles", "surf", "woa"}
    model = build_stub(g, d_model=D_MODEL, patch=(4, 6))
    q = make_query_coords(g.lat[:5], g.lon[:5], g.depth[:5], 4.0)
    out = model(obs, q)
    assert out.shape == (1, 5, 2) and torch.isfinite(out).all()


def test_bridge_zero_profiles():
    g = FakeGrid()
    s = fake_sample(P=0)
    s["prof"] = {"ij": np.zeros((0, 2), int), "lat": np.zeros(0),
                 "lon": np.zeros(0)}
    s["obs"] = {v: np.full((D, g.nlat, g.nlon), np.nan, "float32")
                for v in ("TEMP", "SALT")}
    obs = sample_to_obs(s, g, IdentityNorm())
    model = build_stub(g, d_model=D_MODEL, patch=(4, 6))
    out = model(obs, make_query_coords([0.0], [180.0], [100.0], 4.0))
    assert torch.isfinite(out).all()
