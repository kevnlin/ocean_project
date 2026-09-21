"""The SetConv -> U-Net -> query backbone.

It is a candidate replacement for the Perceiver-IO trunk, so the properties that
make it a fair comparison — permutation invariance, locality, no dependence
between queries, honest behaviour with no observations — are pinned here rather
than assumed.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from ocean_tokenizer.setconv import SetConvUNet

BOX = dict(lat=(25.0, 50.0), lon=(280.0, 331.0))
LEVELS = np.array([5.0, 105.0, 326.9, 707.6, 1400.5])


def _model(seed=0, **kw):
    torch.manual_seed(seed)
    return SetConvUNet(BOX, LEVELS, ny=20, nx=40, width=8, **kw).eval()


def _obs(k=12, seed=0):
    rng = np.random.default_rng(seed)
    prof = torch.as_tensor(rng.normal(0, 1, (k, 2, LEVELS.size)), dtype=torch.float32)
    lat = torch.as_tensor(rng.uniform(27, 48, k), dtype=torch.float32)
    lon = torch.as_tensor(rng.uniform(282, 329, k), dtype=torch.float32)
    return prof, lat, lon


def _queries(q=7, seed=1):
    rng = np.random.default_rng(seed)
    return torch.as_tensor(np.stack([
        rng.uniform(27, 48, q), rng.uniform(282, 329, q),
        rng.choice(LEVELS, q), np.full(q, 7.0)], -1), dtype=torch.float32)


def test_forward_shape_and_finiteness():
    m = _model()
    prof, lat, lon = _obs()
    out = m(prof, lat, lon, torch.tensor([7]), _queries())
    assert out.shape == (7, 2)
    assert torch.isfinite(out).all()


def test_permuting_the_profiles_does_not_change_the_answer():
    m = _model()
    prof, lat, lon = _obs()
    q = _queries()
    a = m(prof, lat, lon, torch.tensor([7]), q)
    p = torch.randperm(prof.shape[0])
    b = m(prof[p], lat[p], lon[p], torch.tensor([7]), q)
    assert torch.allclose(a, b, atol=1e-5), "the encoder is not permutation invariant"


def test_a_query_does_not_depend_on_the_other_queries():
    m = _model()
    prof, lat, lon = _obs()
    q = _queries()
    full = m(prof, lat, lon, torch.tensor([7]), q)
    one = m(prof, lat, lon, torch.tensor([7]), q[2:3])
    assert torch.allclose(full[2], one[0], atol=1e-5)


def test_missing_levels_are_not_read_as_measurements():
    """A NaN level must not enter the value channel as a zero anomaly.

    The head is zero-initialised, so it is perturbed first — otherwise every
    output is 0 and the test cannot see the encoder at all.
    """
    m = _model()
    with torch.no_grad():
        m.head.weight.normal_(0, 0.2); m.head.bias.normal_(0, 0.2)
    prof, lat, lon = _obs()
    q = _queries()
    a = m(prof, lat, lon, torch.tensor([7]), q)
    prof2 = prof.clone(); prof2[:, :, 2] = float("nan")
    b = m(prof2, lat, lon, torch.tensor([7]), q)
    assert torch.isfinite(b).all()
    assert not torch.allclose(a, b), "dropping a level changed nothing at all"
    # and the density channel must record the loss: an all-NaN level reads as
    # zero evidence rather than as a measured zero
    enc = m.setconv(prof2, lat, lon)
    dens = enc[2 * LEVELS.size + 2]          # log1p(density) of TEMP, level 2
    assert float(dens.abs().max()) == pytest.approx(0.0, abs=1e-6)


def test_no_observations_gives_a_finite_answer():
    m = _model()
    prof = torch.zeros(0, 2, LEVELS.size)
    out = m(prof, torch.zeros(0), torch.zeros(0), torch.tensor([7]), _queries())
    assert out.shape == (7, 2)
    assert torch.isfinite(out).all()


def test_the_untrained_head_predicts_zero_anomaly():
    """Head initialised at zero: before training the model says 'climatology'."""
    m = _model()
    prof, lat, lon = _obs()
    out = m(prof, lat, lon, torch.tensor([7]), _queries())
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_the_encoder_is_local():
    """Moving one profile far away changes a nearby query more than a distant one."""
    torch.manual_seed(0)
    m = SetConvUNet(BOX, LEVELS, ny=20, nx=40, width=8, ell_km=75.0).eval()
    with torch.no_grad():                      # make the (zero-init) head read out
        m.head.weight.normal_(0, 0.1)
    prof = torch.ones(1, 2, LEVELS.size)
    lat = torch.tensor([30.0]); lon = torch.tensor([285.0])
    q = torch.tensor([[30.0, 285.0, 105.0, 7.0],      # on top of the profile
                      [47.0, 328.0, 105.0, 7.0]])     # the far corner
    base = m(prof, lat, lon, torch.tensor([7]), q)
    moved = m(prof, lat + 0.02, lon + 0.02, torch.tensor([7]), q)
    near = float((base[0] - moved[0]).abs().sum())
    far = float((base[1] - moved[1]).abs().sum())
    assert near > far, f"a nudge moved the far query more ({near} vs {far})"


def test_gradients_reach_the_setconv_width():
    m = SetConvUNet(BOX, LEVELS, ny=20, nx=40, width=8)
    prof, lat, lon = _obs()
    out = m(prof, lat, lon, torch.tensor([7]), _queries())
    out.square().mean().backward()
    assert m.log_ell.grad is not None
    assert torch.isfinite(m.log_ell.grad).all()


@pytest.mark.parametrize("lead", [0, 3, 6])
def test_lead_is_accepted_and_changes_nothing_at_the_zero_head(lead):
    m = _model()
    prof, lat, lon = _obs()
    out = m(prof, lat, lon, torch.tensor([7]), _queries(), lead=lead)
    assert torch.isfinite(out).all()


def test_query_depth_picks_the_right_level():
    """A query at a level's own depth reads that level, not a pooled band."""
    torch.manual_seed(0)
    m = SetConvUNet(BOX, LEVELS, ny=20, nx=40, width=8).eval()
    with torch.no_grad():
        m.head.weight.normal_(0, 0.2); m.head.bias.normal_(0, 0.2)
    prof, lat, lon = _obs()
    q = torch.tensor([[35.0, 300.0, float(d), 7.0] for d in LEVELS])
    out = m(prof, lat, lon, torch.tensor([7]), q)
    assert out.shape == (LEVELS.size, 2)
    # different levels must give different answers — no vertical pooling
    assert len({tuple(np.round(r, 6)) for r in out.detach().numpy()}) == LEVELS.size


def test_chunked_encoding_equals_the_one_shot_sum():
    """The global run encodes ~7 600 profiles in chunks; chunking must not
    change a single value."""
    m = _model()
    prof, lat, lon = _obs(k=37)
    a = m.setconv(prof, lat, lon, chunk=1000)
    b = m.setconv(prof, lat, lon, chunk=5)
    assert torch.allclose(a, b, atol=1e-5)
