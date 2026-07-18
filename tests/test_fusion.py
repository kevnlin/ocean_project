"""Task-3 tests: fusion variants (StandardPerceiver / FixedBudgetResampler / MBCA).

Interface contracts + an MBCA exact-partition sanity check that certifies the
implementation before hand-off.  The full decisive invariance suite
(duplication factors, grid retokenization, profile resampling — Task 6) lives
in tests/test_mbca_invariance.py and friends [coworker].
"""
import itertools

import numpy as np
import pytest
import torch

from ocean_tokenizer.token_api import TokenBatch, MODALITIES
from ocean_tokenizer.fusion import (
    StandardPerceiver, FixedBudgetResampler, MBCA, VARIANTS,
    build_fusion_model, mbca_weights,
)

D_MODEL, N_LATENT = 32, 16
DEPTHS = np.array([5, 15, 25, 35, 45, 55, 65, 85, 105, 125, 145, 165,
                   186, 222, 267, 327, 408, 527, 707, 985], dtype="float32")
D = len(DEPTHS)


class FakeGrid:
    depth = DEPTHS.astype("float64")


def build(variant, seed=0):
    m = build_fusion_model(variant, FakeGrid(), d_model=D_MODEL,
                           n_latent=N_LATENT, n_heads=4, n_self_blocks=2,
                           patch=(4, 6), seed=seed)
    m.eval()
    return m


def profile_obs(P, B=1, seed=0, month=3):
    rng = np.random.default_rng(seed)
    return dict(
        prof=torch.tensor(rng.normal(size=(B, P, 2, D)).astype("float32")),
        lat=torch.tensor(rng.uniform(-80, 80, (B, P)).astype("float32")),
        lon=torch.tensor(rng.uniform(0, 360, (B, P)).astype("float32")),
        month=torch.full((B,), month))


def surf_obs(B=1, H=8, W=12, seed=1, month=3):
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(B, 2, H, W)).astype("float32")
    f[:, :, :2, :3] = np.nan
    return dict(field=torch.tensor(f), lat=torch.linspace(-80, 80, H),
                lon=torch.linspace(0, 345, W), month=torch.full((B,), month))


def queries(Q=9, B=1, seed=2):
    rng = np.random.default_rng(seed)
    q = np.stack([rng.uniform(-80, 80, Q), rng.uniform(0, 360, Q),
                  rng.uniform(0, 985, Q), np.full(Q, 3.0)], -1).astype("float32")
    return torch.tensor(np.repeat(q[None], B, axis=0))


FULL = lambda: {"profiles": profile_obs(7), "surf": surf_obs()}


# ------------------------------------------------------------------ interface
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_forward_shapes(variant):
    model = build(variant)
    out = model(FULL(), queries())
    assert out.shape == (1, 9, 2) and torch.isfinite(out).all()


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_missing_modalities_any_subset(variant):
    model = build(variant)
    q = queries()
    full = FULL()
    for r in range(len(full) + 1):
        for keys in itertools.combinations(full, r):
            out = model({k: full[k] for k in keys}, q)
            assert out.shape == (1, 9, 2) and torch.isfinite(out).all(), keys


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_zero_profiles(variant):
    model = build(variant)
    out = model({"profiles": profile_obs(0), "surf": surf_obs()}, queries())
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_gradient_flow(variant):
    model = build(variant)
    model.train()
    out = model(FULL(), queries())
    out.pow(2).mean().backward()
    g = model.encoders["profiles"].level_mlp[0].weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    g = model.latent0.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_variants_share_trunk_weights_at_same_seed():
    a, c = build("perceiver", seed=7), build("mbca", seed=7)
    assert torch.equal(a.latent0, c.latent0)
    assert torch.equal(a.encoders["profiles"].level_mlp[0].weight,
                       c.encoders["profiles"].level_mlp[0].weight)
    assert torch.equal(a.head[0].weight, c.head[0].weight)


# ------------------------------------------------------------- mbca weights
def test_mbca_weights_normalise_and_modality_prior():
    B, N = 1, 6
    modality = torch.tensor([[0, 0, 0, 0, 2, 2]])
    mask = torch.ones(B, N, dtype=torch.bool)
    mass = torch.tensor([[4.0, 2.0, 1.0, 1.0, 3.0, 1.0]])
    w = mbca_weights(mass, modality, mask)
    assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-6)
    # each present modality gets pi = 1/2 in total
    assert torch.allclose(w[0, :4].sum(), torch.tensor(0.5), atol=1e-6)
    assert torch.allclose(w[0, 4:].sum(), torch.tensor(0.5), atol=1e-6)
    # within modality, proportional to mass
    assert torch.allclose(w[0, 0] / w[0, 2], torch.tensor(4.0), atol=1e-5)


def test_mbca_weights_missing_modality_renormalises():
    modality = torch.tensor([[0, 0, 0]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    w = mbca_weights(None, modality, mask)          # None -> uniform mass
    assert torch.allclose(w, torch.full((1, 3), 1 / 3), atol=1e-6)


# ------------------------------------------- exact partition sanity (pre-Task 6)
def _split_tokens(tb: TokenBatch, idx: int, n: int) -> TokenBatch:
    """Replace token ``idx`` with n identical children of mass w/n."""
    reps = [1] * tb.emb.shape[1]
    reps[idx] = n
    r = torch.repeat_interleave(torch.arange(tb.emb.shape[1]),
                                torch.tensor(reps))
    mass = tb.support_mass.clone()
    mass[:, idx] /= n
    return TokenBatch(tb.emb[:, r], tb.coord[:, r], tb.modality[:, r],
                      tb.mask[:, r], mass[:, r])


@pytest.mark.parametrize("n", [2, 4, 8])
def test_mbca_exact_partition_invariance(n):
    model = build("mbca", seed=3)
    q = queries()
    tokens = model.encode(FULL(), batch=1)
    if tokens.support_mass is None:
        tokens = TokenBatch(tokens.emb, tokens.coord, tokens.modality,
                            tokens.mask, tokens.mask.float())
    idx = int(tokens.mask[0].nonzero()[0])
    with torch.no_grad():
        y0 = model.decode(model.fuse(tokens), q)
        y1 = model.decode(model.fuse(_split_tokens(tokens, idx, n)), q)
    rel = (y0 - y1).norm() / (y0.norm() + 1e-12)
    assert rel < 1e-5, f"partition n={n}: rel err {rel:.2e}"


def test_standard_attention_not_partition_invariant():
    """The failure mode MBCA fixes: splitting a token shifts standard attention."""
    model = build("perceiver", seed=3)
    q = queries()
    tokens = model.encode(FULL(), batch=1)
    if tokens.support_mass is None:
        tokens = TokenBatch(tokens.emb, tokens.coord, tokens.modality,
                            tokens.mask, tokens.mask.float())
    idx = int(tokens.mask[0].nonzero()[0])
    with torch.no_grad():
        y0 = model.decode(model.fuse(tokens), q)
        y1 = model.decode(model.fuse(_split_tokens(tokens, idx, 8)), q)
    rel = (y0 - y1).norm() / (y0.norm() + 1e-12)
    assert rel > 1e-4, f"expected sensitivity, got rel err {rel:.2e}"
