"""Task 6, Tests A + B — exact token partition & duplicate observations.

Test A: replace one token (k, v, w) with n identical children (k, v, w/n).
        MBCA output must be unchanged to float tolerance; standard attention
        must change (the token gains n-fold multiplicity).
Test B: duplicate selected observations without adding information, dividing
        the original mass across the copies (controlled condition).  MBCA
        exact; standard attention drifts monotonically with the factor.

All probes manipulate the post-encoder TokenBatch via
ocean_tokenizer.invariance, so metadata rides along and "same evidence" is
constructed exactly.  Models are randomly initialised with a fixed seed —
the invariance under test is architectural, not learned.
"""
import numpy as np
import pytest
import torch

from ocean_tokenizer.token_api import MODALITIES, TokenBatch
from ocean_tokenizer.fusion import build_fusion_model
from ocean_tokenizer.invariance import (split_token, duplicate_tokens,
                                        output_change)

D_MODEL, N_LATENT = 32, 16
DEPTHS = np.array([5, 15, 25, 35, 45, 55, 65, 85, 105, 125, 145, 165,
                   186, 222, 267, 327, 408, 527, 707, 985], dtype="float32")
D = len(DEPTHS)
TOL_EXACT = 1e-5          # float32 relative tolerance (brief)
SENSITIVE = 1e-4          # below this, "changed" is not credible


class FakeGrid:
    depth = DEPTHS.astype("float64")


def build(variant, seed=3):
    m = build_fusion_model(variant, FakeGrid(), d_model=D_MODEL,
                           n_latent=N_LATENT, n_heads=4, n_self_blocks=2,
                           patch=(4, 6), seed=seed)
    m.eval()
    return m


def obs(P=7, seed=0, month=3):
    rng = np.random.default_rng(seed)
    surf = rng.normal(size=(1, 2, 8, 12)).astype("float32")
    surf[:, :, :2, :3] = np.nan
    return {
        "profiles": dict(
            prof=torch.tensor(rng.normal(size=(1, P, 2, D)).astype("float32")),
            lat=torch.tensor(rng.uniform(-80, 80, (1, P)).astype("float32")),
            lon=torch.tensor(rng.uniform(0, 360, (1, P)).astype("float32")),
            month=torch.tensor([month])),
        "surf": dict(field=torch.tensor(surf), lat=torch.linspace(-80, 80, 8),
                     lon=torch.linspace(0, 345, 12), month=torch.tensor([month])),
    }


def queries(Q=9, seed=2):
    rng = np.random.default_rng(seed)
    q = np.stack([rng.uniform(-80, 80, Q), rng.uniform(0, 360, Q),
                  rng.uniform(0, 985, Q), np.full(Q, 3.0)], -1).astype("float32")
    return torch.tensor(q)[None]


def encoded(model):
    tb = model.encode(obs(), batch=1)
    assert tb.support_mass is not None
    return tb


def first_token_of(tb, modality_name):
    m_id = MODALITIES[modality_name]
    idx = ((tb.modality[0] == m_id) & tb.mask[0]).nonzero()
    assert len(idx) > 0
    return int(idx[0])


# ============================ Test A: exact partition ============================
@pytest.mark.parametrize("n", [2, 4, 8])
@pytest.mark.parametrize("modality", ["profile", "surf_grid"])
def test_A_mbca_exact_partition(n, modality):
    model = build("mbca")
    tb = encoded(model)
    idx = first_token_of(tb, modality)
    rel = output_change(model, tb, split_token(tb, idx, n), queries())
    assert rel < TOL_EXACT, f"MBCA partition n={n} ({modality}): rel {rel:.2e}"


@pytest.mark.parametrize("n", [2, 4, 8])
def test_A_standard_attention_changes(n):
    model = build("perceiver")
    tb = encoded(model)
    idx = first_token_of(tb, "profile")
    rel = output_change(model, tb, split_token(tb, idx, n), queries())
    assert rel > SENSITIVE, f"expected multiplicity shift, got {rel:.2e}"


def test_A_standard_shift_grows_with_n():
    """More children -> more multiplicity -> larger standard-attention shift."""
    model = build("perceiver")
    tb = encoded(model)
    idx = first_token_of(tb, "profile")
    q = queries()
    rels = [output_change(model, tb, split_token(tb, idx, n), q)
            for n in (2, 4, 8)]
    assert rels[0] < rels[1] < rels[2], rels


# ========================= Test B: duplicate observations =========================
def profile_token_indices(tb, n_profiles=3):
    """All token indices of the first ``n_profiles`` profiles (via parent_id)."""
    m_id = MODALITIES["profile"]
    sel = ((tb.modality[0] == m_id) & tb.mask[0]
           & (tb.parent_id[0] < n_profiles))
    return sel.nonzero().flatten()


@pytest.mark.parametrize("factor", [2, 4, 8])
def test_B_mbca_duplication_invariant(factor):
    """Copies with the original mass divided across them add no evidence."""
    model = build("mbca")
    tb = encoded(model)
    dup = duplicate_tokens(tb, profile_token_indices(tb), factor,
                           divide_mass=True)
    rel = output_change(model, tb, dup, queries())
    assert rel < TOL_EXACT, f"MBCA dup x{factor}: rel {rel:.2e}"


@pytest.mark.parametrize("factor", [2, 4, 8])
def test_B_standard_attention_drifts(factor):
    model = build("perceiver")
    tb = encoded(model)
    dup = duplicate_tokens(tb, profile_token_indices(tb), factor,
                           divide_mass=True)   # no mass in standard attn anyway
    rel = output_change(model, tb, dup, queries())
    assert rel > SENSITIVE, f"expected duplication drift, got {rel:.2e}"


def test_B_factor_1_is_identity():
    model = build("mbca")
    tb = encoded(model)
    dup = duplicate_tokens(tb, profile_token_indices(tb), 1, divide_mass=True)
    assert torch.equal(tb.emb, dup.emb)
    rel = output_change(model, tb, dup, queries())
    assert rel == 0.0


def test_B_naive_reingestion_detectable_by_provenance():
    """Without mass division (naive re-ingestion) copies look like new
    evidence to every fusion rule — but provenance metadata identifies them:
    the duplicated (modality, parent_id, family_id) keys collide."""
    model = build("mbca")
    tb = encoded(model)
    idxs = profile_token_indices(tb)
    dup = duplicate_tokens(tb, idxs, 2, divide_mass=False)
    keys = torch.stack([dup.modality[0], dup.parent_id[0],
                        dup.family_id[0]], -1)[dup.mask[0]]
    uniq = torch.unique(keys, dim=0)
    assert uniq.shape[0] < keys.shape[0]          # collisions exist
