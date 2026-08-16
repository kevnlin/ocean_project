"""D4RT query-decoder gates — docs/superpowers/specs/2026-08-16-d4rt-query-decoder-design.md §5.

Every test here corresponds to a numbered gate in the spec.  The gates that
matter most are the two invariance ones (1, 2): the whole point of the
independent-query decoder is that a query's prediction cannot depend on which
other queries happened to be in the batch.
"""
import numpy as np
import pytest
import torch

from ocean_tokenizer.query_decoder import (LeadEmbedding,
                                           IndependentQueryDecoder)

D_MODEL, N_HEADS, N_LATENT = 64, 4, 32


def _decoder(n_blocks=2, seed=0):
    torch.manual_seed(seed)
    m = IndependentQueryDecoder(d_model=D_MODEL, n_blocks=n_blocks,
                                n_heads=N_HEADS, mlp_ratio=2.0)
    m.eval()
    return m


def _inputs(n_query, batch=1, seed=1):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, n_query, D_MODEL, generator=g)
    latent = torch.randn(batch, N_LATENT, D_MODEL, generator=g)
    return q, latent


# --------------------------------------------------------------------------
# Gate 6 — lead-0 zero-signal invariant
# --------------------------------------------------------------------------
def test_lead_zero_contributes_exactly_zero_at_init():
    emb = LeadEmbedding(max_lead=3, d_model=8)
    out = emb(torch.zeros(2, 5, dtype=torch.long))
    assert torch.equal(out, torch.zeros(2, 5, 8))


def test_lead_zero_stays_exactly_zero_after_training_steps():
    """The invariant that matters: padding_idx must survive the optimizer.

    A plain zero-init would drift off zero on the first step and lead 0 would
    silently stop being pure reconstruction.
    """
    emb = LeadEmbedding(max_lead=3, d_model=8)
    opt = torch.optim.AdamW(emb.parameters(), lr=0.1)
    lead = torch.tensor([[0, 1, 2, 3]])
    for _ in range(10):
        opt.zero_grad()
        # push every row hard, including row 0
        (emb(lead) - 5.0).pow(2).mean().backward()
        opt.step()

    assert torch.equal(emb(torch.zeros(1, 4, dtype=torch.long)),
                       torch.zeros(1, 4, 8))
    # and the other rows genuinely moved, so the test isn't vacuous
    assert emb(torch.tensor([[1, 2, 3]])).abs().min() > 0


# --------------------------------------------------------------------------
# Gate 7 — integer lead indexing, no silent rounding or wrapping
# --------------------------------------------------------------------------
def test_lead_above_max_raises():
    emb = LeadEmbedding(max_lead=3, d_model=8)
    with pytest.raises(ValueError, match="lead"):
        emb(torch.tensor([[4]]))


def test_negative_lead_raises_rather_than_wrapping():
    """t_tgt < t_src is not a forecast; it must never index from the end."""
    emb = LeadEmbedding(max_lead=3, d_model=8)
    with pytest.raises(ValueError, match="lead"):
        emb(torch.tensor([[-1]]))


def test_float_lead_raises_rather_than_rounding():
    emb = LeadEmbedding(max_lead=3, d_model=8)
    with pytest.raises(TypeError, match="lead"):
        emb(torch.tensor([[1.7]]))


# --------------------------------------------------------------------------
# Gate 1 — query permutation invariance
# --------------------------------------------------------------------------
def test_permuting_queries_permutes_output_identically():
    """No self-attention between queries => order cannot matter."""
    dec = _decoder()
    q, latent = _inputs(n_query=64)
    perm = torch.randperm(64, generator=torch.Generator().manual_seed(7))

    with torch.no_grad():
        out = dec(q, latent)
        out_perm = dec(q[:, perm], latent)

    assert torch.equal(out_perm, out[:, perm])


# --------------------------------------------------------------------------
# Gate 2 — query extension invariance (incl. across a chunk boundary)
# --------------------------------------------------------------------------
def test_adding_queries_does_not_change_existing_predictions():
    dec = _decoder()
    q_big, latent = _inputs(n_query=1000)
    q_small = q_big[:, :100]

    with torch.no_grad():
        out_small = dec(q_small, latent)
        out_big = dec(q_big, latent)

    assert torch.equal(out_small, out_big[:, :100])


def test_deleting_queries_does_not_change_the_survivors():
    dec = _decoder()
    q, latent = _inputs(n_query=64)
    keep = torch.arange(0, 64, 2)

    with torch.no_grad():
        out_full = dec(q, latent)
        out_kept = dec(q[:, keep], latent)

    assert torch.equal(out_kept, out_full[:, keep])
