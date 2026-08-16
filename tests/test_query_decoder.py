"""D4RT query-decoder gates — docs/superpowers/specs/2026-08-16-d4rt-query-decoder-design.md §5.

Every test here corresponds to a numbered gate in the spec.  The gates that
matter most are the two invariance ones (1, 2): the whole point of the
independent-query decoder is that a query's prediction cannot depend on which
other queries happened to be in the batch.
"""
import itertools

import numpy as np
import pytest
import torch

from ocean_tokenizer.query_decoder import (LeadEmbedding,
                                           IndependentQueryDecoder,
                                           QueryLocalRefiner,
                                           ChannelExpertHead,
                                           check_causal,
                                           D4RTQueryDecoder,
                                           ReferenceSlots)

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


# --------------------------------------------------------------------------
# Query-local refiner (spec §3.6) — helpers
# --------------------------------------------------------------------------
def _refiner(seed=0):
    torch.manual_seed(seed)
    m = QueryLocalRefiner(d_model=D_MODEL, n_heads=N_HEADS)
    m.eval()
    return m


def _tokens(n_token, batch=1, seed=2, all_masked=False):
    """Encoded observation tokens with coords, evidence mass and mask."""
    g = torch.Generator().manual_seed(seed)
    emb = torch.randn(batch, n_token, D_MODEL, generator=g)
    coord = torch.stack([
        torch.rand(batch, n_token, generator=g) * 120 - 60,      # lat
        torch.rand(batch, n_token, generator=g) * 360,           # lon
        torch.rand(batch, n_token, generator=g) * 985,           # depth
        torch.full((batch, n_token), 3.0),                       # month
    ], dim=-1)
    tau = torch.rand(batch, n_token, generator=g) + 0.1
    t_off = torch.zeros(batch, n_token)                          # days from t_src
    mask = torch.zeros(batch, n_token, dtype=torch.bool) if all_masked \
        else torch.ones(batch, n_token, dtype=torch.bool)
    return dict(emb=emb, coord=coord, tau=tau, time_offset=t_off, mask=mask)


def _queries(n_query, batch=1, seed=3, lead=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, n_query, D_MODEL, generator=g)
    coord = torch.stack([
        torch.rand(batch, n_query, generator=g) * 120 - 60,
        torch.rand(batch, n_query, generator=g) * 360,
        torch.rand(batch, n_query, generator=g) * 985,
        torch.full((batch, n_query), 3.0),
    ], dim=-1)
    return q, coord, torch.full((batch, n_query), lead, dtype=torch.long)


# --------------------------------------------------------------------------
# Gate 8 — masked tokens contribute nothing, and never leak NaN
# --------------------------------------------------------------------------
def test_masked_token_content_cannot_change_the_output():
    """Stronger than "output is finite": a masked token must be *inert*.

    Poison every masked slot - embedding, evidence mass and coordinates - and
    the answer must be bit-identical to the clean run.
    """
    ref = _refiner()
    tok = _tokens(n_token=32)
    tok["mask"][:, 16:] = False
    q, qcoord, lead = _queries(n_query=8)

    with torch.no_grad():
        clean = ref(q, qcoord, lead, **tok)

    poisoned = {k: v.clone() for k, v in tok.items()}
    poisoned["emb"][:, 16:] = float("nan")
    poisoned["tau"][:, 16:] = 0.0
    poisoned["coord"][:, 16:] = float("nan")
    poisoned["time_offset"][:, 16:] = float("nan")

    with torch.no_grad():
        out = ref(q, qcoord, lead, **poisoned)

    assert torch.isfinite(out).all()
    assert torch.equal(out, clean)


# --------------------------------------------------------------------------
# Gate 9 — empty evidence must not produce NaN
# --------------------------------------------------------------------------
def test_zero_active_tokens_gives_finite_output():
    ref = _refiner()
    tok = _tokens(n_token=16, all_masked=True)
    q, qcoord, lead = _queries(n_query=8)

    with torch.no_grad():
        out = ref(q, qcoord, lead, **tok)

    assert torch.isfinite(out).all()


def test_output_with_no_active_tokens_ignores_token_content_entirely():
    ref = _refiner()
    q, qcoord, lead = _queries(n_query=8)
    a = _tokens(n_token=16, seed=11, all_masked=True)
    b = _tokens(n_token=16, seed=99, all_masked=True)

    with torch.no_grad():
        assert torch.equal(ref(q, qcoord, lead, **a),
                           ref(q, qcoord, lead, **b))


# --------------------------------------------------------------------------
# Channel experts (spec §3.7)
# --------------------------------------------------------------------------
def _experts(seed=0):
    torch.manual_seed(seed)
    m = ChannelExpertHead(d_model=D_MODEL, n_heads=N_HEADS)
    m.eval()
    return m


def test_channel_expert_head_returns_temp_and_salt():
    exp = _experts()
    tok = _tokens(n_token=32)
    q, qcoord, lead = _queries(n_query=8)
    with torch.no_grad():
        out = exp(q, qcoord, lead, **tok)
    assert out.shape == (1, 8, 2)
    assert torch.isfinite(out).all()


def test_salinity_expert_is_separate_from_temperature():
    """Perturbing the salinity refiner must move SALT and leave TEMP alone."""
    exp = _experts()
    tok = _tokens(n_token=32)
    q, qcoord, lead = _queries(n_query=8)

    with torch.no_grad():
        before = exp(q, qcoord, lead, **tok)
        for p in exp.salt_refiner.parameters():
            p.add_(0.5)
        after = exp(q, qcoord, lead, **tok)

    assert torch.equal(after[..., 0], before[..., 0]), "TEMP must not move"
    assert not torch.equal(after[..., 1], before[..., 1]), "SALT must move"


# --------------------------------------------------------------------------
# Gates 3 & 4 — causality, checked on centres and on support independently
# --------------------------------------------------------------------------
def _causal_batch(n=8, centre_days=0.0, support_days=30.0):
    """Monthly tokens: centre at the month centre, ~30-day support.

    The causal cutoff is the END of the source month (+15.2 d), not the month
    centre, so an ordinary monthly token sits exactly at the limit.
    """
    return dict(
        time_offset=torch.full((1, n), centre_days),
        support_t=torch.full((1, n), support_days),
        mask=torch.ones(1, n, dtype=torch.bool),
    )


def test_causal_batch_passes():
    check_causal(**_causal_batch())        # must not raise


def test_token_centre_after_t_src_raises():
    """Gate 3: a token centred in the NEXT month is a leak."""
    bad = _causal_batch(centre_days=+30.44, support_days=0.0)
    with pytest.raises(ValueError, match="centre"):
        check_causal(**bad)


def test_support_reaching_past_t_src_raises_even_when_centre_is_legal():
    """Gate 4: the leak can hide inside the support geometry.

    Centre sits inside the source month, which passes gate 3, but a 91-day
    (seasonal) support reaches 45 days out — into t_src+1.
    """
    sneaky = _causal_batch(centre_days=0.0, support_days=91.0)
    check_causal(**{**sneaky, "support_t": torch.zeros(1, 8)})   # centre ok
    with pytest.raises(ValueError, match="support"):
        check_causal(**sneaky)


def test_masked_tokens_are_exempt_from_causality():
    """Padding slots carry junk times and must not trip the check."""
    b = _causal_batch(centre_days=+999.0, support_days=999.0)
    b["mask"][:] = False
    check_causal(**b)


# --------------------------------------------------------------------------
# Composed decoder — gate 5 and chunk independence
# --------------------------------------------------------------------------
def _full(seed=0, max_lead=3):
    torch.manual_seed(seed)
    m = D4RTQueryDecoder(d_model=D_MODEL, n_blocks=2, n_heads=N_HEADS,
                         max_lead=max_lead)
    m.eval()
    return m


def _full_inputs(n_query=64, n_token=32, lead=0, seed=5):
    g = torch.Generator().manual_seed(seed)
    latent = torch.randn(1, N_LATENT, D_MODEL, generator=g)
    tok = _tokens(n_token=n_token, seed=seed)
    qcoord = torch.stack([
        torch.rand(1, n_query, generator=g) * 120 - 60,
        torch.rand(1, n_query, generator=g) * 360,
        torch.rand(1, n_query, generator=g) * 985,
        torch.full((1, n_query), 3.0),
    ], dim=-1)
    lead_t = torch.full((1, n_query), lead, dtype=torch.long)
    return latent, qcoord, lead_t, tok


def test_lead_zero_output_is_untouched_by_the_forecasting_weights():
    """Gate 5: reconstruction must be isolated from the lead machinery."""
    m = _full()
    latent, qcoord, lead, tok = _full_inputs(lead=0)

    with torch.no_grad():
        before = m(latent, qcoord, lead, **tok)
        # move every *forecast* row of the lead table
        m.lead_embed.emb.weight[1:] += 3.0
        after = m(latent, qcoord, lead, **tok)

    assert torch.equal(after, before)


def test_nonzero_lead_does_respond_to_those_weights():
    """The companion to gate 5 - otherwise the isolation is vacuous."""
    m = _full()
    latent, qcoord, lead, tok = _full_inputs(lead=2)

    with torch.no_grad():
        before = m(latent, qcoord, lead, **tok)
        m.lead_embed.emb.weight[1:] += 3.0
        after = m(latent, qcoord, lead, **tok)

    assert not torch.equal(after, before)


def test_chunked_decode_matches_unchunked():
    """Chunking is a memory device only; it must not couple queries."""
    m = _full()
    latent, qcoord, lead, tok = _full_inputs(n_query=300)

    with torch.no_grad():
        whole = m(latent, qcoord, lead, **tok)
        chunked = m(latent, qcoord, lead, chunk=128, **tok)

    assert torch.equal(whole, chunked)


# --------------------------------------------------------------------------
# Reference slots (spec §3.4) — always-present key mass, availability-aware
# --------------------------------------------------------------------------
def test_reference_slots_have_the_specified_shape():
    slots = ReferenceSlots(n_slots=8, d_model=D_MODEL, n_modalities=3)
    out = slots(torch.ones(2, 3))
    assert out.shape == (2, 8, D_MODEL)


def test_reference_slots_respond_to_which_modalities_are_available():
    """A dropped modality must change the reference the latent falls back on."""
    torch.manual_seed(0)
    slots = ReferenceSlots(n_slots=8, d_model=D_MODEL, n_modalities=3)
    full = slots(torch.tensor([[1.0, 1.0, 1.0]]))
    no_profiles = slots(torch.tensor([[0.0, 1.0, 1.0]]))
    assert not torch.equal(full, no_profiles)


def test_reference_slots_are_present_even_with_nothing_available():
    slots = ReferenceSlots(n_slots=8, d_model=D_MODEL, n_modalities=3)
    out = slots(torch.zeros(1, 3))
    assert out.shape == (1, 8, D_MODEL)
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------
# End-to-end: the `d4rt` fusion variant (spec §4 wiring)
# --------------------------------------------------------------------------
DEPTHS = np.array([5, 15, 25, 35, 45, 55, 65, 85, 105, 125, 145, 165,
                   186, 222, 267, 327, 408, 527, 707, 985], dtype="float32")


class _Grid:
    depth = DEPTHS.astype("float64")


def _obs(P=7, B=1, H=8, W=12, month=3, seed=0):
    from ocean_tokenizer.fusion import build_fusion_model  # noqa: F401
    rng = np.random.default_rng(seed)
    prof = dict(prof=torch.tensor(rng.normal(size=(B, P, 2, len(DEPTHS))
                                             ).astype("float32")),
                lat=torch.tensor(rng.uniform(-80, 80, (B, P)).astype("float32")),
                lon=torch.tensor(rng.uniform(0, 360, (B, P)).astype("float32")),
                month=torch.full((B,), month))
    f = rng.normal(size=(B, 2, H, W)).astype("float32")
    surf = dict(field=torch.tensor(f), lat=torch.linspace(-80, 80, H),
                lon=torch.linspace(0, 345, W), month=torch.full((B,), month))
    return {"profiles": prof, "surf": surf}


def _d4rt_model(seed=0):
    from ocean_tokenizer.fusion import build_fusion_model
    m = build_fusion_model("d4rt", _Grid(), d_model=D_MODEL, n_latent=N_LATENT,
                           n_heads=N_HEADS, n_self_blocks=2, patch=(4, 6),
                           seed=seed)
    m.eval()
    return m


def _qcoord(Q=9, B=1, seed=2):
    rng = np.random.default_rng(seed)
    q = np.stack([rng.uniform(-80, 80, Q), rng.uniform(0, 360, Q),
                  rng.uniform(0, 985, Q), np.full(Q, 3.0)], -1).astype("float32")
    return torch.tensor(np.repeat(q[None], B, axis=0))


def test_d4rt_variant_forward_shape_and_finiteness():
    m = _d4rt_model()
    q = _qcoord()
    lead = torch.zeros(1, 9, dtype=torch.long)
    with torch.no_grad():
        out = m(_obs(), q, lead=lead)
    assert out.shape == (1, 9, 2) and torch.isfinite(out).all()


def test_d4rt_variant_accepts_every_lead_in_range():
    m = _d4rt_model()
    q = _qcoord()
    for l in range(4):
        with torch.no_grad():
            out = m(_obs(), q, lead=torch.full((1, 9), l, dtype=torch.long))
        assert torch.isfinite(out).all(), l


def test_d4rt_variant_survives_every_modality_subset():
    m = _d4rt_model()
    q = _qcoord()
    lead = torch.zeros(1, 9, dtype=torch.long)
    full = _obs()
    for r in range(len(full) + 1):
        for keys in itertools.combinations(full, r):
            with torch.no_grad():
                out = m({k: full[k] for k in keys}, q, lead=lead)
            assert torch.isfinite(out).all(), keys


def test_d4rt_variant_default_lead_is_zero_reconstruction():
    """Omitting `lead` must mean reconstruction, so existing callers work."""
    m = _d4rt_model()
    q = _qcoord()
    with torch.no_grad():
        implicit = m(_obs(), q)
        explicit = m(_obs(), q, lead=torch.zeros(1, 9, dtype=torch.long))
    assert torch.equal(implicit, explicit)
