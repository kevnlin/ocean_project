"""GODAS observation contract (mentor doc §2.1).

One sample has 664 tokens at the default settings:

  * 24 synthetic profile columns x 16 depths = 384 point T/S tokens at t_src
  * 70 surface-patch tokens x two context months = 140 surface T/S tokens
  * 70 patch tokens x two context months = 140 SSH tokens

The five masks must stay distinct; conflating any two is how a padding slot
starts contributing evidence or a withheld target starts being scored.
"""
import numpy as np
import pytest
import torch

from ocean_tokenizer.godas_obs import (build_sample, ObsConfig, N_PROFILE_COLS,
                                       N_PATCHES, PATCH, CONTEXT_MONTHS)


def _fields(nt=6, nz=16, ny=38, nx=26, seed=0):
    rng = np.random.default_rng(seed)
    f = dict(
        months=np.array([np.datetime64("2000-01") + i for i in range(nt)]),
        TEMP=rng.normal(0, 1, (nt, nz, ny, nx)),
        SALT=rng.normal(0, 1, (nt, nz, ny, nx)),
        SSH=rng.normal(0, 1, (nt, ny, nx)),
    )
    # a full PATCH-sized land block, not a single cell: one land cell leaves a
    # 4x4 patch's nanmean finite, so nothing would ever be a dead slot and the
    # masking tests would be vacuous
    for k in ("TEMP", "SALT"):
        f[k][:, :, :PATCH, :PATCH] = np.nan
    f["SSH"][:, :PATCH, :PATCH] = np.nan
    return f


def _sample(seed=0, **kw):
    return build_sample(_fields(), t_src=3, cfg=ObsConfig(**kw),
                        rng=np.random.default_rng(seed))


# --------------------------------------------------------------------------
# Token budget — the doc's 664
# --------------------------------------------------------------------------
def test_patch_size_tiles_the_grid_into_the_documented_70():
    ny, nx = 38, 26
    assert -(-ny // PATCH) * -(-nx // PATCH) == N_PATCHES == 70


def test_default_sample_allocates_the_documented_664_token_slots():
    """664 is the padded BUDGET, not the live count.

    On a real ocean grid some slots are all-land and are masked off — 523 live
    of 664 on a real GODAS month. Asserting the live count would make the test
    pass only on synthetic data with negligible land.
    """
    s = _sample()
    assert s["mask"].numel() == 664, "doc §2.1 token budget"
    assert int(s["mask"].sum()) <= 664


def test_token_budget_breaks_down_as_documented():
    s = _sample()
    mod = s["modality"]
    counts = {int(k): int((mod == k).sum()) for k in torch.unique(mod)}
    assert counts[0] == N_PROFILE_COLS * 16 == 384, "profile point tokens"
    assert counts[1] == N_PATCHES * CONTEXT_MONTHS == 140, "surface tokens"
    assert counts[2] == N_PATCHES * CONTEXT_MONTHS == 140, "SSH tokens"


# --------------------------------------------------------------------------
# The five masks are distinct things
# --------------------------------------------------------------------------
def test_the_five_masks_are_all_present_and_distinctly_shaped():
    s = _sample()
    assert s["mask"].dtype == torch.bool                  # real vs padding
    assert s["value_mask"].shape == s["value"].shape      # finite per value
    assert s["support_mask"].shape == s["mask"].shape     # evidence-eligible
    assert s["modality_available"].shape == (3,)          # whole stream kept
    assert s["target_mask"].shape == s["target"].shape    # loss supervision


def test_all_land_slots_are_masked_off_rather_than_dropped():
    """A land slot keeps its place in the budget but must not be a token."""
    s = _sample()
    dead = ~s["mask"]
    assert int(dead.sum()) > 0, "the synthetic land column should kill slots"
    assert not bool(s["value_mask"][dead].any())


def test_missing_values_are_zero_filled_but_still_flagged():
    """Doc §2.1: zero-filled only at the encoder boundary, masks preserved."""
    s = _sample()
    assert torch.isfinite(s["value"]).all(), "no NaN may reach the encoder"
    assert not bool(s["value_mask"].all()), "the land column must be flagged"
    assert float(s["value"][~s["value_mask"]].abs().max()) == 0.0


# --------------------------------------------------------------------------
# Causality — profiles at t_src, gridded over [t_src-1, t_src]
# --------------------------------------------------------------------------
def test_no_token_carries_a_time_later_than_t_src():
    s = _sample()
    assert float(s["coord"][s["mask"], 3].max()) <= 0.0 + 1e-9


def test_profiles_are_only_at_t_src():
    s = _sample()
    prof = s["mask"] & (s["modality"] == 0)
    assert torch.allclose(s["coord"][prof, 3], torch.zeros(int(prof.sum()),
                                                           dtype=s["coord"].dtype))


def test_gridded_streams_span_two_context_months():
    s = _sample()
    surf = s["mask"] & (s["modality"] == 1)
    months = torch.unique(s["coord"][surf, 3])
    assert months.numel() == CONTEXT_MONTHS
    assert float(months.min()) == pytest.approx(-1.0)


# --------------------------------------------------------------------------
# Training-time dropout (doc §2.1)
# --------------------------------------------------------------------------
def test_modality_dropout_always_retains_at_least_one_stream():
    for seed in range(40):
        s = _sample(seed=seed, modality_dropout=0.9, train=True)
        assert int(s["modality_available"].sum()) >= 1


def test_modality_dropout_is_off_outside_training():
    for seed in range(10):
        s = _sample(seed=seed, modality_dropout=0.9, train=False)
        assert int(s["modality_available"].sum()) == 3


def test_a_dropped_modality_contributes_no_live_tokens():
    for seed in range(30):
        s = _sample(seed=seed, modality_dropout=0.9, train=True)
        for m, keep in enumerate(s["modality_available"]):
            if not bool(keep):
                assert int((s["mask"] & (s["modality"] == m)).sum()) == 0


def test_target_channel_withholding_leaves_at_least_one_channel():
    for seed in range(40):
        s = _sample(seed=seed, target_dropout=0.9, train=True)
        per_channel = s["target_mask"].any(dim=0)
        assert int(per_channel.sum()) >= 1


# --------------------------------------------------------------------------
# §9 audit helpers: forced modality availability and the duplicate attack
# --------------------------------------------------------------------------
def test_forcing_modalities_keeps_only_those_streams():
    from ocean_tokenizer.godas_obs import MOD_PROFILE, MOD_SURF, MOD_SSH
    s = build_sample(_fields(), t_src=3,
                     cfg=ObsConfig(force_available=(True, False, False)),
                     rng=np.random.default_rng(0))
    assert int((s["mask"] & (s["modality"] == MOD_PROFILE)).sum()) > 0
    assert int((s["mask"] & (s["modality"] == MOD_SURF)).sum()) == 0
    assert int((s["mask"] & (s["modality"] == MOD_SSH)).sum()) == 0
    assert list(s["modality_available"]) == [True, False, False]


def test_forcing_overrides_training_dropout():
    """The audit must control availability exactly, not roll dice."""
    for seed in range(10):
        s = build_sample(_fields(), t_src=3,
                         cfg=ObsConfig(train=True, modality_dropout=0.9,
                                       force_available=(True, True, False)),
                         rng=np.random.default_rng(seed))
        assert list(s["modality_available"]) == [True, True, False]


def test_duplicate_attack_adds_k_minus_one_copies():
    from ocean_tokenizer.godas_obs import duplicate_profile_attack, MOD_PROFILE
    s = build_sample(_fields(), t_src=3, cfg=ObsConfig(),
                     rng=np.random.default_rng(0))
    n0 = int((s["mask"] & (s["modality"] == MOD_PROFILE)).sum())
    a1 = duplicate_profile_attack(s, k=1, temp_bias=2.0)
    a8 = duplicate_profile_attack(s, k=8, temp_bias=2.0)
    live = lambda x: int((x["mask"] & (x["modality"] == MOD_PROFILE)).sum())
    assert live(a1) == n0
    assert live(a8) == n0 + 7 * 16, "7 extra copies of a 16-depth column"


def test_duplicate_attack_biases_the_targeted_profile_only():
    from ocean_tokenizer.godas_obs import duplicate_profile_attack
    s = build_sample(_fields(), t_src=3, cfg=ObsConfig(),
                     rng=np.random.default_rng(0))
    a = duplicate_profile_attack(s, k=1, temp_bias=2.0)
    d = a["value"][:16, 0] - s["value"][:16, 0]
    assert torch.allclose(d[s["value_mask"][:16, 0]],
                          torch.full_like(d[s["value_mask"][:16, 0]], 2.0))
    assert torch.equal(a["value"][16:s["value"].shape[0]], s["value"][16:])


def test_duplicate_copies_are_exact():
    from ocean_tokenizer.godas_obs import duplicate_profile_attack
    s = build_sample(_fields(), t_src=3, cfg=ObsConfig(),
                     rng=np.random.default_rng(0))
    a = duplicate_profile_attack(s, k=4, temp_bias=2.0)
    n = s["coord"].shape[0]
    for c in range(3):
        lo = n + c * 16
        assert torch.equal(a["coord"][lo:lo + 16], a["coord"][:16])
        assert torch.equal(a["value"][lo:lo + 16], a["value"][:16])


# --------------------------------------------------------------------------
# §9 audit: a controlled attack needs a fixed target and a real baseline
# --------------------------------------------------------------------------
def test_pinning_places_the_first_profile_at_a_fixed_cell():
    """Without this the attack hits different water every month, and the
    between-month variance swamps the effect being measured."""
    from ocean_tokenizer.godas_obs import MOD_PROFILE
    seen = set()
    for seed in range(6):
        s = build_sample(_fields(), t_src=3,
                         cfg=ObsConfig(pin_first_profile=(20, 13)),
                         rng=np.random.default_rng(seed))
        prof = s["coord"][s["modality"] == MOD_PROFILE]
        seen.add((round(float(prof[0, 0]), 6), round(float(prof[0, 1]), 6)))
    assert len(seen) == 1, f"attack location drifted across seeds: {seen}"


def test_unpinned_first_profile_still_varies():
    from ocean_tokenizer.godas_obs import MOD_PROFILE
    seen = set()
    for seed in range(6):
        s = build_sample(_fields(), t_src=3, cfg=ObsConfig(),
                         rng=np.random.default_rng(seed))
        prof = s["coord"][s["modality"] == MOD_PROFILE]
        seen.add((round(float(prof[0, 0]), 6), round(float(prof[0, 1]), 6)))
    assert len(seen) > 1


def test_a_zero_bias_attack_leaves_values_untouched():
    """The unbiased baseline must be a true no-op, or the control is not one."""
    from ocean_tokenizer.godas_obs import duplicate_profile_attack
    s = build_sample(_fields(), t_src=3, cfg=ObsConfig(),
                     rng=np.random.default_rng(0))
    a = duplicate_profile_attack(s, k=1, temp_bias=0.0)
    assert torch.equal(a["value"], s["value"])


def test_attacking_non_profile_tokens_raises():
    """Tokens 0..15 are the first profile column only because profiles are
    built first and dropout is off. If that ever stops holding, the attack
    would silently hit the surface stream instead."""
    from ocean_tokenizer.godas_obs import duplicate_profile_attack
    s = build_sample(_fields(), t_src=3,
                     cfg=ObsConfig(force_available=(False, True, True)),
                     rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="profile"):
        duplicate_profile_attack(s, k=8, temp_bias=2.0)
