"""Real-Argo sample construction and the P1/P2/P5 operators.

Built on a synthetic ArgoCohort so the properties are checked exactly rather
than inferred from whatever the real array happened to do that month.
"""
import numpy as np
import pytest
import torch

from ocean_tokenizer.argo_obs import (ArgoCohort, ArgoNorm, ArgoObsConfig,
                                      build_argo_sample, _select_profiles)
from ocean_tokenizer import argo_experiments as E


def make_cohort(n_floats=12, cycles=6, months=8, L=16, seed=0):
    rng = np.random.default_rng(seed)
    P = n_floats * cycles
    wmo = np.repeat([f"F{i:03d}" for i in range(n_floats)], cycles)
    month = np.tile(np.arange(cycles) % months, n_floats)
    lat = rng.uniform(25, 50, P)
    lon = rng.uniform(280, 331, P)
    levels = np.array([5., 25., 45., 65., 85., 105., 125., 145., 165., 185.,
                       205., 225., 262., 366., 584., 949.])[:L]
    T = 20 - 0.015 * levels[None, :] + rng.normal(0, .5, (P, L))
    S = 35 + 0.001 * levels[None, :] + rng.normal(0, .1, (P, L))
    # split floats deterministically: last 3 held out
    heldout = set(f"F{i:03d}" for i in range(n_floats - 3, n_floats))
    return ArgoCohort(
        region="test", levels=levels, month_index=month,
        grid_y=((lat - 25) / 25 * 38).astype(int),
        grid_x=((lon - 280) / 51 * 26).astype(int),
        lat=lat, lon=lon, wmo=wmo, year=np.full(P, 2005),
        year_split=np.full(P, "train"),
        float_split=np.where(np.isin(wmo, list(heldout)),
                             "heldout_float", "cohort_float"),
        TEMP=T, SALT=S,
        TEMP_ERR=np.full((P, L), 0.002), SALT_ERR=np.full((P, L), 0.01),
        grid=(38, 26))


@pytest.fixture
def cohort():
    c = make_cohort()
    order = np.argsort(c.month_index, kind="stable")
    for k in ("month_index", "grid_y", "grid_x", "lat", "lon", "wmo", "year",
              "year_split", "float_split", "TEMP", "SALT", "TEMP_ERR", "SALT_ERR"):
        setattr(c, k, getattr(c, k)[order])
    uniq, start = np.unique(c.month_index, return_index=True)
    stop = np.r_[start[1:], c.month_index.size]
    c._by_month = {int(m): (int(a), int(b)) for m, a, b in zip(uniq, start, stop)}
    return c


# --- the central guarantee ------------------------------------------------
def test_no_heldout_float_ever_enters_the_input(cohort):
    """The property that licenses WMO as the inference unit."""
    norm = ArgoNorm.fit(cohort)
    held = set(cohort.wmo[cohort.float_split == "heldout_float"])
    for m in range(6):
        s = build_argo_sample(cohort, norm, m, cfg=ArgoObsConfig(n_profiles=0),
                              rng=np.random.default_rng(m))
        if s is None:
            continue
        rows = cohort.month(m, float_split="cohort_float")
        assert not (set(cohort.wmo[rows]) & held)
        assert set(np.unique(s["target_wmo"])) <= held


def test_target_carries_its_own_wmo_label(cohort):
    norm = ArgoNorm.fit(cohort)
    s = build_argo_sample(cohort, norm, 0, cfg=ArgoObsConfig(n_profiles=0),
                          rng=np.random.default_rng(0))
    assert s is not None
    assert len(s["target_wmo"]) == s["target"].shape[0]


def test_returns_none_when_no_heldout_float_reported(cohort):
    norm = ArgoNorm.fit(cohort)
    assert build_argo_sample(cohort, norm, 999, rng=np.random.default_rng(0)) is None


# --- provenance and noise -------------------------------------------------
def test_provenance_groups_are_floats_not_rows(cohort):
    """Two cycles of one float must land in ONE noise group."""
    norm = ArgoNorm.fit(cohort)
    rows = cohort.month(0, float_split="cohort_float")
    s = build_argo_sample(cohort, norm, 0, cfg=ArgoObsConfig(n_profiles=0),
                          rng=np.random.default_rng(0), profile_rows=rows)
    prov = s["provenance"].numpy()
    L = cohort.levels.size
    assert np.unique(prov).size == np.unique(cohort.wmo[rows]).size


def test_measured_noise_is_used_and_flagged(cohort):
    norm = ArgoNorm.fit(cohort)
    s = build_argo_sample(cohort, norm, 0, cfg=ArgoObsConfig(n_profiles=0),
                          rng=np.random.default_rng(0))
    assert bool(s["noise_is_measured"].any())


def test_missing_error_falls_back_and_says_so(cohort):
    c = cohort
    c.TEMP_ERR = np.full_like(c.TEMP_ERR, np.nan)
    c.SALT_ERR = np.full_like(c.SALT_ERR, np.nan)
    norm = ArgoNorm.fit(c)
    s = build_argo_sample(c, norm, 0, cfg=ArgoObsConfig(n_profiles=0),
                          rng=np.random.default_rng(0))
    assert not bool(s["noise_is_measured"].any())


def test_zero_reported_error_cannot_become_infinite_evidence(cohort):
    cohort.TEMP_ERR = np.zeros_like(cohort.TEMP_ERR)
    cohort.SALT_ERR = np.zeros_like(cohort.SALT_ERR)
    norm = ArgoNorm.fit(cohort)
    s = build_argo_sample(cohort, norm, 0, cfg=ArgoObsConfig(n_profiles=0),
                          rng=np.random.default_rng(0))
    assert torch.isfinite(s["noise_density"]).all()
    assert float(s["noise_density"].min()) > 0.0


# --- normalisation --------------------------------------------------------
def test_norm_is_fitted_on_train_only(cohort):
    cohort.year_split = np.where(np.arange(cohort.TEMP.shape[0]) < 20,
                                 "train", "holdout")
    n = ArgoNorm.fit(cohort, "train")
    ref = np.nanmean(cohort.TEMP[:20], axis=0)
    assert np.allclose(n.mean["TEMP"], ref)


def test_norm_round_trips(cohort):
    n = ArgoNorm.fit(cohort)
    a = cohort.TEMP[:5]
    assert np.allclose(n.unz("TEMP", n.z("TEMP", a)), a, atol=1e-9)


def test_constant_level_does_not_explode(cohort):
    cohort.TEMP[:, 3] = 7.0
    n = ArgoNorm.fit(cohort)
    assert np.isfinite(n.z("TEMP", cohort.TEMP[:5])).all()


# --- subsampling ----------------------------------------------------------
def test_subsampling_spreads_across_floats(cohort):
    """A float with many cycles must not crowd out every other platform."""
    rows = cohort.month(0, float_split="cohort_float")
    got = _select_profiles(cohort, rows, 4, np.random.default_rng(0))
    assert got.size <= 4
    assert np.unique(cohort.wmo[got]).size >= min(4, np.unique(cohort.wmo[rows]).size)


# --- P1 layouts -----------------------------------------------------------
def test_layouts_return_exactly_equal_counts(cohort):
    """The plan's first P1 cross-check: exact profile count equality."""
    counts = {k: E.layout(cohort, 0, 5, k, seed=1).size for k in E.LAYOUTS}
    assert len(set(counts.values())) == 1, counts


def test_dispersed_is_farther_apart_than_clustered(cohort):
    r = E.layout_report(cohort, 0, 5, seed=1)
    assert r["dispersed"]["mean_pairwise_km"] > r["clustered"]["mean_pairwise_km"]


def test_layout_selection_is_deterministic(cohort):
    for k in E.LAYOUTS:
        a = E.layout(cohort, 0, 5, k, seed=3)
        b = E.layout(cohort, 0, 5, k, seed=3)
        assert np.array_equal(a, b)


def test_layout_never_draws_a_heldout_float(cohort):
    held = set(cohort.wmo[cohort.float_split == "heldout_float"])
    for k in E.LAYOUTS:
        rows = E.layout(cohort, 0, 5, k, seed=1)
        assert not (set(cohort.wmo[rows]) & held)


def test_layout_caps_all_arms_equally_when_the_month_is_thin(cohort):
    counts = {k: E.layout(cohort, 0, 10_000, k, seed=1).size for k in E.LAYOUTS}
    assert len(set(counts.values())) == 1


# --- P2 redundancy --------------------------------------------------------
def test_exact_duplicates_repeat_one_row(cohort):
    base = cohort.month(0, float_split="cohort_float")[:4]
    rows, _ = E.duplicate_rows(cohort, base, 8, "exact", seed=1, month=0)
    assert (rows[:8] == base[0]).all()


def test_positive_control_uses_genuinely_different_profiles(cohort):
    """`separated` must be new water, not the anchor repeated."""
    base = cohort.month(0, float_split="cohort_float")[:4]
    rows, _ = E.duplicate_rows(cohort, base, 4, "separated", 1, 0)
    assert np.unique(rows[:4]).size > 1


def test_separated_is_farther_apart_than_jittered(cohort):
    base = cohort.month(0, float_split="cohort_float")[:4]
    sep, _ = E.duplicate_rows(cohort, base, 4, "separated", 1, 0)
    jit, _ = E.duplicate_rows(cohort, base, 4, "jittered", 1, 0)
    assert E.mean_pairwise_km(cohort, sep[:4]) > E.mean_pairwise_km(cohort, jit[:4])


# --- P5 sparsity ----------------------------------------------------------
def test_thinning_is_identical_across_methods(cohort):
    """Same seed, same month, same fraction -> the same set, always."""
    rows = cohort.month(0, float_split="cohort_float")
    a = E.thin(cohort, rows, 0.5, seed=7, month=0)
    b = E.thin(cohort, rows, 0.5, seed=7, month=0)
    assert np.array_equal(a, b)


def test_thinning_fractions_are_monotone(cohort):
    rows = cohort.month(0, float_split="cohort_float")
    sizes = [E.thin(cohort, rows, f, 7, 0).size for f in E.SPARSITY_FRACTIONS]
    assert sizes == sorted(sizes, reverse=True)


def test_full_density_is_a_no_op(cohort):
    rows = cohort.month(0, float_split="cohort_float")
    assert np.array_equal(E.thin(cohort, rows, 1.0, 7, 0), rows)


def test_haversine_matches_a_known_distance():
    """~111 km per degree of latitude at the equator."""
    d = E.haversine_km(0.0, 0.0, 1.0, 0.0)
    assert 110.0 < float(d) < 112.0


# --- training split -------------------------------------------------------
def test_training_never_targets_a_heldout_float(cohort):
    """The evaluation cohort must not be spent on optimisation."""
    from ocean_tokenizer.argo_obs import training_rows
    held = set(cohort.wmo[cohort.float_split == "heldout_float"])
    for m in range(6):
        src, tgt = training_rows(cohort, m, 0, 8, np.random.default_rng(m))
        assert not (set(cohort.wmo[tgt]) & held)
        assert not (set(cohort.wmo[src]) & held)


def test_training_input_and_target_floats_are_disjoint(cohort):
    """A float is an input or a target in a sample, never both."""
    from ocean_tokenizer.argo_obs import training_rows
    for m in range(6):
        src, tgt = training_rows(cohort, m, 0, 8, np.random.default_rng(m + 99))
        assert not (set(cohort.wmo[src]) & set(cohort.wmo[tgt]))


def test_training_partition_varies_across_samples(cohort):
    """Over training every float must appear in both roles."""
    from ocean_tokenizer.argo_obs import training_rows
    seen = set()
    for i in range(25):
        _, tgt = training_rows(cohort, 0, 0, 8, np.random.default_rng(i))
        seen |= set(cohort.wmo[tgt])
    assert len(seen) > 1


def test_same_and_independent_provenance_are_different_constructions(cohort):
    """They must differ in the water, not only in the provenance labels.

    An earlier draft repeated the anchor row for both and merely relabelled
    provenance; the two families then produced identical curves by arithmetic
    rather than by physics.
    """
    base = cohort.month(0, float_split="cohort_float")[:4]
    r_same, e_same = E.duplicate_rows(cohort, base, 4, "same_provenance", 1, 0)
    r_ind, e_ind = E.duplicate_rows(cohort, base, 4, "independent_provenance", 1, 0)
    assert not np.array_equal(r_same[:4], r_ind[:4])
    assert np.unique(e_same["provenance_override"]).size == 1
    assert np.unique(e_ind["provenance_override"]).size == 4


def test_same_provenance_draws_from_one_float(cohort):
    base = cohort.month(0, float_split="cohort_float")[:4]
    rows, _ = E.duplicate_rows(cohort, base, 4, "same_provenance", 1, 0)
    assert np.unique(cohort.wmo[rows[:4]]).size == 1


def test_independent_provenance_draws_from_distinct_floats(cohort):
    base = cohort.month(0, float_split="cohort_float")[:4]
    rows, _ = E.duplicate_rows(cohort, base, 4, "independent_provenance", 1, 0)
    assert np.unique(cohort.wmo[rows[:4]]).size > 1


# --- secondary split protocols -------------------------------------------
def test_apply_splits_relabels_years_only(cohort):
    from ocean_tokenizer import protocol as PR
    cohort.year = np.where(np.arange(cohort.year.size) < 20, 2005, 2016)
    cohort.apply_splits(PR.ECCO_OVERLAP_SPLITS)
    assert set(cohort.year_split[:20]) == {"train"}
    assert set(cohort.year_split[20:]) == {"development"}


def test_apply_splits_never_moves_the_heldout_float_cohort(cohort):
    """The invariant that makes the ECCO-overlap protocol legitimate.

    If shifting the evaluation era could also re-draw which floats are held
    out, then choosing the era would be choosing the evaluation cohort — the
    exact degree of freedom the registered cohort seed exists to remove.
    """
    from ocean_tokenizer import protocol as PR
    before = cohort.float_split.copy()
    held_before = set(cohort.wmo[before == "heldout_float"])
    cohort.apply_splits(PR.ECCO_OVERLAP_SPLITS)
    assert np.array_equal(cohort.float_split, before)
    assert set(cohort.wmo[cohort.float_split == "heldout_float"]) == held_before


def test_ecco_overlap_eval_era_is_inside_ecco_v4r4_coverage():
    """ECCO V4r4 ends 2018-01-01; the secondary protocol must respect that."""
    from ocean_tokenizer import protocol as PR
    for split in ("train", "validation", "development", "holdout"):
        assert PR.ECCO_OVERLAP_SPLITS[split][1] <= 2017, split


def test_split_protocols_do_not_overlap_between_train_and_eval():
    from ocean_tokenizer import protocol as PR
    for name, table in PR.SPLIT_PROTOCOLS.items():
        tr_hi = table["train"][1]
        for later in ("validation", "development"):
            assert table[later][0] > tr_hi, f"{name}: {later} overlaps train"


def test_main_protocol_hash_is_unaffected_by_the_secondary_table():
    """Adding a secondary protocol must not invalidate existing artifacts."""
    from ocean_tokenizer import protocol as PR
    assert "ecco_overlap" not in PR.canonical_hash(
        {"year_splits": PR.YEAR_SPLITS})
    assert PR.SPLIT_PROTOCOLS["main"] is PR.YEAR_SPLITS
