"""The plan's S6 acceptance rules, as tests rather than as prose."""
import json
import pytest

from ocean_tokenizer import protocol as P
from ocean_tokenizer.crosscheck import (compare_artifacts, compare_exact,
                                        compare_numeric, check_conclusions,
                                        render_report)


def _art(track="A", **kw):
    a = P.ResultArtifact(package="P0", track=track, region="gulfstream",
                         results=kw.pop("results", {}), **kw)
    a.protocol_hash_ = P.protocol_hash()
    a.protocol_version = P.PROTOCOL_VERSION
    a.data_manifests = {"argo": "aaa", "godas_gulfstream": "bbb"}
    a.counts = {"targets": 100, "wmos": 7, "months": 12}
    return a


# --- S2 / protocol -------------------------------------------------------
def test_protocol_hash_is_stable_and_order_independent():
    assert P.protocol_hash() == P.protocol_hash()
    assert len(P.protocol_hash()) == 64


def test_year_splits_agree_with_the_godas_driver_and_the_argo_cohort():
    """Three copies of a split definition is three chances to drift."""
    import re, pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    drv = (root / "experiments" / "14_godas_dfs_d4rt.py").read_text()
    coh = (root / "experiments" / "44_build_argo_cohort.py").read_text()
    for name, (lo, hi) in P.YEAR_SPLITS.items():
        assert re.search(rf'"{name}":\s*\({lo},\s*{hi}\)', drv), f"{name} in driver"
        assert re.search(rf'"{name}":\s*\({lo},\s*{hi}\)', coh), f"{name} in cohort"


def test_regions_share_a_grid_shape():
    """A region effect must not be a grid-shape effect in disguise."""
    spans = {k: (v["lat"][1] - v["lat"][0], v["lon"][1] - v["lon"][0])
             for k, v in P.REGIONS.items()}
    assert len(set(spans.values())) == 1, spans


def test_absent_dataset_is_recorded_not_omitted(tmp_path):
    """Collection-level view still enumerates every dataset, present or not."""
    h = P.collection_manifest_hashes(str(tmp_path))
    assert set(h) >= {"argo", "argo_cohort", "reference_products"}
    assert all(v == "absent" for v in h.values())


def test_artifact_round_trips_and_flags_a_dirty_tree(tmp_path):
    a = _art().finalize(str(tmp_path))
    p = tmp_path / "art.json"
    sha = a.write(str(p))
    assert len(sha) == 64
    b = P.ResultArtifact.read(str(p))
    assert b.protocol_hash_ == a.protocol_hash_
    if a.git_dirty_:
        assert any("dirty" in w for w in a.warnings)


# --- S6.1 exact-match ----------------------------------------------------
def test_manifest_difference_is_a_protocol_problem():
    a, b = _art("A"), _art("B")
    b.data_manifests["argo"] = "different"
    cc = compare_artifacts(a, b)
    assert not cc.protocol_ok
    assert cc.must_reconcile
    assert cc.summary() == "PROTOCOL PROBLEM"
    assert any("argo" in m.field for m in cc.exact_mismatches)


def test_count_difference_is_a_protocol_problem():
    a, b = _art("A"), _art("B")
    b.counts["wmos"] = 8
    cc = compare_artifacts(a, b)
    assert not cc.protocol_ok


def test_absent_key_on_one_side_is_caught():
    a, b = _art("A"), _art("B")
    del b.counts["months"]
    assert any(m.b == "<absent>" for m in compare_exact(a, b))


def test_identical_artifacts_agree():
    cc = compare_artifacts(_art("A"), _art("B"))
    assert cc.protocol_ok and cc.same_conclusion
    assert cc.summary() == "agree"


def test_refuses_to_crosscheck_a_track_against_itself():
    """S8's rationale collapses if both columns come from the same track."""
    with pytest.raises(ValueError, match="same.*track|track"):
        compare_artifacts(_art("A"), _art("A"))


def test_refuses_mismatched_package_or_region():
    a, b = _art("A"), _art("B")
    b.package = "P1"
    with pytest.raises(ValueError, match="package"):
        compare_artifacts(a, b)


# --- S6.2 numerical ------------------------------------------------------
def test_float_noise_is_tolerated():
    d = compare_numeric({"rmse": 1.0}, {"rmse": 1.0 + 1e-12})
    assert d == []


def test_real_numeric_difference_is_reported_but_not_fatal():
    a, b = _art("A"), _art("B")
    a.results = {"rmse": 1.0}; b.results = {"rmse": 1.05}
    cc = compare_artifacts(a, b)
    assert cc.numeric_diffs
    assert cc.protocol_ok and cc.same_conclusion   # reported, not fatal
    assert not cc.must_reconcile


def test_nan_equals_nan():
    assert compare_numeric({"x": float("nan")}, {"x": float("nan")}) == []


# --- S6.2 the five stopping conditions -----------------------------------
def test_sign_flip_of_dfs_minus_uniform_forces_reconciliation():
    m = check_conclusions({"dfs_minus_uniform": {"point": -0.02, "excludes_zero": True}},
                          {"dfs_minus_uniform": {"point": +0.02, "excludes_zero": True}})
    assert any(x.field.endswith(".sign") for x in m)


def test_ci_zero_inclusion_change_forces_reconciliation():
    m = check_conclusions({"dfs_minus_uniform": {"point": -0.02, "excludes_zero": True}},
                          {"dfs_minus_uniform": {"point": -0.02, "excludes_zero": False}})
    assert any("excludes_zero" in x.field for x in m)


def test_ranking_change_forces_reconciliation():
    m = check_conclusions({"ranking": ["dfs", "uniform", "count"]},
                          {"ranking": ["uniform", "dfs", "count"]})
    assert any(x.field == "ranking" for x in m)


def test_layout_did_sign_change_forces_reconciliation():
    m = check_conclusions({"layout_did": {"point": 0.1}},
                          {"layout_did": {"point": -0.1}})
    assert any(x.field == "layout_did.sign" for x in m)


def test_redundancy_positive_control_disagreement_forces_reconciliation():
    m = check_conclusions({"redundancy_control": {"passes": True}},
                          {"redundancy_control": {"passes": False}})
    assert m and "positive control" in m[0].detail


def test_a_quantity_only_one_track_computed_is_surfaced():
    m = check_conclusions({"layout_did": {"point": 0.1}}, {})
    assert m and "only one track" in m[0].detail


def test_absent_from_both_is_not_a_disagreement():
    assert check_conclusions({}, {}) == []


def test_matching_conclusions_pass():
    r = {"dfs_minus_uniform": {"point": -0.02, "excludes_zero": True},
         "ranking": ["dfs", "uniform"],
         "redundancy_control": {"passes": True}}
    assert check_conclusions(r, dict(r)) == []


# --- S5 rendering --------------------------------------------------------
def test_report_marks_track_b_pending_and_claims_no_agreement(tmp_path):
    md = render_report("P0", "gulfstream", _art("A").finalize(str(tmp_path)), None)
    assert "Track B is pending" in md
    assert "pending" in md
    assert "no agreement should be inferred" in md


# --- seed variance, tolerances and coverage (found by the real cross-check) --
def test_seed_aware_interval_is_wider_than_a_single_seed_interval():
    """The bug the Gulf Stream P0 run exposed.

    A cluster bootstrap resamples floats but conditions on one trained model.
    When seed-to-seed variance dominates, a single seed can report a difference
    that excludes zero while the recipe's true interval does not.
    """
    from ocean_tokenizer.clustered_ci import (ClusterStats, ci_rmse,
                                              ci_rmse_across_seeds)
    import numpy as np
    per_seed = []
    for i, scale in enumerate((1.0, 1.6, 0.7)):     # big between-seed spread
        n = np.full(40, 200.0)
        per_seed.append(ClusterStats(np.arange(40), n * scale, n,
                                     cluster_unit="wmo", method="m"))
    one = ci_rmse(per_seed[0], n_boot=2000, seed=1)
    many = ci_rmse_across_seeds(per_seed, n_boot=2000, seed=1)
    assert many.n_seeds == 3 and one.n_seeds == 1
    assert (many.hi - many.lo) > (one.hi - one.lo)


def test_seed_aware_difference_can_include_zero_when_a_single_seed_does_not():
    from ocean_tokenizer.clustered_ci import (ClusterStats, ci_difference,
                                              ci_difference_across_seeds)
    import numpy as np
    n = np.full(60, 200.0)
    cs = lambda s: ClusterStats(np.arange(60), n * s, n, cluster_unit="wmo")
    # seed 0 favours a strongly; seeds 1 and 2 reverse it
    a = [cs(1.00), cs(1.30), cs(0.90)]
    b = [cs(1.15), cs(1.10), cs(0.85)]
    one = ci_difference(a[0], b[0], n_boot=3000, seed=2)
    many = ci_difference_across_seeds(a, b, n_boot=3000, seed=2)
    assert one.excludes_zero
    assert not many.excludes_zero


def test_seed_t_interval_matches_a_hand_computation():
    from ocean_tokenizer.clustered_ci import seed_t_interval
    import numpy as np
    vals = [-0.0081, 0.0071, -0.0007]
    iv = seed_t_interval(vals)
    m = float(np.mean(vals)); se = float(np.std(vals, ddof=1) / np.sqrt(3))
    assert iv.point == pytest.approx(m)
    assert iv.lo == pytest.approx(m - 4.302652729911275 * se, rel=1e-6)
    assert not iv.excludes_zero


def test_bootstrap_ci_endpoints_get_a_monte_carlo_budget():
    """Two tracks resampling differently cannot match an endpoint to 1e-6."""
    from ocean_tokenizer.crosscheck import compare_numeric
    a = {"d": {"point": 1.0, "lo": 0.90, "hi": 1.10}}
    b = {"d": {"point": 1.0, "lo": 0.93, "hi": 1.07}}
    assert compare_numeric(a, b) == []


def test_a_point_estimate_still_held_to_the_tight_tolerance():
    from ocean_tokenizer.crosscheck import compare_numeric
    d = compare_numeric({"d": {"point": 1.0}}, {"d": {"point": 1.05}})
    assert d and d[0].field.endswith("point")


def test_near_zero_values_are_not_judged_by_relative_error_alone():
    """DFS - Uniform is ~6e-4; a 1e-9 absolute difference is not a finding."""
    from ocean_tokenizer.crosscheck import compare_numeric
    assert compare_numeric({"x": 6.0e-4}, {"x": 6.0e-4 + 1e-9}) == []


def test_coverage_difference_is_reported_but_is_not_a_conclusion_change():
    """Scoring different row sets is a coverage fact, not a disagreement."""
    from ocean_tokenizer.crosscheck import coverage_note, check_conclusions
    a = {"ranking": ["dfs", "uniform", "en4"]}
    b = {"ranking": ["dfs", "uniform"]}
    assert check_conclusions(a, b) == []          # order agrees on the common rows
    note = coverage_note(a, b)
    assert note and note[0].kind == "numeric" and note[0].a == ["en4"]


def test_ranking_order_change_on_common_rows_is_still_a_conclusion_change():
    from ocean_tokenizer.crosscheck import check_conclusions
    m = check_conclusions({"ranking": ["dfs", "uniform", "en4"]},
                          {"ranking": ["uniform", "dfs"]})
    assert m and "rows both tracks scored" in m[0].detail


def test_disjoint_row_sets_are_flagged():
    from ocean_tokenizer.crosscheck import check_conclusions
    m = check_conclusions({"ranking": ["a"]}, {"ranking": ["b"]})
    assert m and "no row in common" in m[0].detail


def test_comparable_block_is_preferred_over_raw_results():
    """Two independent tracks will not share an internal results layout."""
    from ocean_tokenizer.crosscheck import build_comparable
    a, b = _art("A"), _art("B")
    a.results = {"track_a_shape": {"deep": {"nesting": 1.0}},
                 "comparable": build_comparable(rmse={"dfs": {"TEMP": 0.5}})}
    b.results = {"totally_different": [1, 2, 3],
                 "comparable": build_comparable(rmse={"dfs": {"TEMP": 0.5}})}
    cc = compare_artifacts(a, b)
    assert cc.numeric_diffs == []
    assert cc.summary() == "agree"


# --- manifest scoping (a design error the real runs exposed) ---------------
def test_data_manifests_record_only_what_the_run_consumed(tmp_path):
    """Hashing a whole-collection manifest made artifacts sensitive to
    unrelated dataset growth: downloading equatorial floats rewrote the Argo
    manifest and changed the hash recorded by Gulf Stream runs that never
    touched them. `data_manifests` is exact-matched under S6, so that turned an
    unrelated download into a spurious protocol violation."""
    import os
    root = tmp_path
    (root / "data" / "argo_cohort").mkdir(parents=True)
    (root / "data" / "argo_cohort" / "gulfstream.nc").write_bytes(b"cohort-A")
    (root / "data" / "argo").mkdir(parents=True)
    (root / "data" / "argo" / "manifest.json").write_text('{"files": []}')

    before = P.data_manifest_hashes(str(root), "gulfstream")
    # a later, unrelated download rewrites the collection manifest and adds a
    # different region's cohort
    (root / "data" / "argo" / "manifest.json").write_text('{"files": [1, 2, 3]}')
    (root / "data" / "argo_cohort" / "eq_pacific.nc").write_bytes(b"cohort-B")
    after = P.data_manifest_hashes(str(root), "gulfstream")

    assert before == after, "an unrelated download changed this region's hashes"
    assert "argo_cohort:gulfstream" in before
    # the collection view DOES move — that is its job, and it is provenance only
    assert (P.collection_manifest_hashes(str(root))["argo"]
            != "absent")


def test_changing_the_region_cohort_does_change_the_hash(tmp_path):
    """The flip side: the file a run actually read must still be pinned."""
    root = tmp_path
    (root / "data" / "argo_cohort").mkdir(parents=True)
    f = root / "data" / "argo_cohort" / "gulfstream.nc"
    f.write_bytes(b"cohort-A")
    a = P.data_manifest_hashes(str(root), "gulfstream")
    f.write_bytes(b"cohort-A-EDITED")
    b = P.data_manifest_hashes(str(root), "gulfstream")
    assert a != b


def test_absent_cohort_is_recorded_not_omitted(tmp_path):
    h = P.data_manifest_hashes(str(tmp_path), "npac_gyre")
    assert h["argo_cohort:npac_gyre"] == "absent"


def test_collection_manifests_are_not_exact_match_fields():
    assert "collection_manifests" not in P.EXACT_MATCH_FIELDS
    assert "data_manifests" in P.EXACT_MATCH_FIELDS
