"""Properties of the cluster bootstrap the cross-check plan's S6 rules rest on."""
import numpy as np
import pytest

from ocean_tokenizer.clustered_ci import (ClusterStats, accumulate, ci_rmse,
                                          ci_difference, ci_did, ranking,
                                          bootstrap_indices)


def _stats(se, n, unit="source_month", method="m", ids=None):
    ids = np.arange(len(se)) if ids is None else np.asarray(ids)
    return ClusterStats(ids, np.asarray(se, float), np.asarray(n, float),
                        cluster_unit=unit, method=method)


def test_point_estimate_is_the_pooled_rmse_not_the_mean_of_rmses():
    """The distinction that matters when clusters differ in size.

    One cluster with 1000 cells at error 1 and one with 10 cells at error 10:
    pooled RMSE is dominated by the big cluster, the mean-of-RMSEs is not.
    Argo floats differ in size by more than this, so picking the wrong estimand
    is not academic.
    """
    s = _stats(se=[1000 * 1.0, 10 * 100.0], n=[1000, 10])
    assert s.rmse() == pytest.approx(np.sqrt(2000.0 / 1010.0))
    mean_of_rmses = np.mean([1.0, 10.0])
    assert abs(s.rmse() - mean_of_rmses) > 4.0


def test_accumulate_matches_direct_pooling():
    rng = np.random.default_rng(0)
    err = rng.normal(size=500)
    months = rng.integers(0, 7, size=500)
    s = accumulate(months, err ** 2, cluster_unit="source_month")
    assert s.rmse() == pytest.approx(np.sqrt(np.mean(err ** 2)))
    assert s.n.sum() == 500


def test_accumulate_drops_nan_without_poisoning_the_cluster():
    err = np.array([1.0, np.nan, 3.0, 4.0])
    lab = np.array(["a", "a", "b", "b"])
    s = accumulate(lab, err, cluster_unit="wmo")
    assert np.isfinite(s.rmse())
    assert s.n.tolist() == [1.0, 2.0]


def test_invalid_mask_is_honoured():
    err = np.ones(6)
    lab = np.array([0, 0, 0, 1, 1, 1])
    valid = np.array([1, 1, 0, 1, 0, 0], bool)
    s = accumulate(lab, err, valid, cluster_unit="wmo")
    assert s.n.tolist() == [2.0, 1.0]


def test_bootstrap_draws_are_reproducible_across_calls():
    """Two tracks must get identical resamples from the same cohort + seed."""
    a = bootstrap_indices(12, 100, seed=7)
    b = bootstrap_indices(12, 100, seed=7)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, bootstrap_indices(12, 100, seed=8))


def test_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    n = rng.integers(50, 500, size=30).astype(float)
    se = n * rng.uniform(0.5, 2.0, size=30)
    s = _stats(se, n)
    iv = ci_rmse(s, n_boot=2000, seed=3)
    assert iv.lo <= iv.point <= iv.hi
    assert iv.n_clusters == 30
    assert iv.cluster_unit == "source_month"


def test_paired_difference_is_tighter_than_naive_independent_intervals():
    """The reason ci_difference exists.

    Two methods whose per-cluster errors move together: an unpaired treatment
    would widen the difference interval enough to swallow zero and trip the
    plan's stopping rule for no reason.
    """
    rng = np.random.default_rng(2)
    n = np.full(40, 200.0)
    shared = rng.uniform(0.5, 3.0, size=40)          # month-to-month difficulty
    a = _stats(n * shared * 1.00, n, method="a")
    b = _stats(n * shared * 1.21, n, method="b")     # b uniformly ~10% worse
    paired = ci_difference(a, b, n_boot=4000, seed=5)
    assert paired.point < 0                          # a is better
    assert paired.excludes_zero                      # and detectably so

    ia, ib = ci_rmse(a, n_boot=4000, seed=5), ci_rmse(b, n_boot=4000, seed=5)
    naive_width = (ia.hi - ia.lo) + (ib.hi - ib.lo)
    assert (paired.hi - paired.lo) < naive_width


def test_difference_sign_convention_negative_means_first_is_better():
    n = np.full(20, 100.0)
    good = _stats(n * 1.0, n, method="good")
    bad = _stats(n * 4.0, n, method="bad")
    d = ci_difference(good, bad, n_boot=1000)
    assert d.point == pytest.approx(1.0 - 2.0)


def test_difference_requires_matching_cluster_unit():
    n = np.full(5, 10.0)
    with pytest.raises(ValueError, match="cluster unit"):
        ci_difference(_stats(n, n, unit="wmo"), _stats(n, n, unit="source_month"))


def test_difference_aligns_on_common_clusters_only():
    """ECCO stops in 2017; pairing must intersect, never shift the axis."""
    n = np.full(4, 100.0)
    a = _stats(n * 1.0, n, ids=[2015, 2016, 2017, 2018], method="a")
    b = _stats(n[:3] * 4.0, n[:3], ids=[2015, 2016, 2017], method="b")
    d = ci_difference(a, b, n_boot=500)
    assert d.n_clusters == 3
    assert d.point == pytest.approx(1.0 - 2.0)


def test_did_sign_is_positive_when_uniform_is_hurt_more_by_clustering():
    """The P1 quantity, in the direction the DFS claim predicts.

    Uniform loses a lot when the layout clusters; DFS loses little.  The plan's
    DiD = gap(Uniform) - gap(DFS) must then come out positive.
    """
    n = np.full(25, 200.0)
    dfs_disp = _stats(n * 1.00, n, method="dfs_dispersed")
    dfs_clus = _stats(n * 1.02, n, method="dfs_clustered")     # barely hurt
    uni_disp = _stats(n * 1.00, n, method="uni_dispersed")
    uni_clus = _stats(n * 1.60, n, method="uni_clustered")     # badly hurt
    did = ci_did(dfs_clus, dfs_disp, uni_clus, uni_disp, n_boot=2000)
    assert did.point > 0
    assert did.excludes_zero


def test_did_is_zero_when_both_methods_react_identically():
    n = np.full(25, 200.0)
    lo = _stats(n * 1.0, n)
    hi = _stats(n * 1.3, n)
    did = ci_did(hi, lo, hi, lo, n_boot=1000)
    assert did.point == pytest.approx(0.0, abs=1e-12)
    assert not did.excludes_zero


def test_excludes_zero_is_false_for_a_straddling_interval():
    rng = np.random.default_rng(4)
    n = np.full(30, 100.0)
    noise = rng.normal(1.0, 0.3, size=30) ** 2
    a = _stats(n * noise, n, method="a")
    b = _stats(n * noise[::-1], n, method="b")   # same values, permuted
    d = ci_difference(a, b, n_boot=3000, seed=11)
    assert not d.excludes_zero


def test_empty_cohort_gives_nan_not_zero():
    s = ClusterStats(np.array([]), np.array([]), np.array([]),
                     cluster_unit="wmo")
    assert np.isnan(s.rmse())


def test_ranking_orders_best_first():
    order = ranking([_stats([400.0], [100.0], method="bad"),
                     _stats([100.0], [100.0], method="good")])
    assert [m for m, _ in order] == ["good", "bad"]


def test_bca_available_and_brackets_point_on_few_clusters():
    """The GODAS holdout has 8 eligible source months."""
    rng = np.random.default_rng(6)
    n = rng.integers(80, 200, size=8).astype(float)
    se = n * rng.uniform(0.5, 2.5, size=8)
    s = _stats(se, n)
    iv = ci_rmse(s, n_boot=4000, seed=9, kind="bca")
    assert iv.kind == "bca"
    assert iv.lo <= iv.point <= iv.hi
