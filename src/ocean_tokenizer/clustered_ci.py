"""Cluster bootstrap confidence intervals — the CI machinery the plan requires.

The cross-check plan asks for month-clustered and WMO-clustered CIs on every
headline number (P0), and its acceptance rules (S6) are stated in terms of CI
behaviour: *"CI inclusion/exclusion of zero changes"* and *"the sign of
DFS - Uniform changes"* are what force two tracks to stop and reconcile.  So the
CI is not decoration here; it is the object the cross-check compares.

Three things this module gets right that a naive implementation gets wrong
------------------------------------------------------------------------

**1. Resample clusters, recompute the ratio — do not average per-cluster RMSE.**
Pooled RMSE is ``sqrt(sum_i SE_i / sum_i N_i)``, a nonlinear function of two
sums.  The mean of per-cluster RMSEs is a different estimand and is biased
whenever clusters differ in size, which Argo floats emphatically do (a float
reporting 200 cycles and one reporting 12 are both one WMO).  We therefore carry
per-cluster *sufficient statistics* ``(SE, N)``, resample whole clusters, and
re-form the ratio.

**2. Pair the resamples across methods.**  DFS and Uniform are scored on the
same months and the same floats; their errors are strongly positively
correlated.  Bootstrapping each method independently and differencing the
intervals produces a CI for the difference that is far too wide, and would
manufacture the plan's "CI includes zero" stopping condition out of nothing.
Every method here is resampled on **the same draw of cluster indices**, so the
difference CI reflects the paired design.  ``ci_difference`` and ``ci_did``
depend on this.

**3. A cluster is the unit of independence, and it is a choice.**  A month's
errors are correlated through the shared ocean state; a float's are correlated
through the instrument and its water mass.  The plan names WMO/platform as the
primary unit and source-month as the secondary, and requires both be reported.
``cluster_unit`` is therefore an explicit, recorded field, never a default.

Reference: Cameron, Gelbach & Miller (2008); Efron & Tibshirani (1993) S8.
The percentile interval is used, with BCa available where the extra accuracy is
wanted at small cluster counts (the GODAS holdout has only 8 eligible source
months, where percentile intervals are known to under-cover).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable, Sequence

import numpy as np

DEFAULT_B = 10000
DEFAULT_ALPHA = 0.05


# --------------------------------------------------------------------------
# sufficient statistics
# --------------------------------------------------------------------------
@dataclass
class ClusterStats:
    """Per-cluster squared-error sums for one method and one channel.

    ``se[c]`` is the sum of squared errors over every scored cell in cluster
    ``c``; ``n[c]`` is how many cells that was.  Pooled RMSE is
    ``sqrt(se.sum() / n.sum())`` — this is exactly the quantity the existing
    GODAS ``evaluate()`` computes, only kept per cluster instead of collapsed,
    so nothing about the point estimate changes.
    """
    cluster_ids: np.ndarray          # (C,) labels, order defines the axis
    se: np.ndarray                   # (C,) sum of squared error
    n: np.ndarray                    # (C,) count of scored cells
    cluster_unit: str                # "source_month" | "wmo" | ...
    channel: str = ""
    method: str = ""

    def __post_init__(self):
        self.cluster_ids = np.asarray(self.cluster_ids)
        self.se = np.asarray(self.se, dtype=np.float64)
        self.n = np.asarray(self.n, dtype=np.float64)
        if not (self.cluster_ids.shape == self.se.shape == self.n.shape):
            raise ValueError("cluster_ids, se and n must have the same shape")
        if self.se.ndim != 1:
            raise ValueError("cluster stats must be 1-D over clusters")
        if np.any(self.n < 0) or np.any(self.se < 0):
            raise ValueError("negative se or n")

    @staticmethod
    def from_dict(d: dict, cluster_unit: str, channel: str = "",
                  method: str = "") -> "ClusterStats":
        """Rebuild from the (ids, se, n) an artifact stored."""
        return ClusterStats(np.asarray(d["ids"]), np.asarray(d["se"], float),
                            np.asarray(d["n"], float), cluster_unit=cluster_unit,
                            channel=channel, method=method)

    @property
    def n_clusters(self) -> int:
        return int(self.cluster_ids.size)

    def rmse(self) -> float:
        """The pooled point estimate. Empty -> NaN, never a silent zero."""
        tot = self.n.sum()
        return float(np.sqrt(self.se.sum() / tot)) if tot > 0 else float("nan")


def accumulate(cluster_labels: Sequence, sq_err: np.ndarray,
               valid: np.ndarray | None = None, *, cluster_unit: str,
               channel: str = "", method: str = "") -> ClusterStats:
    """Fold per-cell squared errors into per-cluster (se, n).

    ``cluster_labels`` is one label per *cell* (repeat the month/WMO across that
    cell's rows).  ``valid`` masks cells that were not scored; NaNs in
    ``sq_err`` are treated as invalid too, so an unmasked NaN cannot silently
    poison a whole cluster's sum.
    """
    sq_err = np.asarray(sq_err, dtype=np.float64).ravel()
    labels = np.asarray(cluster_labels).ravel()
    if labels.size != sq_err.size:
        raise ValueError(f"{labels.size} labels for {sq_err.size} errors")
    ok = np.isfinite(sq_err)
    if valid is not None:
        ok &= np.asarray(valid).ravel().astype(bool)
    labels, sq_err = labels[ok], sq_err[ok]
    uniq, inv = np.unique(labels, return_inverse=True)
    se = np.bincount(inv, weights=sq_err, minlength=uniq.size)
    n = np.bincount(inv, minlength=uniq.size).astype(np.float64)
    return ClusterStats(uniq, se, n, cluster_unit=cluster_unit,
                        channel=channel, method=method)


# --------------------------------------------------------------------------
# the bootstrap
# --------------------------------------------------------------------------
def bootstrap_indices(n_clusters: int, n_boot: int = DEFAULT_B,
                      seed: int = 20260905) -> np.ndarray:
    """(n_boot, n_clusters) cluster draws, shared by every method.

    Generated from the cluster COUNT and a seed alone, so two tracks that agree
    on the cohort get bit-identical resamples and any CI difference between them
    is a real difference in the errors, not in the random draw.  This is what
    makes the S6 acceptance rules decidable.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_clusters, size=(n_boot, n_clusters))


def _pooled(se: np.ndarray, n: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Pooled RMSE for every bootstrap row. idx: (B, C) cluster indices."""
    tot_se = se[idx].sum(axis=1)
    tot_n = n[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(np.where(tot_n > 0, tot_se / np.maximum(tot_n, 1e-300),
                                np.nan))


def _percentile_ci(draws: np.ndarray, alpha: float) -> tuple[float, float]:
    d = draws[np.isfinite(draws)]
    if d.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _bca_ci(draws: np.ndarray, theta_hat: float, jack: np.ndarray,
            alpha: float) -> tuple[float, float]:
    """Bias-corrected and accelerated interval (Efron 1987).

    Worth the extra jackknife pass when the cluster count is small: the GODAS
    holdout has 8 eligible source months, and percentile intervals under-cover
    badly there.  Falls back to percentile if the acceleration is degenerate.
    """
    from scipy.stats import norm
    d = draws[np.isfinite(draws)]
    if d.size == 0 or not np.isfinite(theta_hat):
        return float("nan"), float("nan")
    prop = np.mean(d < theta_hat)
    if prop <= 0 or prop >= 1:
        return _percentile_ci(draws, alpha)
    z0 = norm.ppf(prop)
    jm = jack.mean()
    num = ((jm - jack) ** 3).sum()
    den = 6.0 * (((jm - jack) ** 2).sum() ** 1.5)
    a = num / den if den > 0 else 0.0
    zl, zu = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)
    def adj(z):
        return norm.cdf(z0 + (z0 + z) / max(1 - a * (z0 + z), 1e-12))
    ql, qu = adj(zl), adj(zu)
    if not (0 < ql < qu < 1):
        return _percentile_ci(draws, alpha)
    lo, hi = np.percentile(d, [100 * ql, 100 * qu])
    return float(lo), float(hi)


def _jackknife_pooled(se: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Leave-one-cluster-out pooled RMSE, for the BCa acceleration."""
    tse, tn = se.sum(), n.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(np.where(tn - n > 0, (tse - se) / np.maximum(tn - n, 1e-300),
                                np.nan))


@dataclass
class Interval:
    point: float
    lo: float
    hi: float
    n_clusters: int
    cluster_unit: str
    n_boot: int
    alpha: float
    method_label: str = ""
    kind: str = "percentile"
    #: how many training seeds the interval integrates over. 1 means the
    #: interval CONDITIONS on one trained model and cannot see seed variance.
    n_seeds: int = 1

    @property
    def excludes_zero(self) -> bool:
        """The S6 stopping condition, computed once rather than eyeballed."""
        return bool(np.isfinite(self.lo) and np.isfinite(self.hi)
                    and (self.lo > 0 or self.hi < 0))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["excludes_zero"] = self.excludes_zero
        return d

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.lo:.4f}, {self.hi:.4f}]"


def ci_rmse(stats: ClusterStats, n_boot: int = DEFAULT_B,
            alpha: float = DEFAULT_ALPHA, seed: int = 20260905,
            kind: str = "percentile") -> Interval:
    """Clustered CI for one method's pooled RMSE."""
    idx = bootstrap_indices(stats.n_clusters, n_boot, seed)
    draws = _pooled(stats.se, stats.n, idx)
    point = stats.rmse()
    if kind == "bca":
        lo, hi = _bca_ci(draws, point, _jackknife_pooled(stats.se, stats.n), alpha)
    else:
        lo, hi = _percentile_ci(draws, alpha)
    return Interval(point, lo, hi, stats.n_clusters, stats.cluster_unit,
                    n_boot, alpha, stats.method or "", kind)


def _align(a: ClusterStats, b: ClusterStats) -> tuple[np.ndarray, np.ndarray,
                                                      np.ndarray, np.ndarray,
                                                      np.ndarray]:
    """Put two methods on a common cluster axis.

    A paired bootstrap is only valid if row c of both arrays is the SAME
    cluster.  Two methods evaluated on the same cohort should already agree, but
    a method that skips a month (ECCO stops in 2017) would otherwise silently
    shift the axis and pair unrelated clusters.  We intersect and say so.
    """
    if a.cluster_unit != b.cluster_unit:
        raise ValueError(f"cluster unit mismatch: {a.cluster_unit} vs {b.cluster_unit}")
    common = np.intersect1d(a.cluster_ids, b.cluster_ids)
    if common.size == 0:
        raise ValueError("no clusters in common between the two methods")
    ia = np.searchsorted(a.cluster_ids, common, sorter=np.argsort(a.cluster_ids))
    ia = np.argsort(a.cluster_ids)[ia]
    ib = np.searchsorted(b.cluster_ids, common, sorter=np.argsort(b.cluster_ids))
    ib = np.argsort(b.cluster_ids)[ib]
    return common, a.se[ia], a.n[ia], b.se[ib], b.n[ib]


def ci_difference(a: ClusterStats, b: ClusterStats, n_boot: int = DEFAULT_B,
                  alpha: float = DEFAULT_ALPHA, seed: int = 20260905,
                  kind: str = "percentile") -> Interval:
    """Paired clustered CI for ``RMSE(a) - RMSE(b)``.

    Sign convention: negative means ``a`` is better.  ``excludes_zero`` on the
    result is precisely the plan's "CI inclusion/exclusion of zero" test.
    """
    common, ase, an, bse, bn = _align(a, b)
    idx = bootstrap_indices(common.size, n_boot, seed)
    draws = _pooled(ase, an, idx) - _pooled(bse, bn, idx)
    point = (float(np.sqrt(ase.sum() / an.sum())) -
             float(np.sqrt(bse.sum() / bn.sum())))
    if kind == "bca":
        lo, hi = _bca_ci(draws, point,
                         _jackknife_pooled(ase, an) - _jackknife_pooled(bse, bn),
                         alpha)
    else:
        lo, hi = _percentile_ci(draws, alpha)
    return Interval(point, lo, hi, int(common.size), a.cluster_unit, n_boot,
                    alpha, f"{a.method or 'a'} - {b.method or 'b'}", kind)


def ci_did(a_hi: ClusterStats, a_lo: ClusterStats,
           b_hi: ClusterStats, b_lo: ClusterStats,
           n_boot: int = DEFAULT_B, alpha: float = DEFAULT_ALPHA,
           seed: int = 20260905, kind: str = "percentile") -> Interval:
    """Paired CI for the plan's P1 difference-in-differences.

        layout_gap(m) = J_clustered(m) - J_dispersed(m)
        DiD           = layout_gap(b) - layout_gap(a)

    Called as ``ci_did(dfs_clustered, dfs_dispersed, uni_clustered, uni_dispersed)``
    this returns ``layout_gap(Uniform) - layout_gap(DFS)``, matching the plan's
    stated sign: **positive means Uniform is hurt more by clustering than DFS is**,
    which is the direction the DFS claim predicts.

    All four arms share one draw of cluster indices.  They must: the same month
    supplies the clustered and dispersed layout for both methods, so treating the
    four arms as independent would inflate the interval enormously.
    """
    for x in (a_lo, b_hi, b_lo):
        if x.cluster_unit != a_hi.cluster_unit:
            raise ValueError("all four DiD arms need the same cluster unit")
    common = a_hi.cluster_ids
    for x in (a_lo, b_hi, b_lo):
        common = np.intersect1d(common, x.cluster_ids)
    if common.size == 0:
        raise ValueError("no clusters common to all four DiD arms")

    def take(s: ClusterStats):
        order = np.argsort(s.cluster_ids)
        pos = order[np.searchsorted(s.cluster_ids, common, sorter=order)]
        return s.se[pos], s.n[pos]

    (ahs, ahn), (als, aln) = take(a_hi), take(a_lo)
    (bhs, bhn), (bls, bln) = take(b_hi), take(b_lo)
    idx = bootstrap_indices(common.size, n_boot, seed)
    gap_a = _pooled(ahs, ahn, idx) - _pooled(als, aln, idx)
    gap_b = _pooled(bhs, bhn, idx) - _pooled(bls, bln, idx)
    draws = gap_b - gap_a
    r = lambda s, n: float(np.sqrt(s.sum() / n.sum()))
    point = ((r(bhs, bhn) - r(bls, bln)) - (r(ahs, ahn) - r(als, aln)))
    lo, hi = _percentile_ci(draws, alpha)
    return Interval(point, lo, hi, int(common.size), a_hi.cluster_unit, n_boot,
                    alpha, "DiD = gap(b) - gap(a)", kind)


# --------------------------------------------------------------------------
# seed variance — the component a single-model bootstrap cannot see
# --------------------------------------------------------------------------
# Measured on the real P0 Gulf Stream run: pooled RMSE moved by +/-0.02 across
# training seeds while DFS - Uniform was ~0.005.  A cluster bootstrap resamples
# FLOATS but conditions on one trained model, so it reports the uncertainty in
# "this model's score" when the quantity of interest is "a model trained by this
# recipe".  Seed 1234 accordingly produced a DFS - Uniform interval excluding
# zero that seeds 1235 and 1236 contradicted.
#
# The estimand these functions target is therefore the score of a model trained
# with a RANDOM seed.  Seeds are resampled with replacement, and within each
# drawn seed its floats are resampled too, so both variance components enter.
#
# A caution that belongs in the code rather than only in a report: with three
# seeds the between-seed component has 2 degrees of freedom and is estimated
# very poorly.  These intervals are honest about direction and about "does it
# cross zero", not precise.  ``seed_t_interval`` gives the small-sample t
# alternative, and the two disagreeing is itself information.


def _seed_arrays(stats_by_seed: Sequence[ClusterStats]):
    out = []
    for st in stats_by_seed:
        if st is None or st.n_clusters == 0:
            continue
        out.append((st.se, st.n))
    if not out:
        raise ValueError("no usable per-seed statistics")
    return out


def ci_rmse_across_seeds(stats_by_seed: Sequence[ClusterStats],
                         n_boot: int = DEFAULT_B, alpha: float = DEFAULT_ALPHA,
                         seed: int = 20260905) -> Interval:
    """Pooled RMSE CI integrating over training seeds AND float clusters."""
    arrs = _seed_arrays(stats_by_seed)
    S = len(arrs)
    rng = np.random.default_rng(seed)
    per_seed_point = np.array([np.sqrt(se.sum() / n.sum()) for se, n in arrs])
    point = float(per_seed_point.mean())
    if S == 1:
        return ci_rmse(stats_by_seed[0], n_boot, alpha, seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        vals = []
        for si in rng.integers(0, S, S):
            se, n = arrs[si]
            idx = rng.integers(0, se.size, se.size)
            tot_n = n[idx].sum()
            vals.append(np.sqrt(se[idx].sum() / tot_n) if tot_n > 0 else np.nan)
        draws[b] = np.nanmean(vals)
    lo, hi = _percentile_ci(draws, alpha)
    first = next(s for s in stats_by_seed if s is not None)
    return Interval(point, lo, hi, first.n_clusters, first.cluster_unit,
                    n_boot, alpha, first.method or "", "seed+cluster", S)


def ci_difference_across_seeds(a_by_seed: Sequence[ClusterStats],
                               b_by_seed: Sequence[ClusterStats],
                               n_boot: int = DEFAULT_B,
                               alpha: float = DEFAULT_ALPHA,
                               seed: int = 20260905) -> Interval:
    """Paired ``RMSE(a) - RMSE(b)`` CI over both seeds and float clusters.

    Within a drawn seed the SAME float resample is applied to both methods, so
    the pairing that makes the difference precise is preserved; across seeds the
    pair is drawn together, so a seed that favours ``a`` cannot be matched with
    a different seed that favours ``b``.
    """
    pairs = [(x, y) for x, y in zip(a_by_seed, b_by_seed)
             if x is not None and y is not None and x.n_clusters and y.n_clusters]
    if not pairs:
        raise ValueError("no seed has both methods")
    S = len(pairs)
    if S == 1:
        return ci_difference(pairs[0][0], pairs[0][1], n_boot, alpha, seed)
    aligned = [_align(x, y) for x, y in pairs]
    per_seed_point = np.array([
        np.sqrt(ase.sum() / an.sum()) - np.sqrt(bse.sum() / bn.sum())
        for _, ase, an, bse, bn in aligned])
    point = float(per_seed_point.mean())
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        vals = []
        for si in rng.integers(0, S, S):
            _, ase, an, bse, bn = aligned[si]
            idx = rng.integers(0, ase.size, ase.size)
            ta, tb = an[idx].sum(), bn[idx].sum()
            vals.append((np.sqrt(ase[idx].sum() / ta) - np.sqrt(bse[idx].sum() / tb))
                        if ta > 0 and tb > 0 else np.nan)
        draws[b] = np.nanmean(vals)
    lo, hi = _percentile_ci(draws, alpha)
    a0 = pairs[0][0]
    return Interval(point, lo, hi, int(aligned[0][0].size), a0.cluster_unit,
                    n_boot, alpha,
                    f"{a0.method or 'a'} - {pairs[0][1].method or 'b'}",
                    "seed+cluster", S)


def seed_t_interval(per_seed_values: Sequence[float],
                    alpha: float = DEFAULT_ALPHA) -> Interval:
    """Small-sample t interval on the per-seed point estimates alone.

    Ignores within-seed float uncertainty entirely and treats the seed means as
    the only observations.  With three seeds that is 2 degrees of freedom and
    ``t = 4.303``, so the interval is wide -- deliberately.  It is the plain
    answer to "could this sign be a seed accident?" and needs no bootstrap.
    """
    from scipy import stats as sps
    v = np.asarray([x for x in per_seed_values if x is not None and np.isfinite(x)],
                   dtype=float)
    if v.size < 2:
        m = float(v[0]) if v.size else float("nan")
        return Interval(m, float("nan"), float("nan"), 0, "seed", 0, alpha,
                        "seed-t", "seed_t", int(v.size))
    m = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(v.size))
    t = float(sps.t.ppf(1 - alpha / 2, v.size - 1))
    return Interval(m, m - t * se, m + t * se, 0, "seed", 0, alpha,
                    "seed-t", "seed_t", int(v.size))


def ranking(stats: Iterable[ClusterStats]) -> list[tuple[str, float]]:
    """Methods ordered best-first by pooled RMSE — the S6 'method ranking'."""
    return sorted(((s.method, s.rmse()) for s in stats),
                  key=lambda kv: (np.isnan(kv[1]), kv[1]))
