"""Layout (P1), redundancy (P2) and sparsity (P5) operators on real Argo rows.

Every operator here returns a **row selection** into an `ArgoCohort`, or a
modified sample, and nothing else.  `build_argo_sample` then constructs the
tokens exactly as it always does.  Keeping it that way is deliberate: an
experiment that built its own tokens could change two things at once — the
layout AND the encoding — and the difference would be uninterpretable.

Everything is a pure function of (cohort, month, seed), so two tracks running
the same package on the same cohort select the *same* profiles.  The plan's P1
cross-check explicitly requires "deterministic selection" and "exact profile
count equality"; those are asserted here rather than hoped for.
"""
from __future__ import annotations

import numpy as np

from .argo_obs import ArgoCohort

EARTH_R_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance. Real floats span 25 deg of latitude, where a flat
    lat/lon metric is wrong by tens of percent — and the layout experiment is
    *about* distance, so the metric cannot be the approximation."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def mean_pairwise_km(c: ArgoCohort, rows: np.ndarray) -> float:
    """The P1 cross-check statistic: mean pairwise separation of a layout."""
    if rows.size < 2:
        return float("nan")
    la, lo = c.lat[rows], c.lon[rows]
    d = haversine_km(la[:, None], lo[:, None], la[None, :], lo[None, :])
    iu = np.triu_indices(rows.size, k=1)
    return float(np.mean(d[iu]))


# --------------------------------------------------------------------------
# P1 — layout
# --------------------------------------------------------------------------
def layout(c: ArgoCohort, month: int, n: int, kind: str, seed: int,
           float_split: str = "cohort_float") -> np.ndarray:
    """Pick exactly ``n`` whole real profiles in one of three arrangements.

    ``natural``    a uniform random draw — the layout the array actually had.
    ``clustered``  the ``n`` profiles closest to a seed profile, i.e. a dense
                   local patch.  This is the pathology: many observations of
                   nearly the same water.
    ``dispersed``  a greedy farthest-point sweep, maximising minimum separation.

    All three return the SAME count from the SAME month and the SAME float
    split, so the only thing that varies is where the observations sit.  If the
    month cannot supply ``n``, every arm is capped to what it can supply — an
    empty array if that is zero — so the counts stay equal by construction
    rather than by luck.

    The greedy sweep is deterministic given the seed profile, and the seed
    profile is drawn from the registered RNG, so ``dispersed`` is reproducible.
    """
    rows = c.month(month, float_split=float_split)
    if rows.size == 0 or n <= 0:
        return rows[:0]
    n = min(n, rows.size)
    rng = np.random.default_rng([seed, month, n])
    if kind == "natural":
        return np.sort(rng.choice(rows, n, replace=False))

    la, lo = c.lat[rows], c.lon[rows]
    start = int(rng.integers(0, rows.size))
    if kind == "clustered":
        d = haversine_km(la[start], lo[start], la, lo)
        return np.sort(rows[np.argsort(d, kind="stable")[:n]])
    if kind == "dispersed":
        chosen = [start]
        dmin = haversine_km(la[start], lo[start], la, lo)
        while len(chosen) < n:
            nxt = int(np.argmax(dmin))
            if nxt in chosen:                       # exhausted distinct sites
                remaining = [i for i in range(rows.size) if i not in chosen]
                if not remaining:
                    break
                nxt = remaining[0]
            chosen.append(nxt)
            dmin = np.minimum(dmin, haversine_km(la[nxt], lo[nxt], la, lo))
        return np.sort(rows[np.array(chosen)])
    raise ValueError(f"unknown layout {kind!r}")


LAYOUTS = ("natural", "clustered", "dispersed")


def layout_report(c: ArgoCohort, month: int, n: int, seed: int) -> dict:
    """The P1 verification table: counts equal, distances ordered as intended."""
    out = {}
    for k in LAYOUTS:
        r = layout(c, month, n, k, seed)
        out[k] = dict(n=int(r.size), mean_pairwise_km=mean_pairwise_km(c, r),
                      n_floats=int(np.unique(c.wmo[r]).size))
    return out


# --------------------------------------------------------------------------
# P2 — redundancy families
# --------------------------------------------------------------------------
REDUNDANCY_FAMILIES = ("exact", "jittered", "same_provenance",
                       "independent_provenance", "separated")


def duplicate_rows(c: ArgoCohort, base: np.ndarray, k: int, family: str,
                   seed: int, month: int,
                   jitter_km: float = 25.0) -> tuple[np.ndarray, dict]:
    """Build the k-fold redundancy stress for one family.

    Returns ``(rows, edits)`` where ``edits`` describes per-copy modifications
    the sample builder must apply (coordinate jitter, provenance override).
    Rows are duplicated by INDEX, so a duplicate carries the identical measured
    values — which is what "exact duplicate" has to mean.

        exact                   k bit-identical copies of one profile
        jittered                k copies moved by ~jitter_km, i.e. the same
                                water reported at slightly different positions
        same_provenance         k copies sharing ONE platform id: the
                                dual-stream float, the paper's opening example
        independent_provenance  k copies with DISTINCT platform ids — same
                                water, genuinely independent instruments
        separated               k genuinely different profiles, far apart.
                                THE POSITIVE CONTROL. If evidence does not grow
                                here, the estimator is not suppressing
                                redundancy, it is ignoring profiles.

    The distinction between ``same_provenance`` and ``independent_provenance``
    is the one real Argo makes available and synthetic columns could not: both
    are k observations of the same water, and only the first should collapse.
    """
    if base.size == 0:
        return base, {}
    anchor = int(base[0])
    if family == "separated":
        pool = c.month(month, float_split="cohort_float")
        if pool.size < k:
            k = max(pool.size, 1)
        la, lo = c.lat[pool], c.lon[pool]
        chosen = [int(np.flatnonzero(pool == anchor)[0])] if anchor in pool else [0]
        dmin = haversine_km(la[chosen[0]], lo[chosen[0]], la, lo)
        while len(chosen) < k:
            nxt = int(np.argmax(dmin))
            if nxt in chosen:
                break
            chosen.append(nxt)
            dmin = np.minimum(dmin, haversine_km(la[nxt], lo[nxt], la, lo))
        rows = np.r_[pool[np.array(chosen)], base[1:]]
        return rows, {"provenance_override": None, "jitter_km": np.zeros(len(chosen))}

    edits: dict = {}
    # The family must enter the seed by its INDEX, not by `hash(family)`.
    # Python randomizes str hashing per process (PYTHONHASHSEED), so
    # `abs(hash(family))` gave `jittered` a different set of coordinate offsets
    # on every run -- the one family whose construction draws random numbers was
    # the one that could not be reproduced, including by the same code on the
    # same machine an hour later. Two tracks running concurrently disagreed on
    # it while agreeing exactly on the other four, which is how it surfaced.
    rng = np.random.default_rng([seed, month, k, REDUNDANCY_FAMILIES.index(family)])

    if family in ("same_provenance", "independent_provenance"):
        # These two must differ in exactly ONE respect: how many platforms the
        # k reports came from.  Repeating the anchor row for both (an earlier
        # draft did) makes them the same construction with a relabelled
        # provenance vector, and they then produce identical curves by
        # arithmetic rather than by physics.
        #
        #   same_provenance         k reports from ONE float -- its own nearby
        #                           cycles, i.e. one instrument seen k times
        #   independent_provenance  k reports of the same water from k
        #                           DIFFERENT floats
        #
        # Both are k observations of nearly the same water; only the first is
        # genuinely one measurement re-ingested, and only it should collapse.
        pool = c.month(month, float_split="cohort_float")
        a_lat, a_lon = c.lat[anchor], c.lon[anchor]
        d = haversine_km(a_lat, a_lon, c.lat[pool], c.lon[pool])
        if family == "same_provenance":
            same = pool[c.wmo[pool] == c.wmo[anchor]]
            if same.size >= 2:
                order = np.argsort(haversine_km(a_lat, a_lon, c.lat[same],
                                                c.lon[same]), kind="stable")
                pick = same[order]
                sel = np.resize(pick, k)      # cycle through if the float has < k
            else:
                sel = np.full(k, anchor)      # single-cycle float: exact copies
            edits["provenance_override"] = np.zeros(k, dtype=int)
        else:
            # nearest profiles that come from OTHER floats
            other = pool[c.wmo[pool] != c.wmo[anchor]]
            if other.size:
                order = np.argsort(haversine_km(a_lat, a_lon, c.lat[other],
                                                c.lon[other]), kind="stable")
                by_float, seen = [], set()
                for i in order:
                    w = c.wmo[other[i]]
                    if w not in seen:
                        seen.add(w); by_float.append(other[i])
                    if len(by_float) >= k:
                        break
                sel = np.resize(np.array(by_float), k)
            else:
                sel = np.full(k, anchor)
            edits["provenance_override"] = np.arange(k, dtype=int)
        edits["n_distinct_rows"] = int(np.unique(sel).size)
        return np.r_[sel, base[1:]], edits

    rows = np.r_[np.full(k, anchor), base[1:]]
    if family == "jittered":
        edits["jitter_km"] = rng.normal(0.0, jitter_km, size=k)
    edits["provenance_override"] = np.zeros(k, dtype=int)   # one platform
    edits["n_distinct_rows"] = 1
    return rows, edits


# --------------------------------------------------------------------------
# P5 — sparsity
# --------------------------------------------------------------------------
SPARSITY_FRACTIONS = (1.0, 0.75, 0.50, 0.25, 0.10)


def thin(c: ArgoCohort, rows: np.ndarray, fraction: float, seed: int,
         month: int) -> np.ndarray:
    """Keep a fraction of the input profiles.

    The realization depends on ``(seed, month, fraction)`` and NOT on the
    method, so every method under test sees the identical thinned set — the
    plan's "identical thinning seeds / realizations across methods".  Thinning
    per method would confound the sparsity response with the draw.

    Thinning is by PROFILE, not by float: dropping whole floats would change the
    provenance structure at the same time as the density, and P2 is the
    experiment about provenance.
    """
    if rows.size == 0 or fraction >= 1.0:
        return rows
    n = max(1, int(round(fraction * rows.size)))
    rng = np.random.default_rng([seed, month, int(fraction * 10000)])
    return np.sort(rng.choice(rows, n, replace=False))


def absolute_budget(c: ArgoCohort, rows: np.ndarray, budget: int, seed: int,
                    month: int) -> np.ndarray:
    """The plan's optional absolute-profile budget, same determinism rule."""
    if rows.size <= budget:
        return rows
    rng = np.random.default_rng([seed, month, budget])
    return np.sort(rng.choice(rows, budget, replace=False))
