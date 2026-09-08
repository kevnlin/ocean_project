"""Comparing two tracks' artifacts under the plan's S6 acceptance rules.

The plan draws a hard line that this module enforces:

* **Exact-match quantities** (S6.1) — manifest hashes, cohort ids, counts,
  registered hyperparameters.  "Any mismatch here is a protocol problem."  A
  difference is an ERROR; there is no tolerance to tune.

* **Numerical-result quantities** (S6.2) — independently recomputed metrics.
  "Small floating-point differences are acceptable."  But five specific
  qualitative changes force the tracks to stop and reconcile before the result
  is used, and those are checked as booleans, never eyeballed off a table.

The distinction is the whole point.  A tolerance applied to a manifest hash
would let two tracks run on different data and call it agreement; a hard
equality applied to a bootstrap CI endpoint would flag every legitimate run.

One track, honestly labelled
----------------------------
This repository currently produces Track A only.  `render_report` therefore
emits Track B's columns as ``pending`` rather than filling them, and
`compare_artifacts` raises if handed two artifacts with the same ``track``
label.  That is deliberate: the plan's S8 rationale is that an independent
implementation "reduces the chance that one implementation is unconsciously
adjusted to match the other", and a second column written by the same author
does not do that.  The harness is built so the intern's artifacts drop in.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from .protocol import EXACT_MATCH_FIELDS, ResultArtifact

#: relative tolerance for "the same number, recomputed independently".
#: 1e-6 is far tighter than any real reimplementation difference and far looser
#: than float noise, so it separates "different code, same maths" from "different
#: maths" without needing to be tuned per metric.
DEFAULT_RTOL = 1e-6

#: Absolute floor. `DFS - Uniform` is ~6e-4, so an absolute disagreement of
#: 9e-9 -- pure float associativity between a pandas groupby and a numpy
#: bincount -- is a 1.5e-5 RELATIVE difference and would be reported as a real
#: one. Any quantity here is an RMSE or a difference of RMSEs in z units, where
#: 1e-8 is far below anything that could change a conclusion.
DEFAULT_ATOL = 1e-8

#: Bootstrap CI ENDPOINTS are Monte-Carlo estimates.  Two tracks that resample
#: differently on purpose -- index resampling versus multinomial weights -- will
#: not reproduce an endpoint to 1e-6 no matter how correct both are, while their
#: point estimates should agree to float precision.  Holding endpoints to the
#: same tolerance as points reports Monte-Carlo noise as a disagreement, so they
#: get their own budget.  What must still agree exactly is the QUALITATIVE
#: verdict -- `excludes_zero` -- and that is checked as a conclusion, not a
#: number.
MC_RTOL = 0.15
_MC_FIELDS = ("lo", "hi", "ci_lo", "ci_hi")


@dataclass
class Mismatch:
    field: str
    a: Any
    b: Any
    kind: str                    # "exact" | "numeric" | "conclusion"
    detail: str = ""

    def __str__(self) -> str:
        return f"[{self.kind}] {self.field}: A={self.a!r} B={self.b!r} {self.detail}".strip()


@dataclass
class CrossCheck:
    package: str
    region: str
    exact_mismatches: list = field(default_factory=list)
    numeric_diffs: list = field(default_factory=list)
    conclusion_changes: list = field(default_factory=list)

    @property
    def protocol_ok(self) -> bool:
        return not self.exact_mismatches

    @property
    def same_conclusion(self) -> bool:
        return not self.conclusion_changes

    @property
    def must_reconcile(self) -> bool:
        """S6: stop and reconcile before the result is used in the paper."""
        return bool(self.exact_mismatches or self.conclusion_changes)

    def summary(self) -> str:
        if self.exact_mismatches:
            return "PROTOCOL PROBLEM"
        if self.conclusion_changes:
            return "RECONCILE — scientific conclusion differs"
        return "agree"


def _walk(prefix: str, a: Any, b: Any, out: list) -> None:
    """Recursively compare two exact-match structures."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            _walk(f"{prefix}.{k}" if prefix else k,
                  a.get(k, "<absent>"), b.get(k, "<absent>"), out)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append(Mismatch(prefix, a, b, "exact", "different lengths"))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                _walk(f"{prefix}[{i}]", x, y, out)
    elif a != b:
        out.append(Mismatch(prefix, a, b, "exact"))


def compare_exact(a: ResultArtifact, b: ResultArtifact) -> list[Mismatch]:
    """S6.1 — every difference here is a protocol problem."""
    out: list[Mismatch] = []
    for f in EXACT_MATCH_FIELDS:
        _walk(f, getattr(a, f), getattr(b, f), out)
    return out


def _rel(x: float, y: float) -> float:
    if x == y:
        return 0.0
    d = max(abs(x), abs(y))
    return abs(x - y) / d if d > 0 else float("inf")


def compare_numeric(a: dict, b: dict, rtol: float = DEFAULT_RTOL,
                    prefix: str = "") -> list[Mismatch]:
    """S6.2 — report every numeric difference above rtol, but do not fail on it."""
    out: list[Mismatch] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            key = f"{prefix}.{k}" if prefix else k
            if k not in a or k not in b:
                out.append(Mismatch(key, a.get(k, "<absent>"),
                                    b.get(k, "<absent>"), "numeric",
                                    "present in only one track"))
                continue
            out += compare_numeric(a[k], b[k], rtol, key)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, bool) or isinstance(b, bool):
            if a != b:
                out.append(Mismatch(prefix, a, b, "numeric"))
        elif math.isnan(a) and math.isnan(b):
            pass
        else:
            leaf = prefix.rsplit(".", 1)[-1]
            tol = MC_RTOL if leaf in _MC_FIELDS else rtol
            if abs(float(a) - float(b)) <= DEFAULT_ATOL:
                return out
            r = _rel(float(a), float(b))
            if r > tol:
                note = (f"rel={r:.3e}" +
                        (" (bootstrap Monte-Carlo, beyond the MC budget)"
                         if leaf in _MC_FIELDS else ""))
                out.append(Mismatch(prefix, a, b, "numeric", note))
    return out


# --------------------------------------------------------------------------
# S6 — the five conditions that force reconciliation
# --------------------------------------------------------------------------
def _sign(x: float | None) -> int | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return (x > 0) - (x < 0)


def coverage_note(a: dict, b: dict) -> list[Mismatch]:
    """Rows only one track scored. Reported, but never a conclusion change.

    Two tracks may legitimately cover different rows -- Track B need not
    re-derive a shared gridded-reference adapter to verify the learned
    evaluation path. Classifying that as a scientific disagreement made the
    whole cross-check read RECONCILE when every shared number agreed.
    """
    ra, rb = a.get("ranking"), b.get("ranking")
    if ra is None or rb is None:
        return []
    na = [m for m, *_ in ra] if ra and isinstance(ra[0], (list, tuple)) else list(ra)
    nb = [m for m, *_ in rb] if rb and isinstance(rb[0], (list, tuple)) else list(rb)
    only_a, only_b = sorted(set(na) - set(nb)), sorted(set(nb) - set(na))
    if not (only_a or only_b):
        return []
    return [Mismatch("ranking_coverage", only_a, only_b, "numeric",
                     "rows scored by only one track — not a disagreement; the "
                     "ranking comparison is restricted to the rows in common")]


def check_conclusions(a: dict, b: dict) -> list[Mismatch]:
    """The plan's five stopping conditions, evaluated as booleans.

    ``a`` and ``b`` are each a track's ``results`` dict, expected to carry
    whichever of these keys the package produced:

        dfs_minus_uniform     {"point": float, "excludes_zero": bool}
        layout_did            {"point": float, "excludes_zero": bool}
        ranking               [method, ...] best-first
        redundancy_control    {"passes": bool}

    A key absent from BOTH tracks is simply not part of that package and is
    skipped.  A key present in only one is itself a mismatch — one track
    computed something the other did not, which the report must show rather
    than silently drop.
    """
    out: list[Mismatch] = []

    for key in ("dfs_minus_uniform", "layout_did"):
        pa, pb = a.get(key), b.get(key)
        if pa is None and pb is None:
            continue
        if pa is None or pb is None:
            out.append(Mismatch(key, pa, pb, "conclusion",
                                "computed by only one track"))
            continue
        sa, sb = _sign(pa.get("point")), _sign(pb.get("point"))
        if sa != sb:
            out.append(Mismatch(f"{key}.sign", sa, sb, "conclusion",
                                "the effect points in opposite directions"))
        ea, eb = pa.get("excludes_zero"), pb.get("excludes_zero")
        if ea is not None and eb is not None and bool(ea) != bool(eb):
            out.append(Mismatch(f"{key}.excludes_zero", ea, eb, "conclusion",
                                "one track calls it significant, the other does not"))

    ra, rb = a.get("ranking"), b.get("ranking")
    if ra is not None and rb is not None:
        na = [m for m, *_ in ra] if ra and isinstance(ra[0], (list, tuple)) else list(ra)
        nb = [m for m, *_ in rb] if rb and isinstance(rb[0], (list, tuple)) else list(rb)
        # Compare the ORDER of the rows both tracks scored.  Two tracks may
        # legitimately cover different row sets -- Track B need not re-derive a
        # shared gridded-reference adapter to verify the learned evaluation
        # path -- and comparing lists of different membership reports a ranking
        # change that is really a coverage difference.  Coverage is reported
        # separately so it cannot hide.
        common = [m for m in na if m in set(nb)]
        common_b = [m for m in nb if m in set(na)]
        if not common:
            out.append(Mismatch("ranking", na, nb, "conclusion",
                                "the tracks scored no row in common"))
        elif common != common_b:
            out.append(Mismatch("ranking", common, common_b, "conclusion",
                                "method ranking differs on the rows both "
                                "tracks scored"))

    elif (ra is None) != (rb is None):
        out.append(Mismatch("ranking", ra, rb, "conclusion",
                            "computed by only one track"))

    ca, cb = a.get("redundancy_control"), b.get("redundancy_control")
    if ca is not None and cb is not None:
        if bool(ca.get("passes")) != bool(cb.get("passes")):
            out.append(Mismatch("redundancy_control.passes",
                                ca.get("passes"), cb.get("passes"), "conclusion",
                                "the positive control disagrees — the redundancy "
                                "result may not be used until this is resolved"))
    return out


# --------------------------------------------------------------------------
# the comparison schema
# --------------------------------------------------------------------------
#: Two independent implementations will not share an internal results layout --
#: that is the point of them being independent.  Comparing raw `results` dicts
#: therefore drowns the real signal in "present in only one track" noise about
#: nesting.  Each track instead emits this fixed, flat schema, and the
#: cross-check compares THAT.  Anything a track wants to keep that is not
#: comparable stays in `results` and is simply not compared.
COMPARABLE_SCHEMA = ("rmse", "per_seed_rmse", "n_wmos", "n_targets",
                     "n_months", "dfs_minus_uniform", "ranking",
                     "layout_did", "redundancy_control")


def build_comparable(*, rmse: dict | None = None,
                     per_seed_rmse: dict | None = None,
                     n_wmos=None, n_targets=None, n_months=None,
                     dfs_minus_uniform: dict | None = None,
                     ranking=None, layout_did: dict | None = None,
                     redundancy_control: dict | None = None) -> dict:
    """Assemble the flat block both tracks emit.

    ``rmse`` is {row: {channel: seed-averaged value}}; ``per_seed_rmse`` keeps
    the individual seeds, because plan S5 asks for per-seed values and because
    an average can agree while the seeds behind it do not.
    """
    out = {}
    for k, v in (("rmse", rmse), ("per_seed_rmse", per_seed_rmse),
                 ("n_wmos", n_wmos), ("n_targets", n_targets),
                 ("n_months", n_months),
                 ("dfs_minus_uniform", dfs_minus_uniform),
                 ("ranking", ranking), ("layout_did", layout_did),
                 ("redundancy_control", redundancy_control)):
        if v is not None:
            out[k] = v
    return out


def compare_artifacts(a: ResultArtifact, b: ResultArtifact,
                      rtol: float = DEFAULT_RTOL) -> CrossCheck:
    """Full S5/S6 comparison of two tracks' artifacts for one package."""
    if a.track == b.track:
        raise ValueError(
            f"both artifacts are labelled track {a.track!r}. A cross-check "
            f"between two runs of the same track is not the independence the "
            f"plan's S8 asks for; label them A and B deliberately.")
    if a.package != b.package:
        raise ValueError(f"package mismatch: {a.package} vs {b.package}")
    if a.region != b.region:
        raise ValueError(f"region mismatch: {a.region} vs {b.region}")
    # Prefer the flat comparison block when both tracks emit one; fall back to
    # the raw results only when a track predates the schema.
    ca = a.results.get("comparable")
    cb = b.results.get("comparable")
    if ca is None or cb is None:
        ca, cb = a.results, b.results
    return CrossCheck(
        package=a.package, region=a.region,
        exact_mismatches=compare_exact(a, b),
        numeric_diffs=compare_numeric(ca, cb, rtol) + coverage_note(ca, cb),
        conclusion_changes=check_conclusions(ca, cb))


# --------------------------------------------------------------------------
# S5 — rendering
# --------------------------------------------------------------------------
def render_report(package: str, region: str, a: ResultArtifact | None,
                  b: ResultArtifact | None,
                  cc: CrossCheck | None = None) -> str:
    """Markdown for one package's cross-check, with pending Track B columns."""
    L = [f"# Cross-check — {package} ({region})", ""]
    if b is None:
        L += ["> **Track B is pending.** Track A ran; the intern's independent "
              "run has not been supplied. The comparison columns below are "
              "placeholders, and no agreement should be inferred from them.",
              ""]
    L += ["## Provenance", "",
          "| field | Track A | Track B |", "|---|---|---|"]
    for label, attr in (("protocol version", "protocol_version"),
                        ("protocol hash", "protocol_hash_"),
                        ("git commit", "git_commit_"),
                        ("git dirty", "git_dirty_"),
                        ("seeds", "seeds"),
                        ("created (UTC)", "created_utc")):
        va = getattr(a, attr, "—") if a else "—"
        vb = getattr(b, attr, "pending") if b else "pending"
        L.append(f"| {label} | `{va}` | `{vb}` |")
    L.append("")

    if a:
        L += ["## Data manifests", "",
              "| dataset | Track A sha256 | Track B sha256 |", "|---|---|---|"]
        for k, v in sorted(a.data_manifests.items()):
            vb = (b.data_manifests.get(k, "—") if b else "pending")
            L.append(f"| {k} | `{str(v)[:16]}` | `{str(vb)[:16]}` |")
        L.append("")
        if a.counts:
            L += ["## Counts (exact-match quantities)", "",
                  "| quantity | Track A | Track B |", "|---|---:|---:|"]
            for k, v in sorted(a.counts.items()):
                vb = (b.counts.get(k, "—") if b else "pending")
                L.append(f"| {k} | {v} | {vb} |")
            L.append("")

    if cc is not None:
        L += ["## Acceptance (plan S6)", "",
              f"**Verdict: {cc.summary()}**", ""]
        if cc.exact_mismatches:
            L += ["### Protocol problems — exact-match quantities differ", ""]
            L += [f"- {m}" for m in cc.exact_mismatches] + [""]
        if cc.conclusion_changes:
            L += ["### Scientific conclusion differs — stop and reconcile", ""]
            L += [f"- {m}" for m in cc.conclusion_changes] + [""]
        if cc.numeric_diffs:
            L += [f"### Numerical differences above rtol ({len(cc.numeric_diffs)})",
                  ""]
            L += [f"- {m}" for m in cc.numeric_diffs[:40]]
            if len(cc.numeric_diffs) > 40:
                L.append(f"- ... and {len(cc.numeric_diffs) - 40} more")
            L.append("")

    if a and a.warnings:
        L += ["## Warnings", ""] + [f"- {w}" for w in a.warnings] + [""]
    return "\n".join(L)
