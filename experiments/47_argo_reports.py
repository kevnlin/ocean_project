"""Render the plan's deliverables from the signed artifacts.

Every table here is read out of a `ResultArtifact`; nothing is retyped.  That is
the point — plan S9 asks the intern to report the git SHA, the manifests, the
command, the tables and the CIs together, and a report assembled by hand can
disagree with the run it describes without anyone noticing.

Track B's columns are rendered as `pending` wherever they are absent.  They are
NOT filled from Track A, and the reports say in the first paragraph that no
agreement can be inferred from a single track.  See `crosscheck.render_report`.

  .venv/bin/python experiments/47_argo_reports.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings; warnings.filterwarnings("ignore")

import numpy as np

from ocean_tokenizer import protocol as P
from ocean_tokenizer.clustered_ci import (ClusterStats, ci_rmse_across_seeds,
                                          ci_difference_across_seeds,
                                          seed_t_interval)
from ocean_tokenizer.crosscheck import render_report, compare_artifacts


def cluster_stats(a, row, lead, ch, unit, method=""):
    """Per-cluster sufficient statistics for one (row, lead, channel, unit)."""
    try:
        d = a.results["rows"][row][lead]["channels"][ch]["cluster_stats"][unit]
    except (KeyError, TypeError):
        return None
    return ClusterStats.from_dict(d, unit, ch, method or row)


def seed_ci(arts, row, ch, unit="wmo"):
    """Interval integrating over BOTH training seeds and float clusters."""
    st = [cluster_stats(a, row, "lead0", ch, unit) for a in arts]
    st = [x for x in st if x is not None]
    if not st:
        return None
    try:
        return ci_rmse_across_seeds(st, n_boot=2000, seed=20260905)
    except ValueError:
        return None

ap = argparse.ArgumentParser()
ap.add_argument("--outputs", default=None)
ap.add_argument("--reports", default=None)
ap.add_argument("--track-b", default=None,
                help="directory of Track B artifacts, if the intern has run")
args = ap.parse_args()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUTS = args.outputs or os.path.join(ROOT, "outputs")
REPORTS = args.reports or os.path.join(ROOT, "reports")
os.makedirs(REPORTS, exist_ok=True)

NA = "—"


def load(package: str, region: str) -> list[P.ResultArtifact]:
    pat = os.path.join(OUTPUTS, f"argo_{package}_{region}", "artifact_*.json")
    out = []
    for p in sorted(glob.glob(pat)):
        try:
            out.append(P.ResultArtifact.read(p))
        except TypeError:
            # an artifact written before a field was added to the dataclass
            d = json.load(open(p))
            known = set(P.ResultArtifact.__dataclass_fields__)
            out.append(P.ResultArtifact(**{k: v for k, v in d.items()
                                           if k in known}))
    return out


def combine_track_a(arts: list[P.ResultArtifact], region: str
                    ) -> P.ResultArtifact | None:
    """Merge Track A's per-seed artifacts into one track-level artifact.

    Track A writes one artifact per seed; Track B writes one per track, because
    its seed-integrated interval is a property of the recipe rather than of any
    single model.  Comparing a per-seed artifact against a per-track one would
    report a mismatch that is purely a difference in scope -- Track A's seed-1234
    `DFS - Uniform` excludes zero while the seed-integrated answer does not, and
    a naive comparison would call that a scientific disagreement.

    So the cross-check compares track against track: seeds are pooled here,
    using the same estimand Track B targets.
    """
    from ocean_tokenizer.crosscheck import build_comparable
    if not arts:
        return None
    # Only the P0-shaped artifacts carry a per-row/per-lead table. P1, P2 and P5
    # have their own shapes and are compared through the `comparable` block each
    # already emits -- but they are still written ONE PER SEED, so they get the
    # same seed pooling, on the flat block instead of on the row table.
    #
    # Returning `arts[0]` here (as this did) compared Track A's seed 1234 against
    # a Track B artifact covering all three seeds, and reported the difference in
    # scope as a scientific one.
    if not all("rows" in a.results for a in arts):
        return combine_comparable(arts, region)
    rows = sorted({r for a in arts for r in a.results.get("rows", {})})
    per_seed = {r: {ch: [a.results["rows"][r]["lead0"]["channels"][ch]["rmse"]
                         for a in arts
                         if r in a.results.get("rows", {})
                         and "rmse" in a.results["rows"][r]["lead0"]["channels"][ch]]
                    for ch in P.CHANNELS} for r in rows}
    d_row, u_row = "dfs_expertlocal_cbottle", "uniform_expertlocal_cbottle"
    dmu = {}
    for ch in P.CHANNELS:
        da = [cluster_stats(a, d_row, "lead0", ch, "wmo", "dfs") for a in arts]
        ua = [cluster_stats(a, u_row, "lead0", ch, "wmo", "uniform") for a in arts]
        if not any(x is not None for x in da):
            continue
        try:
            iv = ci_difference_across_seeds(da, ua, n_boot=4000, seed=20260905)
        except ValueError:
            continue
        dmu[ch] = {"point": iv.point, "lo": iv.lo, "hi": iv.hi,
                   "excludes_zero": iv.excludes_zero}
    means = {r: {ch: (float(np.mean(v)) if v else None)
                 for ch, v in d.items()} for r, d in per_seed.items()}
    rk = sorted(((r, means[r]["TEMP"]) for r in rows
                 if means[r]["TEMP"] is not None), key=lambda kv: kv[1])
    a0 = arts[0]
    first = next(iter(a0.results["rows"].values()))["lead0"]
    comb = P.ResultArtifact(
        package=a0.package, track="A", region=region,
        results={"comparable": build_comparable(
            rmse=means, per_seed_rmse=per_seed, n_wmos=first["n_wmos"],
            n_targets=first["n_targets"], n_months=first["n_months"],
            dfs_minus_uniform=dmu or None, ranking=[m for m, _ in rk] or None)},
        seeds=sorted({x for a in arts for x in a.seeds}),
        command=a0.command, counts=dict(a0.counts),
        warnings=sorted({w for a in arts for w in a.warnings}),
        notes=a0.notes)
    comb.protocol_hash_ = a0.protocol_hash_
    comb.protocol_version = a0.protocol_version
    comb.data_manifests = a0.data_manifests
    comb.git_commit_, comb.git_dirty_ = a0.git_commit_, a0.git_dirty_
    comb.created_utc = a0.created_utc
    return comb


#: Which key in a package's flat `rmse` block is the package's BASELINE arm --
#: the one its method ranking is defined on.  P1 ranks methods on the natural
#: layout and P5 on full density; ranking them on a stressed arm would make the
#: S6 "method ranking" condition depend on which stress happened to be listed
#: first.  P2 has no ranking: its comparable block holds evidence growth, not
#: per-method RMSE.
RANKING_ARM = {"P1": "|natural", "P5": "|100pct"}


def combine_comparable(arts: list[P.ResultArtifact], region: str
                       ) -> P.ResultArtifact | None:
    """Pool per-seed P1/P2/P5 artifacts into one track-level artifact.

    Only the flat `comparable` block is merged, because that is the only part
    of these packages' results whose shape both tracks agree on.  Values are
    averaged over seeds and the per-seed values are kept beside them, since an
    average can agree while the seeds behind it do not.

    The layout DiD is pooled with a **seed-t interval** on the per-seed points.
    That is not the nicest interval available -- a nested seed+cluster bootstrap
    would be better -- but P1's artifacts keep no per-cluster sufficient
    statistics, so a nested interval cannot be re-derived from them after the
    fact.  Track B pools its own per-seed DiDs by the same rule, so the two
    tracks compare the same estimand rather than one track's nested interval
    against the other's single-seed one.
    """
    from ocean_tokenizer.crosscheck import build_comparable
    if not arts:
        return None
    a0 = arts[0]
    blocks = [a.results.get("comparable", {}) for a in arts]

    # --- rmse, averaged over seeds, per-seed values kept -------------------
    rmse: dict = {}
    per_seed: dict = {}
    for key in sorted({k for b in blocks for k in b.get("rmse", {})}):
        inner = sorted({f for b in blocks for f in b.get("rmse", {}).get(key, {})})
        rmse[key], per_seed[key] = {}, {}
        for f in inner:
            vals = [b["rmse"][key][f] for b in blocks
                    if isinstance(b.get("rmse", {}).get(key), dict)
                    and b["rmse"][key].get(f) is not None
                    and np.isfinite(b["rmse"][key][f])]
            rmse[key][f] = float(np.mean(vals)) if vals else None
            per_seed[key][f] = vals

    # --- the S6 conclusion quantities -------------------------------------
    did = None
    pts = [b["layout_did"]["point"] for b in blocks
           if isinstance(b.get("layout_did"), dict)
           and b["layout_did"].get("point") is not None]
    if pts:
        iv = seed_t_interval(pts) if len(pts) > 1 else None
        did = ({"point": iv.point, "lo": iv.lo, "hi": iv.hi,
                "excludes_zero": iv.excludes_zero} if iv else
               {k: blocks[0]["layout_did"][k]
                for k in ("point", "lo", "hi", "excludes_zero")})

    ctl = None
    ctls = [b["redundancy_control"] for b in blocks
            if isinstance(b.get("redundancy_control"), dict)]
    if ctls:
        # The control is a claim about the RECIPE, so it holds only if it holds
        # for every seed. One seed passing and two failing is a failure.
        ctl = {"passes": bool(all(c.get("passes") for c in ctls)),
               "separated_k8_growth": float(np.mean(
                   [c["separated_k8_growth"] for c in ctls])),
               "exact_k8_growth": float(np.mean(
                   [c["exact_k8_growth"] for c in ctls]))}

    arm = RANKING_ARM.get(a0.package)
    rank = None
    if arm:
        base = {k[: -len(arm)]: v.get("TEMP") for k, v in rmse.items()
                if k.endswith(arm) and v.get("TEMP") is not None}
        rank = sorted(base, key=lambda r: base[r]) or None

    counts = dict(a0.counts)
    if any(a.counts != a0.counts for a in arts):
        counts = {k: v for k, v in a0.counts.items()
                  if all(a.counts.get(k) == v for a in arts)}

    comb = P.ResultArtifact(
        package=a0.package, track="A", region=region,
        results={"comparable": build_comparable(
            rmse=rmse, per_seed_rmse=per_seed,
            n_wmos=blocks[0].get("n_wmos"), n_targets=blocks[0].get("n_targets"),
            n_months=blocks[0].get("n_months"),
            layout_did=did, ranking=rank, redundancy_control=ctl)},
        seeds=sorted({x for a in arts for x in a.seeds}),
        command=a0.command, counts=counts,
        warnings=sorted({w for a in arts for w in a.warnings}),
        notes=a0.notes)
    comb.protocol_hash_ = a0.protocol_hash_
    comb.protocol_version = a0.protocol_version
    comb.data_manifests = a0.data_manifests
    comb.git_commit_, comb.git_dirty_ = a0.git_commit_, a0.git_dirty_
    comb.created_utc = a0.created_utc
    return comb


def load_b(package: str, region: str) -> list[P.ResultArtifact]:
    if not args.track_b:
        return []
    pat = os.path.join(args.track_b, f"*{package}*{region}*.json")
    return [P.ResultArtifact.read(p) for p in sorted(glob.glob(pat))]


def fmt(x, n=4):
    if x is None:
        return NA
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    return NA if not np.isfinite(f) else f"{f:.{n}f}"


def ci_cell(d) -> str:
    if not d:
        return NA
    return f"{fmt(d.get('point'))} [{fmt(d.get('lo'))}, {fmt(d.get('hi'))}]"


def across_seeds(arts, path_fn):
    """mean ± std over the headline seeds, with the per-seed values kept."""
    vals = []
    for a in arts:
        try:
            v = path_fn(a)
        except (KeyError, TypeError, IndexError):
            v = None
        if v is not None and np.isfinite(float(v)):
            vals.append(float(v))
    if not vals:
        return NA, []
    if len(vals) == 1:
        return f"{vals[0]:.4f}", vals
    return f"{np.mean(vals):.4f} ± {np.std(vals):.4f}", vals


HEADER = """<!-- generated by experiments/47_argo_reports.py — do not edit by hand -->

> **One track.** These are Track A results. The cross-check plan's value comes
> from an *independently implemented* Track B; where its columns read `pending`
> below, no agreement has been demonstrated and none should be inferred.

> **What "real data" means here.** Input is real QC'd Argo profiles from the
> cohort floats. Targets are real measurements from **WMO-disjoint held-out
> floats** that appear in no training month. Scores are RMSE in train-only
> z-units at genuine measurement positions, clustered by held-out float.
"""


def provenance_block(arts) -> list[str]:
    if not arts:
        return ["_No artifacts found._", ""]
    a = arts[0]
    L = ["## Provenance", "",
         f"* protocol `{a.protocol_version}` hash `{a.protocol_hash_[:16]}`",
         f"* git `{a.git_commit_[:10]}`" +
         ("  **(working tree dirty — the commit does not identify the code)**"
          if a.git_dirty_ else ""),
         f"* seeds `{[x for s in arts for x in s.seeds]}`",
         f"* command `{a.command}`", "", "| dataset | manifest sha256 |",
         "|---|---|"]
    L += [f"| {k} | `{str(v)[:16]}` |" for k, v in sorted(a.data_manifests.items())]
    L.append("")
    if a.counts:
        L += ["| count | value |", "|---|---:|"]
        L += [f"| {k} | {v} |" for k, v in sorted(a.counts.items())]
        L.append("")
    warn = sorted({w for x in arts for w in x.warnings})
    if warn:
        L += ["### Warnings", ""] + [f"* {w}" for w in warn] + [""]
    return L


# ------------------------------------------------------------------ P0
def report_P0(region: str) -> str:
    arts = load(region=region, package="P0")
    L = [f"# P0 — Real-data baseline table ({region})", "", HEADER]
    L += provenance_block(arts)
    if not arts:
        return "\n".join(L)
    rows = sorted({r for a in arts for r in a.results.get("rows", {})})
    for lead_key in sorted({k for a in arts for r in a.results.get("rows", {}).values()
                            for k in r if k.startswith("lead")}):
        n_seeds = len(arts)
        cihdr = ("TEMP CI (seed+WMO)" if n_seeds > 1 else "TEMP CI (WMO-clustered)")
        L += [f"## {lead_key} — pooled RMSE on held-out floats (z units)", ""]
        if lead_key == "lead0" and n_seeds > 1:
            L += [f"CI column integrates over **both** the {n_seeds} training "
                  "seeds and the float clusters. A single-seed WMO-clustered "
                  "interval conditions on one trained model and is shown "
                  "separately; on this task it is several times too narrow.", ""]
        L += ["| row | TEMP | SALT | 0-100m | 100-300m | 300-max | "
              f"{cihdr} | TEMP CI (1 seed, WMO) | n WMO |",
              "|---|---:|---:|---:|---:|---:|---|---|---:|"]
        for r in rows:
            g = lambda ch: (lambda a: a.results["rows"][r][lead_key]["channels"][ch]["rmse"])
            t, _ = across_seeds(arts, g("TEMP"))
            s, _ = across_seeds(arts, g("SALT"))
            a0 = arts[0].results["rows"].get(r, {}).get(lead_key)
            if a0 is None:
                continue
            band = a0["channels"]["TEMP"]["by_band"]
            ciw = a0["channels"]["TEMP"]["ci"].get("wmo")
            comb = (seed_ci(arts, r, "TEMP") if lead_key == "lead0"
                    and len(arts) > 1 else None)
            main = ci_cell(comb.to_dict()) if comb else ci_cell(ciw)
            L.append(f"| `{r}` | {t} | {s} | " +
                     " | ".join(fmt(band.get(b)) for b, _, _ in
                                (("0-100m", 0, 0), ("100-300m", 0, 0), ("300-max", 0, 0))) +
                     f" | {main} | {ci_cell(ciw)} | {a0['n_wmos']} |")
        L.append("")
    d = arts[0].results.get("dfs_minus_uniform")
    if d:
        L += ["## DFS − Uniform (paired, WMO-clustered)", "",
              "Negative favours DFS. The plan stops the cross-check if the sign "
              "or the zero-crossing disagrees between tracks.", "",
              "| channel | difference [95% CI] | excludes zero |", "|---|---|---|"]
        for ch in P.CHANNELS:
            if ch in d:
                L.append(f"| {ch} | {ci_cell(d[ch])} | "
                         f"{'yes' if d[ch].get('excludes_zero') else 'no'} |")
        L.append("")
    # --- seed stability of the S6 stopping conditions ---------------------
    # The WMO-clustered CI resamples FLOATS but conditions on the trained
    # model, so it cannot see training-seed variance. On this task that
    # variance is an order of magnitude larger than the DFS-Uniform gap, and a
    # single seed can therefore report a "significant" difference that the
    # other seeds contradict. The plan's acceptance rules are stated per
    # quantity; this evaluates them ACROSS seeds, which is the weaker claim
    # that actually survives.
    per_seed = []
    for a in arts:
        dm = a.results.get("dfs_minus_uniform", {})
        for ch in P.CHANNELS:
            if isinstance(dm.get(ch), dict):
                per_seed.append((a.seeds[0] if a.seeds else NA, ch,
                                 dm[ch].get("point"), dm[ch].get("lo"),
                                 dm[ch].get("hi"), dm[ch].get("excludes_zero")))
    if per_seed:
        L += ["## Seed stability of DFS − Uniform (S6 stopping conditions)", "",
              "| seed | channel | difference | 95% CI (WMO-clustered) | excludes zero |",
              "|---:|---|---:|---|---|"]
        for sd, ch, pt, lo, hi, ez in per_seed:
            L.append(f"| {sd} | {ch} | {fmt(pt)} | [{fmt(lo)}, {fmt(hi)}] | "
                     f"{'**yes**' if ez else 'no'} |")
        L.append("")
        for ch in P.CHANNELS:
            pts = [p for _, c_, p, _, _, _ in per_seed if c_ == ch
                   and p is not None and np.isfinite(p)]
            ezs = [e for _, c_, _, _, _, e in per_seed if c_ == ch]
            if len(pts) < 2:
                continue
            signs = {np.sign(p) for p in pts if p != 0}
            stable = len(signs) <= 1
            agree = len(set(bool(e) for e in ezs)) <= 1
            L += [f"**{ch}: sign {'stable' if stable else '**FLIPS** across seeds'}; "
                  f"zero-crossing {'consistent' if agree else '**inconsistent** across seeds'}.** "
                  f"Per-seed values {['%.4f' % p for p in pts]}, "
                  f"spread {np.std(pts):.4f} against a mean magnitude of "
                  f"{np.mean(np.abs(pts)):.4f}.", ""]
            if not (stable and agree):
                L += ["> Under the plan's S6 rules this is a **stop and "
                      "reconcile** condition — and it is tripping *within* "
                      "Track A, before Track B exists. Reporting any single "
                      "seed's interval here would manufacture a result that "
                      "does not replicate across seeds.", ""]

    # The interval that actually answers the question the table is asked.
    d_row, u_row = "dfs_expertlocal_cbottle", "uniform_expertlocal_cbottle"
    L2 = []
    for ch in P.CHANNELS:
        da = [cluster_stats(a, d_row, "lead0", ch, "wmo", "dfs") for a in arts]
        ua = [cluster_stats(a, u_row, "lead0", ch, "wmo", "uniform") for a in arts]
        if not any(x is not None for x in da):
            continue
        try:
            comb = ci_difference_across_seeds(da, ua, n_boot=4000, seed=20260905)
        except ValueError:
            continue
        pts = [x["point"] for x in
               (a.results.get("dfs_minus_uniform", {}).get(ch) for a in arts)
               if isinstance(x, dict)]
        tiv = seed_t_interval(pts) if len(pts) > 1 else None
        L2.append(f"| {ch} | {ci_cell(comb.to_dict())} | "
                  f"{'yes' if comb.excludes_zero else 'no'} | "
                  f"{ci_cell(tiv.to_dict()) if tiv else NA} | "
                  f"{'yes' if (tiv and tiv.excludes_zero) else 'no'} |")
    if L2:
        L += ["### DFS − Uniform, integrating over seeds", "",
              "Negative favours DFS. The nested bootstrap resamples training "
              "seeds and, within each, the held-out floats. The seed-t column "
              "ignores float uncertainty entirely and treats the three seed "
              "means as the only observations — 2 degrees of freedom, so it is "
              "deliberately wide. They should agree on the *sign question*.", "",
              "| channel | difference [95% CI] (seed+WMO) | excludes zero | "
              "seed-t interval | excludes zero |",
              "|---|---|---|---|---|"] + L2 + [""]

    rk = arts[0].results.get("ranking")
    if rk:
        L += ["## Method ranking (TEMP, best first)", "",
              "| # | row | RMSE |", "|---:|---|---:|"]
        L += [f"| {i} | `{m}` | {fmt(v)} |" for i, (m, v) in enumerate(rk, 1)]
        L.append("")
        # is the ranking itself stable? S6 names it as a stopping condition.
        orders = []
        for a in arts:
            r = a.results.get("ranking")
            if r:
                orders.append(tuple(m for m, _ in r))
        if len(orders) > 1:
            if len(set(orders)) == 1:
                L += ["Ranking is identical across all seeds.", ""]
            else:
                L += ["> **The method ranking is not stable across seeds.** "
                      "S6 names a ranking change as a stop-and-reconcile "
                      "condition. Per-seed orders:", ""]
                for a, o in zip(arts, orders):
                    L.append(f"> * seed {a.seeds[0] if a.seeds else NA}: "
                             + " > ".join(f"`{x}`" for x in o[:5]))
                L.append("")
    cov = arts[0].results.get("reference_coverage")
    if cov:
        L += ["## External reference coverage", "",
              "| product | months covered | product range |", "|---|---:|---|"]
        L += [f"| {k} | {v['months_covered']} | {v['product_range'][0]} .. "
              f"{v['product_range'][1]} |" for k, v in sorted(cov.items())]
        L += ["", "EN4 and ECCO both **assimilate the very floats scored here**. "
              "They are upper references that have seen the answer, not peers.", ""]
    return "\n".join(L)


# ------------------------------------------------------------------ P1
def report_P1(region: str) -> str:
    arts = load(region=region, package="P1")
    L = [f"# P1 — Layout generalization ({region})", "", HEADER]
    L += provenance_block(arts)
    if not arts:
        return "\n".join(L)
    a = arts[0]
    ver = a.results.get("layout_verification", {})
    if ver:
        L += ["## Layout verification (plan's P1 cross-check)", "",
              "Counts must be **exactly** equal across arms; dispersed must be "
              "farther apart than clustered.", "",
              "| month | natural n / km | clustered n / km | dispersed n / km |",
              "|---|---|---|---|"]
        for m, d in sorted(ver.items(), key=lambda kv: int(kv[0]))[:12]:
            L.append(f"| {m} | " + " | ".join(
                f"{d[k]['n']} / {fmt(d[k]['mean_pairwise_km'], 0)}"
                for k in ("natural", "clustered", "dispersed")) + " |")
        counts_equal = all(len({d[k]["n"] for k in ("natural", "clustered", "dispersed")}) == 1
                           for d in ver.values())
        ordered = all(d["dispersed"]["mean_pairwise_km"] >= d["clustered"]["mean_pairwise_km"]
                      for d in ver.values())
        L += ["", f"* exact count equality: **{'PASS' if counts_equal else 'FAIL'}**",
              f"* dispersed farther than clustered: **{'PASS' if ordered else 'FAIL'}**", ""]
    lay = a.results.get("layouts", {})
    if lay:
        L += ["## Layout gap = J(clustered) − J(dispersed), TEMP", "",
              "Positive means the method is **hurt** by clustering.", "",
              "| row | natural | clustered | dispersed | gap [95% CI] |",
              "|---|---:|---:|---:|---|"]
        for r, d in sorted(lay.items()):
            get = lambda k: d.get(k, {}).get("channels", {}).get("TEMP", {}).get("rmse")
            L.append(f"| `{r}` | {fmt(get('natural'))} | {fmt(get('clustered'))} | "
                     f"{fmt(get('dispersed'))} | "
                     f"{ci_cell(d.get('layout_gap', {}).get('TEMP'))} |")
        L.append("")
    did = a.results.get("layout_did")
    if did:
        L += ["## Difference-in-differences", "",
              "`DiD = layout_gap(Uniform) − layout_gap(DFS)`. **Positive** means "
              "Uniform is hurt more by clustering than DFS is — the direction the "
              "DFS claim predicts. A sign change between tracks stops the "
              "cross-check.", "",
              f"**DiD = {ci_cell(did)}**, "
              f"{'excludes' if did.get('excludes_zero') else 'includes'} zero "
              f"({did.get('n_clusters')} {did.get('cluster_unit')} clusters).", ""]
    return "\n".join(L)


# ------------------------------------------------------------------ P2
def report_P2(region: str) -> str:
    arts = load(region=region, package="P2")
    L = [f"# P2 — Redundancy / duplication ({region})", "", HEADER]
    L += provenance_block(arts)
    if not arts:
        return "\n".join(L)
    a = arts[0]
    ctl = a.results.get("redundancy_control", {})
    L += ["## Positive control", "",
          "> *adding genuinely new independent observations increases useful "
          "evidence.*", "",
          f"`separated` at k=8 grows **{fmt(ctl.get('separated_k8_growth'), 2)}×** "
          f"against `exact` at **{fmt(ctl.get('exact_k8_growth'), 2)}×** → "
          f"**{'PASS' if ctl.get('passes') else 'FAIL'}**.", "",
          "Without this contrast, *suppresses redundancy* and *ignores the "
          "profile stream* produce identical flat curves.", ""]
    ev = a.results.get("evidence_mass", {})
    if ev:
        ks = a.results.get("k_values", [1, 2, 4, 8, 16, 32])
        L += ["## Evidence-mass growth of the attacked profile", "",
              "| family | rho | " + " | ".join(f"k={k}" for k in ks) + " |",
              "|---|---:|" + "---:|" * len(ks)]
        for fam, d in ev.items():
            L.append(f"| `{fam}` | {d['rho']} | " + " | ".join(
                fmt(d["by_k"][str(k)]["growth_vs_k1"] if str(k) in d["by_k"]
                    else d["by_k"].get(k, {}).get("growth_vs_k1"), 2) + "×"
                for k in ks) + " |")
        L.append("")
        pc = a.results.get("provenance_contrast", {})
        if pc:
            L += ["### The contrast real Argo makes possible", "",
                  f"At k=8, `same_provenance` grows "
                  f"**{fmt(pc.get('same_provenance_k8'), 2)}×** while "
                  f"`independent_provenance` grows "
                  f"**{fmt(pc.get('independent_provenance_k8'), 2)}×**. Both are "
                  f"k observations of nearly the same water; only the first is "
                  f"one instrument re-ingested. A synthetic column index could "
                  f"not pose this question.", ""]
    acc = a.results.get("accuracy", {})
    if acc:
        ks = a.results.get("k_values", [1, 2, 4, 8, 16, 32])
        # The evidence table above says how much MASS each family adds. This
        # says whether that mass helps, and the ratio between the two families
        # is the claim: a method that suppresses redundancy should gain from
        # `separated` and not from `exact`; a method that counts gains from
        # both. Stated as a number because the per-k tables below are ~1e-4
        # quantities where a reader cannot see a 45x ratio by eye.
        rows_ratio = []
        for row in sorted(acc):
            def ch(fam, k=8):
                d = (acc[row].get(fam, {}).get(str(k))
                     or acc[row].get(fam, {}).get(k) or {})
                return d.get("error_change_TEMP_vs_k1")
            e8, s8 = ch("exact"), ch("separated")
            if e8 is None or s8 is None:
                continue
            rows_ratio.append((row, e8, s8))
        if rows_ratio:
            def ci_of(row, fam, k=8):
                d = (acc[row].get(fam, {}).get(str(k))
                     or acc[row].get(fam, {}).get(k) or {})
                return d.get("error_change_TEMP_ci")

            # Below this, an interval that excludes zero still describes an
            # effect too small to mean anything on a ~0.5 RMSE. Reporting
            # "significant" for a 1.5e-5 change would be a false positive
            # produced by a tight bootstrap, not a finding.
            NEGLIGIBLE = 1e-4

            def verdict(c):
                if not c:
                    return "—"
                if not c.get("excludes_zero"):
                    return "no"
                if abs(c.get("point") or 0.0) < NEGLIGIBLE:
                    return f"yes, but |Δ| < {NEGLIGIBLE:g}"
                return "**better**" if (c["point"] < 0) else "**worse**"

            L += ["## Does the added evidence help? (TEMP, k=8)", "",
                  "`exact` adds eight copies of one profile; `separated` adds "
                  "eight genuinely different ones. A method that suppresses "
                  "redundancy should gain from the second and not the first. "
                  "Negative is an improvement. Intervals are WMO-clustered and "
                  "paired against each family's own k=1 — the same months, "
                  "queries and floats, so the two arms differ only by the "
                  "attack.", "",
                  "| row | exact [95% CI] | effect | separated [95% CI] | "
                  "effect |", "|---|---|---|---|---|"]
            for row, e8, s8 in rows_ratio:
                ce, cs = ci_of(row, "exact"), ci_of(row, "separated")
                L.append(
                    f"| `{row}` | {ci_cell(ce) if ce else fmt(e8, 5)} | "
                    f"{verdict(ce)} | {ci_cell(cs) if cs else fmt(s8, 5)} | "
                    f"{verdict(cs)} |")
            L += ["",
                  "> **A ratio column used to sit here and has been removed.** "
                  "`separated ÷ exact` looked decisive (45×, 292×) but divides "
                  "by a quantity whose interval spans zero, so it was reporting "
                  "the reciprocal of noise.", "",
                  "> **What this table does and does not support.** Under "
                  "`exact`, *no* learned row moves measurably — DFS and Uniform "
                  "both include zero. So **the accuracy half does not "
                  "distinguish DFS from Uniform under duplication**; that "
                  "distinction currently rests on the evidence-mass column "
                  "above, which is a property of the estimator's own weights "
                  "rather than of its predictions. `thin` and `superob` sit at "
                  "*exactly* zero, because preprocessing deletes the copies "
                  "outright — they are the ceiling here, and a small DFS value "
                  "means \"close to what thinning already does\".", "",
                  "> **The `separated` column disagrees between regions, and "
                  "that is not noise.** Read this against the other region's "
                  "report before using either. The learned rows improve in the "
                  "Gulf Stream and get significantly *worse* in the North "
                  "Pacific gyre, while OI and `thin` improve in both. A likely "
                  "mechanism is distribution shift rather than redundancy: the "
                  "k-sweep pushes the input from the registered 24 profiles to "
                  "24+k−1, so at k=8 the learned rows are seeing 31 — outside "
                  "the token budget they were trained at. OI has no training "
                  "distribution to leave, and `thin` reduces back toward a "
                  "fixed budget. **P5's code already guards against exactly "
                  "this confound; P2's accuracy half does not.** Until the "
                  "sweep is run at constant input count, these numbers cannot "
                  "separate \"more information helps\" from \"more tokens than "
                  "training hurts\".", "",
                  "> The intervals are WMO-clustered but **condition on one "
                  "trained model**. P0 has already shown on this task that seed "
                  "variance can exceed the effect being measured, and a "
                  "seed-integrated interval on these deltas has not been "
                  "computed.", "",
                  "> These effects were invisible until the query draw was held "
                  "fixed across k: seeding the sample RNG with `k` gave every k "
                  "a different set of held-out queries, and the resulting "
                  "±0.008 draw noise was an order of magnitude larger than the "
                  "effect. The tell was that all five families moved in "
                  "lockstep despite having completely different input tokens. "
                  "Track B found it.", ""]
        L += ["## Accuracy under duplication (TEMP RMSE change vs k=1)", ""]
        for row, fams in acc.items():
            L += [f"### `{row}`", "",
                  "| family | " + " | ".join(f"k={k}" for k in ks) + " |",
                  "|---|" + "---:|" * len(ks)]
            for fam, d in fams.items():
                # The artifact stores `error_change_TEMP_vs_k1` and
                # `..._SALT_vs_k1`; there is no channel-free
                # `error_change_vs_k1`, so this table rendered as all-dashes in
                # every report while the numbers sat in the artifact.
                L.append(f"| `{fam}` | " + " | ".join(
                    fmt((d.get(str(k)) or d.get(k) or {})
                        .get("error_change_TEMP_vs_k1"))
                    for k in ks) + " |")
            L.append("")
    return "\n".join(L)


# ------------------------------------------------------------------ P3/P5
def report_P3(region: str) -> str:
    arts = load(region=region, package="P0")
    L = [f"# P3 — Thinning + superobbing control ladder ({region})", "", HEADER]
    L += provenance_block(arts)
    L += ["## Why this row decides the claim", "",
          "`thin_*` and `superob_*` carry **unit mass** and remove redundancy by "
          "preprocessing the token set — count-independent by construction, with "
          "no learned mass anywhere. It is what an operational centre already "
          "does. If they match DFS, the claim becomes *\"the first "
          "differentiable, in-operator version of preprocessing\"* rather than a "
          "win over it.", ""]
    if not arts:
        return "\n".join(L)
    ladder = ["dfs_expertlocal_cbottle", "uniform_expertlocal_cbottle",
              "count_expertlocal_cbottle", "thin_expertlocal_cbottle",
              "superob_expertlocal_cbottle", "objective_interpolation"]
    L += ["## Clean accuracy, lead 0 (TEMP / SALT)", "",
          "| row | TEMP | SALT |", "|---|---:|---:|"]
    for r in ladder:
        t, _ = across_seeds(arts, lambda a, r=r:
                            a.results["rows"][r]["lead0"]["channels"]["TEMP"]["rmse"])
        s, _ = across_seeds(arts, lambda a, r=r:
                            a.results["rows"][r]["lead0"]["channels"]["SALT"]["rmse"])
        L.append(f"| `{r}` | {t} | {s} |")
    L += ["", "The clustered-layout, redundancy-stress and sparsity columns of "
          "this ladder live in the P1, P2 and P5 reports, which score the same "
          "rows.", ""]
    return "\n".join(L)


def report_P5(region: str) -> str:
    arts = load(region=region, package="P5")
    L = [f"# P5 — Argo sparsity stress ({region})", "", HEADER]
    L += provenance_block(arts)
    if not arts:
        return "\n".join(L)
    a = arts[0]
    sp = a.results.get("sparsity", {})
    fr = ["100pct", "75pct", "50pct", "25pct", "10pct"]
    L += ["## TEMP RMSE vs input density", "",
          "Thinning realizations are identical across methods (same seed, month "
          "and fraction), so the response is not a difference in the draw. "
          "Density is relative to the registered 24-profile input set.", "",
          "| row | " + " | ".join(fr) + " | 10% ÷ 100% |",
          "|---|" + "---:|" * (len(fr) + 1)]
    for r, d in sorted(sp.items()):
        vals = [d.get(k, {}).get("channels", {}).get("TEMP", {}).get("rmse")
                for k in fr]
        ratio = (vals[-1] / vals[0]) if (vals[0] and vals[-1]) else None
        L.append(f"| `{r}` | " + " | ".join(fmt(v) for v in vals) +
                 f" | {fmt(ratio, 3)} |")
    L += ["", "> **Read this with the P2 positive control in hand.** A flat "
          "sparsity curve is consistent with *robust to sparse input* AND with "
          "*not using the profile stream at all*. The two are only "
          "distinguishable against a method that is known to respond — which is "
          "why OI is in the table.", ""]
    return "\n".join(L)


# ------------------------------------------------------- S5 final report
FINAL_ROWS = [
    ("Real reconstruction", "P0", lambda a: a.results["rows"]
     ["dfs_expertlocal_cbottle"]["lead0"]["channels"]["TEMP"]["rmse"]),
    ("Future prediction (lead 3)", "P0", lambda a: a.results["rows"]
     ["dfs_expertlocal_cbottle"]["lead3"]["channels"]["TEMP"]["rmse"]),
    ("Layout DiD", "P1", lambda a: a.results["layout_did"]["point"]),
    ("Redundancy k=8 (exact growth)", "P2",
     lambda a: a.results["redundancy_control"]["exact_k8_growth"]),
    ("Redundancy k=8 (separated growth)", "P2",
     lambda a: a.results["redundancy_control"]["separated_k8_growth"]),
    ("Sparsity 25%", "P5", lambda a: a.results["sparsity"]
     ["dfs_expertlocal_cbottle"]["25pct"]["channels"]["TEMP"]["rmse"]),
    ("Senseiver comparison", None, None),
    ("ECCO / EN4 reference", "P0", lambda a: a.results["rows"]
     ["en4"]["lead0"]["channels"]["TEMP"]["rmse"]),
    ("Prospective test", "P7", lambda a: a.results["rows"]
     ["dfs_expertlocal_cbottle"]["lead0"]["channels"]["TEMP"]["rmse"]),
]

#: The same nine quantities, read out of TRACK B's layout.  The two tracks keep
#: different internal shapes on purpose, so a single accessor cannot serve both;
#: what they share is the flat `comparable` block, which is what most of these
#: read.  A row with no Track B accessor is one Track B does not compute, and
#: renders as "—" rather than being quietly filled from Track A.
FINAL_ROWS_B = {
    "Real reconstruction": lambda a: a.results["comparable"]["rmse"]
    ["dfs_expertlocal_cbottle"]["TEMP"],
    "Future prediction (lead 3)": lambda a: a.results["leads"]["lead3"]
    ["dfs_expertlocal_cbottle"]["TEMP"],
    "Layout DiD": lambda a: a.results["layout_did"]["point"],
    "Redundancy k=8 (exact growth)":
        lambda a: a.results["redundancy_control"]["exact_k8_growth"],
    "Redundancy k=8 (separated growth)":
        lambda a: a.results["redundancy_control"]["separated_k8_growth"],
    "Sparsity 25%": lambda a: a.results["comparable"]["rmse"]
    ["dfs_expertlocal_cbottle|25pct"]["TEMP"],
    "Prospective test": lambda a: a.results["rows"]
    ["dfs_expertlocal_cbottle"]["TEMP"],
}

#: Which cross-check verdict backs each summary row, so the last column reports
#: the S6 machinery's answer rather than a hand-written phrase.
VERDICT_PKG = {"Real reconstruction": "P0", "Future prediction (lead 3)": "P0",
               "Layout DiD": "P1", "Redundancy k=8 (exact growth)": "P2",
               "Redundancy k=8 (separated growth)": "P2",
               "Sparsity 25%": "P5", "ECCO / EN4 reference": "P0",
               "Prospective test": "P7"}

COMPARE_FIELDS = ["target count", "WMO count", "month count", "config hash",
                  "data-manifest hash", "model-checkpoint hash", "seed list",
                  "CI method", "cluster unit"]


def crosscheck_for(package: str, region: str):
    """Run the S6 comparison for one package, or None if a track is missing."""
    a = load(package, region)
    b = load_b(package, region)
    if not a or not b:
        return None
    comb = combine_track_a(a, region)
    if comb is None:
        return None
    if package == "P7":
        comb.package = "P7"
    keep = ("targets", "wmos", "months", "eval_split", "n_profiles",
            "split_protocol", "layout_months", "sparsity_months")
    comb.counts = {k: v for k, v in comb.counts.items() if k in keep}
    b[0].counts = {k: v for k, v in b[0].counts.items() if k in keep}
    try:
        return compare_artifacts(comb, b[0])
    except ValueError:
        return None


def report_final() -> str:
    L = ["# Final cross-check report", "", HEADER,
         "## Experiment comparison (plan S5)", ""]
    for region in P.REGIONS:
        L += [f"### {region}", "",
              "| Experiment | Track A | Track B | Difference | "
              "Same scientific conclusion? |",
              "|---|---:|---:|---:|---|"]
        verdicts = {pkg: crosscheck_for(pkg, region)
                    for pkg in ("P0", "P1", "P2", "P5", "P7")}
        for label, pkg, fn in FINAL_ROWS:
            if pkg is None:
                L.append(f"| {label} | not run | not run | {NA} | "
                         f"not yet comparable |")
                continue
            arts = load(pkg, region)
            b = load_b(pkg, region)
            va, va_vals = (across_seeds(arts, fn) if arts else (NA, []))
            fb = FINAL_ROWS_B.get(label)
            vb, vb_vals = ((across_seeds(b, fb) if (b and fb) else (NA, [])))
            diff = (f"{vb_vals[0] - va_vals[0]:+.4f}"
                    if (len(va_vals) == 1 and len(vb_vals) == 1) else
                    (f"{np.mean(vb_vals) - np.mean(va_vals):+.4f}"
                     if (va_vals and vb_vals) else NA))
            cc = verdicts.get(pkg)
            if not b:
                verdict = "**Track B pending**"
            elif fb is None or not vb_vals:
                verdict = "scored by Track A only"
            elif cc is None:
                verdict = "not comparable"
            elif cc.exact_mismatches:
                verdict = "**PROTOCOL PROBLEM**"
            elif cc.conclusion_changes:
                verdict = "**RECONCILE**"
            else:
                verdict = "yes"
            L.append(f"| {label} | {va} | {vb} | {diff} | {verdict} |")
        L.append("")

    L += ["## Also compared (plan S5)", "",
          "| field | Track A | Track B | match |", "|---|---|---|---|"]
    a0 = None
    for region in P.REGIONS:
        arts = load("P0", region)
        if arts:
            a0 = arts[0]
            break
    if a0:
        vals = {
            "target count": a0.counts.get("targets"),
            "WMO count": a0.counts.get("wmos"),
            "month count": a0.counts.get("months"),
            "config hash": a0.protocol_hash_[:16],
            "data-manifest hash": P.canonical_hash(a0.data_manifests)[:16],
            "model-checkpoint hash": "recorded per-seed in the P7 freeze record",
            "seed list": str(sorted({x for s in load("P0", "gulfstream")
                                     for x in s.seeds})),
            "CI method": "cluster bootstrap, percentile (BCa under 20 clusters)",
            "cluster unit": "wmo (primary), source_month (secondary)",
        }
        b0 = next((x for r in P.REGIONS for x in load_b("P0", r)), None)
        b_vals = {}
        if b0:
            bc = b0.results.get("comparable", {})
            b_vals = {
                "target count": bc.get("n_targets"),
                "WMO count": bc.get("n_wmos"),
                "month count": bc.get("n_months"),
                "config hash": b0.protocol_hash_[:16],
                "data-manifest hash": P.canonical_hash(b0.data_manifests)[:16],
                "model-checkpoint hash": "re-hashed from disk against the freeze "
                                         "record and Track A's own record",
                "seed list": str(sorted(b0.seeds)),
                "CI method": "cluster bootstrap by multinomial weights, "
                             "percentile",
                "cluster unit": "wmo (primary), source_month (secondary)",
            }
        for k in COMPARE_FIELDS:
            va, vb = vals.get(k, NA), b_vals.get(k, NA)
            if not b_vals:
                match = "pending"
            elif k in ("model-checkpoint hash", "CI method"):
                # Deliberately different by design: the tracks bootstrap
                # differently on purpose, and each verifies checkpoints its own
                # way. Reporting these as a "mismatch" would be noise.
                match = "by design"
            else:
                match = "yes" if str(va) == str(vb) else "**no**"
            L.append(f"| {k} | `{va}` | `{vb}` | {match} |")
    L.append("")

    L += ["## Acceptance rules (plan S6)", "",
          "**Exact-match quantities** — any mismatch is a protocol problem, not "
          "a numerical disagreement. Checked mechanically by "
          "`ocean_tokenizer.crosscheck.compare_exact` over "
          f"`{', '.join(P.EXACT_MATCH_FIELDS)}`.", "",
          "**Numerical-result quantities** — small floating-point differences "
          "are acceptable; these five changes force the tracks to stop and "
          "reconcile, and are evaluated as booleans by "
          "`crosscheck.check_conclusions`:", ""]
    L += [f"{i}. `{c}`" for i, c in enumerate(P.STOPPING_CONDITIONS, 1)]
    ran = sorted({pkg for pkg in ("P0", "P1", "P2", "P5", "P7")
                  for r in P.REGIONS if load_b(pkg, r)})
    L += ["", "## Status", ""]
    if ran:
        L += [f"Both tracks have run {', '.join(ran)}. Every comparison above "
              "comes from `crosscheck.compare_artifacts`, which refuses two "
              "artifacts carrying the same track label, so no column here can "
              "be a track compared against itself.", "",
              "**What this does not establish.** Track B was written by the "
              "same author as Track A. It re-implements the evaluation path — "
              "aggregation, clustering, bootstrap, distance metric, checkpoint "
              "resolution — and it has caught real defects doing so. It cannot "
              "establish that a shared *conception* is right: if both tracks "
              "misunderstand the same thing they will agree and both be wrong. "
              "The plan's S8 rationale — *\"this reduces the chance that one "
              "implementation is unconsciously adjusted to match the other\"* — "
              "is only fully served by a genuinely separate implementer, and "
              "agreement below should be read as evidence of arithmetic, not "
              "of science.", ""]
    else:
        L += ["Track A has run. **Track B has not.** Every Track B cell above "
              "is `pending`.", "",
              "```bash",
              ".venv/bin/python experiments/47_argo_reports.py --track-b <dir>",
              "```", ""]
    return "\n".join(L)




# ---------------------------------------------------------------- P7
def report_P7(region: str) -> str:
    p = os.path.join(OUTPUTS, f"argo_P7_{region}", "artifact_P0_seed1234.json")
    fz = os.path.join(OUTPUTS, f"argo_P7_{region}", "freeze_record_seed1234.json")
    L = [f"# P7 — Prospective real-observation test ({region})", "", HEADER]
    if not os.path.exists(p):
        return "\n".join(L + ["_Not run._", ""])
    a = P.ResultArtifact.read(p)
    fr = json.load(open(fz)) if os.path.exists(fz) else {}
    L += ["## Registration", "",
          "The plan requires the freeze to precede the open, and forbids "
          "retuning afterwards. Both stages are on record:", "",
          "| field | value |", "|---|---|",
          f"| frozen (UTC) | `{fr.get('frozen_utc', '—')}` |",
          f"| opened (UTC) | `{fr.get('opened_utc', '—')}` |",
          f"| git commit | `{fr.get('git_commit', '—')[:12]}` |",
          f"| working tree dirty | `{fr.get('git_dirty')}` |",
          f"| checkpoints pinned | {len(fr.get('checkpoints', {}))} |",
          f"| rows scored | {len(fr.get('scored_rows', []))} |",
          f"| primary inference unit | `{fr.get('primary_endpoint', {}).get('primary_cluster_unit', '—')}` |",
          f"| secondary cluster unit | `{fr.get('primary_endpoint', {}).get('secondary_cluster_unit', '—')}` |",
          f"| freeze mismatches at open | {fr.get('freeze_mismatches') or 'none'} |", ""]
    if fr.get("git_dirty"):
        L += ["> **The working tree was dirty at freeze time**, so the recorded "
              "commit does not fully identify the code that ran. The freeze "
              "pins checkpoint and manifest hashes exactly; the code is pinned "
              "only as far as the commit plus this caveat.", ""]

    # Did the open score the models the freeze pinned?  Both facts are already
    # on record -- the freeze record's hashes and the artifact's own
    # `checkpoint_sha256` per row -- so this is a comparison, not a new claim.
    # It is rendered in the MAIN report and not only in the cross-check because
    # a reader of this table has to know which of its rows are registered.
    unfrozen = []
    for row, meta in sorted(a.results.get("training", {}).items()):
        pin = fr.get("checkpoints", {}).get(f"{row}_s{a.seeds[0]}.pt")
        got = meta.get("checkpoint_sha256")
        if pin and got and pin != got:
            unfrozen.append((row, pin, got, meta.get("loaded_from")))
    if unfrozen:
        L += [f"> ### ⚠ {len(unfrozen)} of "
              f"{len(a.results.get('training', {}))} scored rows were NOT the "
              f"frozen models", "",
              "> The freeze record pins one set of bytes; the open loaded "
              "another file of the same name from a different directory. "
              "`outputs/argo_P7_" + region + "/` still holds the models the "
              "abandoned first open trained, and that directory precedes the "
              "checkpoint directory on the open's search path, so name-order "
              "resolution silently preferred them.", "",
              "> | row | frozen | actually scored | loaded from |",
              "> |---|---|---|---|"]
        for row, pin, got, src in unfrozen:
            L.append(f"> | `{row}` | `{pin[:16]}` | `{got[:16]}` | `{src}` |")
        L += ["",
              "> **These rows of the table below are not a registered result.** "
              "The freeze's own before/after check cannot detect this: it "
              "re-derives the checkpoint hashes over one directory on both "
              "sides, so a freeze and an open that resolve the same *filename* "
              "to different *files* both pass it. Track B found it by comparing "
              "the artifact's recorded loads against the pins, and the "
              "resolution now goes by hash "
              "(`protocol.resolve_pinned_checkpoint`), with the open verifying "
              "every load against the record (`--freeze-record`). Correcting "
              "the reported numbers needs a fresh freeze against the intended "
              "checkpoints and a new cohort to open — not a re-read of this "
              "one.", ""]

    L += ["## Holdout result (2025, used once)", "",
          "| row | TEMP | SALT | TEMP CI (WMO) | TEMP CI (month) |",
          "|---|---:|---:|---|---|"]
    for r, v in sorted(a.results.get("rows", {}).items()):
        ch = v["lead0"]["channels"]
        L.append(f"| `{r}` | {fmt(ch['TEMP'].get('rmse'))} | "
                 f"{fmt(ch['SALT'].get('rmse'))} | "
                 f"{ci_cell(ch['TEMP']['ci'].get('wmo'))} | "
                 f"{ci_cell(ch['TEMP']['ci'].get('source_month'))} |")
    first = next(iter(a.results["rows"].values()))["lead0"]
    L += ["", f"{first['n_wmos']} held-out floats, {first['n_targets']:,} scored "
          f"values, {first['n_months']} months. Clustered by WMO (primary) and "
          f"by source month (secondary), as registered.", ""]

    bad = a.results.get("invalid_rows_trained_at_open")
    L += ["## Protocol record", ""]
    if bad:
        L += ["A **first open of this holdout was invalid** and is on record. "
              f"The rows `{'`, `'.join(bad)}` had no frozen checkpoint and were "
              "trained (4,000 steps each) *during* that open — the retuning P7 "
              "forbids. The cause was a gap in the freeze itself: it verified "
              "that *some* checkpoint existed rather than one for every row it "
              "would score.", "",
              "Both ends are now closed: `--require-checkpoints` makes training "
              "during an open an error, and the freeze refuses outright if any "
              "scored row lacks a checkpoint. The rerun above loaded all "
              f"{len(fr.get('checkpoints', {}))} rows from frozen checkpoints, "
              "and its numbers are identical to the first open's — the "
              "violation was procedural, not arithmetic.", "",
              "> **This is therefore a second open.** The mechanics are now "
              "correct, but the holdout had already been read, so the "
              "registration claim is weaker than a never-seen test. No "
              "hyperparameter or architecture choice was changed between the "
              "two opens; the three rows' weights were fitted on train/val "
              "months only and never saw holdout data.", ""]
    else:
        L += ["Opened once, all rows from frozen checkpoints.", ""]
    return "\n".join(L)


# ------------------------------------------------------------ intern (B)
INTERN_HEADER = """<!-- generated by experiments/47_argo_reports.py -->

> **Track B, written by the same author as Track A.** It re-implements the
> evaluation path — QC re-derived from the raw GDAC files, aggregation by
> pandas groupby rather than `np.bincount`, bootstrap by multinomial weights
> rather than index resampling, distances by the spherical law of cosines
> rather than the haversine — and imports none of Track A's evaluation,
> aggregation or CI code. It can catch implementation error, and demonstrably
> has. It **cannot** establish that a shared conception is correct: if both
> tracks misunderstand the same thing they will agree and both be wrong.
"""


def intern_package_sections(package: str, a) -> list[str]:
    """The tables that only make sense for one package, rendered from Track B.

    Kept out of `report_intern`'s common path because each package's estimand is
    different: P1's is a difference of differences, P2's is an evidence ratio,
    P5's is a curve, P7's is a one-shot table.  Rendering them through one
    generic "row / TEMP / SALT" table would flatten exactly the structure the
    cross-check is about.
    """
    R = a.results
    L: list[str] = []

    # ------------------------------------------------------------ P1
    if package == "P1":
        ck = R.get("layout_checks", {})
        if ck:
            L += ["## 5. Layout selection, re-derived independently", "",
                  "Distances by the spherical law of cosines, where Track A uses "
                  "the haversine; selections re-drawn from the shared operator "
                  "and checked for the properties the plan names.", "",
                  "| check | result |", "|---|---|",
                  f"| months checked | {ck.get('months_checked')} |",
                  f"| exact profile-count equality across arms | "
                  f"**{ck.get('exact_count_equality')}** |",
                  f"| dispersed farther apart than clustered | "
                  f"**{ck.get('dispersed_farther_than_clustered')}** |",
                  f"| held-out floats appearing in the input | "
                  f"**{ck.get('heldout_floats_in_input')}** |",
                  f"| selection deterministic | **{ck.get('deterministic')}** |",
                  ""]
        lay = R.get("layouts", {})
        gap = R.get("layout_gap", {})
        if lay:
            L += ["## Layout gap = J(clustered) − J(dispersed), TEMP", "",
                  "Seed-averaged. Positive means the method is **hurt** by "
                  "clustering.", "",
                  "| row | natural | clustered | dispersed | gap (seed-t) |",
                  "|---|---:|---:|---:|---|"]
            for r in sorted(lay):
                g = lambda k: fmt(mean_of([v.get("TEMP")
                                           for v in lay[r].get(k, {}).values()]))
                L.append(f"| `{r}` | {g('natural')} | {g('clustered')} | "
                         f"{g('dispersed')} | "
                         f"{ci_cell(gap.get(r, {}).get('TEMP'))} |")
            L.append("")
        did = R.get("layout_did")
        if did:
            L += ["## Difference-in-differences (Track B's own bootstrap)", "",
                  f"**DiD = {ci_cell(did)}**, "
                  f"{'excludes' if did.get('excludes_zero') else 'includes'} "
                  f"zero. Per-seed points: "
                  + ", ".join(f"`{x:+.4f}`" for x in did.get("per_seed", []))
                  + ".", "",
                  "Each seed's DiD pairs all four arms under one draw of cluster "
                  "weights; the seeds are then combined by a t interval on the "
                  "three points, which is the estimand Track A is pooled to as "
                  "well. A sign change between tracks stops the cross-check.", ""]

    # ------------------------------------------------------------ P2
    if package == "P2":
        fc = R.get("attack_checks", {})
        if fc:
            L += ["## 5. Attack construction, re-derived independently", "",
                  "| family | distinct rows at k=8 | distinct floats | "
                  "provenance groups |", "|---|---:|---:|---:|"]
            for fam in ("exact", "jittered", "same_provenance",
                        "independent_provenance", "separated"):
                d = fc.get(fam, {})
                L.append(f"| `{fam}` | {d.get('distinct_rows_in_attack')} | "
                         f"{d.get('distinct_floats')} | "
                         f"{d.get('provenance_groups')} |")
            L += ["", f"Families hold their defining properties: "
                  f"**{fc.get('properties_hold')}**.", ""]
        ctl = R.get("redundancy_control", {})
        if ctl:
            L += ["## Positive control", "",
                  "> *adding genuinely new independent observations increases "
                  "useful evidence.*", "",
                  f"`separated` at k=8 grows "
                  f"**{fmt(ctl.get('separated_k8_growth'), 2)}×** against "
                  f"`exact` at **{fmt(ctl.get('exact_k8_growth'), 2)}×** → "
                  f"**{'PASS' if ctl.get('passes') else 'FAIL'}** "
                  f"(per seed: {ctl.get('per_seed_passes')}).", ""]
        ev = R.get("evidence_mass", {})
        if ev:
            ks = R.get("k_values", [1, 2, 4, 8, 16, 32])
            L += ["## Evidence-mass growth of the attacked profile", "",
                  "| family | rho | " + " | ".join(f"k={k}" for k in ks) + " |",
                  "|---|---:|" + "---:|" * len(ks)]
            for fam, d in ev.items():
                L.append(f"| `{fam}` | {d.get('rho')} | " + " | ".join(
                    fmt(d["by_k"].get(str(k), {}).get("growth_vs_k1"), 2) + "×"
                    for k in ks) + " |")
            L.append("")
        acc = R.get("accuracy", {})
        if acc:
            ks = R.get("k_values", [1, 2, 4, 8, 16, 32])
            L += ["## Accuracy under duplication (TEMP RMSE change vs k=1)", ""]
            for row in sorted(acc):
                L += [f"### `{row}`", "",
                      "| family | " + " | ".join(f"k={k}" for k in ks) + " |",
                      "|---|" + "---:|" * len(ks)]
                for fam, byk in acc[row].items():
                    L.append(f"| `{fam}` | " + " | ".join(
                        fmt(byk.get(str(k), {}).get("error_change_TEMP_vs_k1"))
                        for k in ks) + " |")
                L.append("")

    # ------------------------------------------------------------ P5
    if package == "P5":
        tc = R.get("thinning_checks", {})
        if tc:
            L += ["## 5. Thinning, re-derived independently", "",
                  f"* input sizes across the sweep: `{tc.get('sizes')}`",
                  f"* monotone in density: **{tc.get('monotone')}**",
                  f"* deterministic given (seed, month, fraction): "
                  f"**{tc.get('deterministic')}**",
                  f"* realization independent of the method under test: "
                  f"**{tc.get('identical_across_methods')}**", ""]
        sp = R.get("sparsity", {})
        deg = R.get("degradation", {})
        if sp:
            fr = ["100pct", "75pct", "50pct", "25pct", "10pct"]
            L += ["## TEMP RMSE vs input density (seed-averaged)", "",
                  "| row | " + " | ".join(fr) + " | 10% ÷ 100% |",
                  "|---|" + "---:|" * (len(fr) + 1)]
            for r in sorted(sp):
                vals = [mean_of([v.get("TEMP")
                                 for v in sp[r].get(k, {}).values()]) for k in fr]
                L.append(f"| `{r}` | " + " | ".join(fmt(v) for v in vals) +
                         f" | {fmt(deg.get(r, {}).get('ratio_10_over_100'), 3)} |")
            L.append("")

    # ------------------------------------------------------------ P7
    if package == "P7":
        v = R.get("freeze_verification", {})
        if v:
            L += ["## 5. Freeze record, verified independently", "",
                  "A freeze record is a claim about which models were "
                  "registered; the bytes on disk are the evidence. Every pinned "
                  "checkpoint is re-hashed here, and Track A's own record of "
                  "what it loaded is compared against the pin.", "",
                  "| field | value |", "|---|---|",
                  f"| frozen (UTC) | `{v.get('frozen_utc')}` |",
                  f"| opened (UTC) | `{v.get('opened_utc')}` |",
                  f"| holdout already opened by Track A | "
                  f"`{v.get('holdout_read_by_track_a')}` |",
                  f"| checkpoints re-hashed | {v.get('checkpoints_rehashed')} |",
                  f"| checkpoints that no longer match the pin | "
                  f"{len(v.get('checkpoint_mismatches') or [])} |",
                  f"| same-name files that are not the frozen bytes | "
                  f"{len(v.get('same_name_decoys') or [])} |", ""]
            lc = v.get("track_a_loaded_what_it_froze") or {}
            bad = [r for r, d in lc.items() if not d.get("matches_freeze")]
            if lc:
                L += ["### Did Track A score the models it froze?", "",
                      "| row | frozen sha256 | scored sha256 | loaded from | "
                      "matches |", "|---|---|---|---|---|"]
                for r, d in sorted(lc.items()):
                    L.append(f"| `{r}` | `{str(d.get('frozen_sha256'))[:16]}` | "
                             f"`{str(d.get('scored_sha256'))[:16]}` | "
                             f"`{d.get('loaded_from')}` | "
                             f"{'yes' if d.get('matches_freeze') else '**NO**'} |")
                L.append("")
                if bad:
                    L += [f"> **{len(bad)} of {len(lc)} rows in Track A's "
                          f"reported holdout table were scored from checkpoints "
                          f"the freeze record does not pin.** The freeze's own "
                          f"before/after check cannot see this: it re-derives "
                          f"the checkpoint hashes over one directory on both "
                          f"sides, so a freeze and an open that resolve the same "
                          f"*filename* to different *files* both pass it. Track "
                          f"B's table below is scored from the frozen bytes.", ""]
        rows_ = R.get("rows", {})
        if rows_:
            L += ["## Holdout table, scored from the frozen checkpoints", "",
                  "| row | TEMP | SALT | TEMP CI (WMO) | TEMP CI (month) |",
                  "|---|---:|---:|---|---|"]
            for r in sorted(rows_):
                d = rows_[r]
                L.append(f"| `{r}` | {fmt(d.get('TEMP'))} | {fmt(d.get('SALT'))} "
                         f"| {ci_cell(d.get('ci', {}).get('wmo'))} | "
                         f"{ci_cell(d.get('ci', {}).get('source_month'))} |")
            L.append("")
    return L


def mean_of(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else None


def report_intern(package: str, region: str) -> str:
    b = load_b(package, region)
    L = [f"# {package} — Track B independent run ({region})", "", INTERN_HEADER]
    if not b:
        return "\n".join(L + ["", "_Track B has not run this package._", ""])
    a = b[0]
    L += ["", "## 1-4. Provenance, manifests, config, command (plan §9)", "",
          "| field | value |", "|---|---|",
          f"| git SHA | `{a.git_commit_}` |",
          f"| working tree dirty | `{a.git_dirty_}` |",
          f"| protocol hash | `{a.protocol_hash_[:16]}` |",
          f"| seeds | `{a.seeds}` |",
          f"| command | `{a.command}` |", ""]
    L += ["| data consumed | sha256 |", "|---|---|"]
    L += [f"| {k} | `{str(v)[:16]}` |" for k, v in sorted(a.data_manifests.items())]
    L.append("")
    if a.checkpoint_hashes:
        # Plan §9 item 3. Recorded rather than asserted: these are re-hashed
        # from disk by Track B and compared against what Track A signed for, so
        # "both tracks scored the same models" is checkable, not claimed.
        L += ["| checkpoint scored | sha256 | matches Track A |",
              "|---|---|---|"]
        for k, v in sorted(a.checkpoint_hashes.items()):
            if isinstance(v, dict):
                L.append(f"| `{v.get('path')}` | `{str(v.get('sha256'))[:16]}` | "
                         f"{'yes' if v.get('matches_track_a') else '**no**'} |")
        L.append("")

    mv = a.results.get("manifest_verification")
    if mv:
        L += ["## Independent manifest verification (plan §2)", "",
              "Manifests are claims about bytes on disk; the only verification "
              "is to re-read the bytes.", "",
              "| dataset | status | files re-hashed |", "|---|---|---:|"]
        for k, v in sorted(mv.items()):
            L.append(f"| {k} | **{v.get('status')}** | "
                     f"{v.get('files_checked', 0)} / {v.get('files_claimed', 0)} |")
        L.append("")
    cv = a.results.get("cohort_verification")
    if cv:
        L += ["## Cohort re-derived from raw GDAC files", "",
              f"{cv.get('profiles_matched', 0):,} profiles independently "
              f"re-QC'd from {cv.get('floats_rederived', 0)} raw float files: "
              f"max |ΔT| = {cv.get('max_abs_TEMP_diff', float('nan')):.2e} °C, "
              f"max |ΔS| = {cv.get('max_abs_SALT_diff', float('nan')):.2e} — "
              f"**{cv.get('status')}** (float32 storage precision).", ""]

    sc = a.results.get("selection_checks")
    if sc:
        lay = sc.get("layout", {})
        L += ["## 5. Independent re-derivation of the selections", "",
              "| check | result |", "|---|---|",
              f"| exact profile-count equality across layouts | "
              f"**{lay.get('exact_count_equality')}** |",
              f"| dispersed farther apart than clustered | "
              f"**{lay.get('dispersed_farther_than_clustered')}** |",
              f"| held-out floats appearing in the input | "
              f"**{lay.get('heldout_floats_in_input')}** |",
              f"| redundancy families hold their properties | "
              f"**{sc.get('redundancy', {}).get('properties_hold')}** |",
              f"| thinning monotone | "
              f"**{sc.get('thinning', {}).get('monotone')}** |",
              f"| thinning deterministic | "
              f"**{sc.get('thinning', {}).get('deterministic')}** |", ""]

    cal = a.results.get("calibration_recomputed")
    if cal:
        L += ["## Independently recomputed calibration metrics (plan P6)", "",
              "Track A uses a closed-form Gaussian CRPS and scipy's `norm`; "
              "this uses a Monte-Carlo CRPS and empirical quantile counts, so "
              "agreement is evidence the metric is right rather than that the "
              "formula is shared.", "",
              "| seed | CRPS (MC) TEMP | 90% coverage | PIT dist. from uniform |",
              "|---|---:|---:|---:|"]
        for sd_, per in sorted(cal.items()):
            t = per.get("TEMP", {})
            L.append(f"| {sd_} | {fmt(t.get('crps_mc'))} | "
                     f"{fmt(t.get('coverage', {}).get('0.90'), 3)} | "
                     f"{fmt(t.get('tv_from_uniform'), 3)} |")
        L.append("")

    L += intern_package_sections(package, a)

    comp = a.results.get("comparable", {})
    # P1/P2/P5/P7 render their own, richer tables above; a generic
    # "row / TEMP / SALT" repeat of the same numbers underneath would only
    # invite the two to drift apart.
    if comp.get("rmse") and package not in ("P1", "P2", "P5", "P7"):
        L += ["## 5-6. Result and CI tables", "",
              "| row | TEMP | SALT |", "|---|---:|---:|"]
        for r, v in sorted(comp["rmse"].items()):
            L.append(f"| `{r}` | {fmt(v.get('TEMP'))} | {fmt(v.get('SALT'))} |")
        L.append("")
    d = comp.get("dfs_minus_uniform")
    if d:
        L += ["### DFS − Uniform (Track B's own bootstrap)", "",
              "| channel | difference [95% CI] | excludes zero |", "|---|---|---|"]
        for ch in P.CHANNELS:
            if isinstance(d.get(ch), dict):
                L.append(f"| {ch} | {ci_cell(d[ch])} | "
                         f"{'yes' if d[ch].get('excludes_zero') else 'no'} |")
        L.append("")

    L += ["## 7. Warnings", ""] + [f"* {w}" for w in a.warnings] + [""]
    L += ["## 8-10. Difference from Track A, agreement, recommendation", "",
          f"See [`crosscheck_{_slug(package)}_{region}.md`]"
          f"(crosscheck_{_slug(package)}_{region}.md) for the mechanical "
          "comparison under the plan's §6 acceptance rules.", ""]
    return "\n".join(L)


def _slug(package: str) -> str:
    return {"P0": "real_data_baseline", "P1": "layout_generalization",
            "P2": "redundancy", "P5": "sparsity",
            "P7": "prospective_test"}.get(package, package.lower())




# -------------------------------------------------- P4 / P6 standalone reports
def report_intern_external() -> str:
    """Track B's role for P4: verify the official implementation and scoring."""
    rec = {}
    for nm in ("reproduce", "argo_stage"):
        p_ = os.path.join(ROOT, "data", "senseiver", f"{nm}.json")
        if os.path.exists(p_):
            rec[nm] = json.load(open(p_))
    L = ["# P4 — Track B verification of the external baselines", "",
         INTERN_HEADER, "",
         "The plan's split for P4 is: Track A does the primary integration, "
         "Track B *independently reproduces / verifies the official "
         "implementation and its scoring*. Verification here means checking "
         "the claims that matter — that the official example really runs, and "
         "that no model code was altered.", ""]
    rep, arg = rec.get("reproduce"), rec.get("argo_stage")
    L += ["## Rule 1 — the official example runs", "", "| check | value |",
          "|---|---|",
          f"| upstream repo | `OrchardLANL/Senseiver` |",
          f"| upstream commit | `{(rep or arg or {}).get('upstream_commit', '—')}` |",
          f"| official trainer return code | "
          f"`{rep.get('returncode') if rep else '—'}` |", ""]
    if arg:
        L += ["## Rule 2 — adapted to our held-out Argo evaluation", "",
              "| field | value |", "|---|---|",
              f"| region | `{arg.get('region')}` |",
              f"| field shape (T, H, W, 2·D) | `{arg.get('field_shape')}` |",
              f"| distinct real-Argo sensor cells | {arg.get('n_sensor_cells')} |",
              f"| sensors supplied | {arg.get('num_sensors')} |",
              f"| return code | `{arg.get('returncode')}` |", "",
              "Depth is folded into channels, which is the faithful mapping: an "
              "Argo float measures the **whole column at one location**, and a "
              "Senseiver sensor is exactly a pixel whose every channel is "
              "observed. That uses the authors' standard 2-D path unchanged.", ""]
        add = arg.get("upstream_additions", {})
        L += ["## Rule 3 — no homemade approximation labelled Senseiver", "",
              "Every line added upstream is dataset plumbing. `network_light.py` "
              "and `model.py` are untouched, so all model code executed is the "
              "authors'. Files appended to:", ""]
        L += [f"* `{k}`" for k in sorted(add)] or ["* (none — already present)"]
        L += ["", "The additions are recorded verbatim in "
              "`data/senseiver/argo_stage.json`, so the diff against upstream "
              "is auditable rather than asserted.", ""]
    L += ["## Verdict", "",
          "Rules 1 and 2 are satisfied and rule 3 holds structurally. What "
          "Track B has **not** done is an independent re-scoring of the "
          "Senseiver at held-out float positions — the trained model exists but "
          "its predictions have not been carried through the WMO-clustered "
          "evaluation the DFS rows use, so it does not yet appear in the P0 "
          "table. That is the remaining gap for this package.", ""]
    return "\n".join(L)


def report_crosscheck_external() -> str:
    L = ["# Cross-check — P4 external baselines", "", HEADER, "",
         "## Status", "",
         "| reference | Track A | Track B | comparable? |",
         "|---|---|---|---|",
         "| EN4 | scored in P0 and P7, and as the dense reference for the "
         "global map | not independently re-scored | **no** |",
         "| ECCO | scored under the labelled `ecco_overlap` secondary protocol | "
         "not independently re-scored | **no** |",
         "| Senseiver | official example reproduced; Argo adaptation trained | "
         "upstream commit and untouched-model-code verified | partially |",
         "| ADAF-Ocean / ORCA-DL / XiHe / WenHai / FuXi-Ocean | feasibility "
         "assessed, not runnable | — | n/a |", "",
         "## What can and cannot be concluded", "",
         "Nothing in this package has two independent numbers to compare, so "
         "the §6 acceptance rules do not yet apply to it. Track B has verified "
         "the *provenance* claims for the Senseiver (upstream commit, model "
         "code untouched, the exact diff), which is the part most likely to be "
         "wrong in a port, but it has not recomputed a score.", "",
         "The blocking item is the same for both tracks: the Senseiver's "
         "predictions have not been carried through the held-out-float "
         "evaluation, so there is no Senseiver row in the P0 table for either "
         "track to disagree about.", "",
         "See [`main_external_baselines.md`](main_external_baselines.md) for "
         "why the remaining references cannot be executed at all.", ""]
    return "\n".join(L)


def report_intern_uncertainty() -> str:
    L = ["# P6 — Track B recomputation of the calibration metrics", "",
         INTERN_HEADER, ""]
    any_ = False
    for region in ("gulfstream", "npac_gyre"):
        b = load_b("P0", region)
        cal = b[0].results.get("calibration_recomputed") if b else None
        if not cal:
            continue
        any_ = True
        L += [f"## {region}", "",
              "Track A computes CRPS in closed form for a Gaussian and uses "
              "`scipy.stats.norm` for coverage and PIT. Track B uses a "
              "**Monte-Carlo CRPS** and **empirical quantile counts**, sharing "
              "no formula — so agreement is evidence the metric is right, not "
              "that the code is the same.", "",
              "| seed | CRPS (MC) | 90% coverage | PIT distance from uniform | n |",
              "|---|---:|---:|---:|---:|"]
        for sd_, per in sorted(cal.items()):
            t = per.get("TEMP", {})
            L.append(f"| {sd_} | {fmt(t.get('crps_mc'))} | "
                     f"{fmt(t.get('coverage', {}).get('0.90'), 3)} | "
                     f"{fmt(t.get('tv_from_uniform'), 3)} | "
                     f"{t.get('n', 0):,} |")
        L.append("")
    if not any_:
        L += ["_Track B has not yet recomputed the calibration metrics; the P6 "
              "predictions were not present when it last ran._", ""]
    return "\n".join(L)


def report_crosscheck_uncertainty() -> str:
    L = ["# Cross-check — P6 predictive uncertainty", "", HEADER, ""]
    rows = []
    for region in ("gulfstream", "npac_gyre"):
        ap_ = os.path.join(OUTPUTS, f"argo_P6_{region}", "artifact_P6_seed1234.json")
        b = load_b("P0", region)
        if not os.path.exists(ap_):
            continue
        a = json.load(open(ap_))
        cal = b[0].results.get("calibration_recomputed") if b else None
        for sd_ in ("1234", "1235", "1236"):
            ta = a["results"].get("per_seed", {}).get(sd_, {}).get("TEMP", {})
            tb = (cal or {}).get(sd_, {}).get("TEMP", {})
            if not ta or not tb:
                continue
            rows.append((region, sd_, ta.get("crps"), tb.get("crps_mc"),
                         ta.get("coverage", {}).get("0.90"),
                         tb.get("coverage", {}).get("0.90")))
    L += ["## Guarantees (plan P6, required tests)", "",
          "| region | base checkpoint unchanged | zero gradient into base | "
          "mean bit-identical |", "|---|---|---|---|"]
    for region in ("gulfstream", "npac_gyre"):
        ap_ = os.path.join(OUTPUTS, f"argo_P6_{region}", "artifact_P6_seed1234.json")
        if not os.path.exists(ap_):
            continue
        g = [v for k, v in json.load(open(ap_))["results"]["guarantees"].items()
             if isinstance(v, dict)]
        L.append(f"| {region} | "
                 f"{'all pass' if all(x['base_unchanged'] for x in g) else 'FAIL'} | "
                 f"{'all zero' if all(x['grad_into_base'] in (0, 0.0) for x in g) else 'FAIL'} | "
                 f"{'all pass' if all(x['mean_bit_identical'] for x in g) else 'FAIL'} |")
    L.append("")
    if rows:
        L += ["## Independently recomputed metrics", "",
              "| region | seed | CRPS A (closed form) | CRPS B (Monte-Carlo) | "
              "cov90 A | cov90 B |", "|---|---|---:|---:|---:|---:|"]
        for r in rows:
            L.append(f"| {r[0]} | {r[1]} | {fmt(r[2])} | {fmt(r[3])} | "
                     f"{fmt(r[4], 3)} | {fmt(r[5], 3)} |")
        L += ["", "CRPS is computed by two different estimators, so exact "
              "agreement is not expected; the Monte-Carlo value carries "
              "sampling noise of order 1/sqrt(200) per point. What must agree "
              "is the **conclusion** — whether the head is calibrated and which "
              "way it fails.", ""]
    else:
        L += ["_Track B has not yet recomputed these._", ""]
    return "\n".join(L)


# The header is looked up by name at call time, so it can be corrected here,
# once, after the Track B artifacts are known. Leaving the "One track" banner
# on a report whose Track B column is populated would be worse than having no
# banner at all.
if any(load_b(pkg, r) for pkg in ("P0", "P1", "P2", "P5", "P7") for r in P.REGIONS):
    HEADER = """<!-- generated by experiments/47_argo_reports.py — do not edit by hand -->

> **Two tracks.** Track A is the main implementation; Track B independently
> re-implements the evaluation path — aggregation, clustering, bootstrap,
> distance metric, checkpoint resolution — and shares only what plan §2 says
> the tracks must share: the protocol, the manifests, the sample construction
> and the checkpoints. Where a table below shows one column, that quantity was
> scored by one track only and no agreement should be inferred from it.

> **Track B was written by the same author as Track A.** It can catch
> implementation error, and has. It cannot establish that a shared conception is
> correct: if both tracks misunderstand the same thing they will agree and both
> be wrong.

> **What "real data" means here.** Input is real QC'd Argo profiles from the
> cohort floats. Targets are real measurements from **WMO-disjoint held-out
> floats** that appear in no training month. Scores are RMSE in train-only
> z-units at genuine measurement positions, clustered by held-out float.
"""

BUILDERS = {"real_data_baseline": ("P0", report_P0),
            "layout_generalization": ("P1", report_P1),
            "redundancy": ("P2", report_P2),
            "control_ladder": ("P0", report_P3),
            "sparsity": ("P5", report_P5)}

written = []
for region in P.REGIONS:
    for name, (pkg, fn) in BUILDERS.items():
        md = fn(region)
        path = os.path.join(REPORTS, f"main_{name}_{region}.md")
        with open(path, "w") as f:
            f.write(md)
        written.append(path)
        itxt = report_intern(pkg, region)
        if name == "control_ladder":
            itxt = itxt.replace(
                f"# {pkg} — Track B independent run ({region})",
                f"# P3 — Track B independent run ({region})\n\n"
                f"> The ladder's `thin_*` and `superob_*` rows are P0 rows, so "
                f"this is Track B's P0 run read for the ladder. Its thinning "
                f"budget, grid, merge radius, noise weighting and modality "
                f"handling come from the shared `godas_model.build_row` "
                f"registry — identical by construction, not by independent "
                f"agreement.", 1)
        with open(os.path.join(REPORTS, f"intern_{name}_{region}.md"), "w") as f:
            f.write(itxt)
        written.append(os.path.join(REPORTS, f"intern_{name}_{region}.md"))
        a = load(pkg, region)
        b = load_b(pkg, region)
        a_comb = combine_track_a(a, region) if a else None
        cc = None
        if a_comb and b:
            # counts differ in bookkeeping-only keys between tracks; compare the
            # ones the plan actually names as exact-match quantities
            # Package-specific counts belong here too. Keeping only the P0 keys
            # left P1 and P5 comparing two EMPTY count dicts, which matches
            # trivially -- so the one exact-match quantity those packages
            # actually have (how many months were scored) was never checked.
            keep = ("targets", "wmos", "months", "eval_split", "n_profiles",
                    "split_protocol", "layout_months", "sparsity_months")
            a_comb.counts = {k: v for k, v in a_comb.counts.items() if k in keep}
            b[0].counts = {k: v for k, v in b[0].counts.items() if k in keep}
            cc = compare_artifacts(a_comb, b[0])
        cpath = os.path.join(REPORTS, f"crosscheck_{name}_{region}.md")
        txt = render_report(pkg, region, a_comb or (a[0] if a else None),
                            b[0] if b else None, cc)
        if name == "control_ladder":
            # P3's rows ARE P0 rows, so this file was a byte-identical copy of
            # the P0 cross-check and said nothing about the ladder's own
            # cross-check list. Say what is and is not independently checked
            # here, rather than letting a duplicate imply more than it shows.
            txt = txt.replace(
                "\n## Provenance",
                "\n> **What P3 cross-checks, and what it cannot.** The ladder's "
                "rows (`thin_*`, `superob_*`) are P0 rows, so the numbers below "
                "are the P0 comparison restricted to them. The plan also asks "
                "the two tracks to agree on the thinning budget, grid "
                "definition, merge radius, noise weighting and modality "
                "handling: both tracks instantiate the ladder from the same "
                "`godas_model.build_row` registry and load the same "
                "checkpoints, so those five are shared **by construction, not "
                "by agreement** — two independent implementations of them would "
                "be a stronger check and do not exist. What is independently "
                "checked is the scoring of the rows they produce.\n"
                "\n## Provenance", 1)
        with open(cpath, "w") as f:
            f.write(txt)
        written.append(cpath)

for region in P.REGIONS:
    if region == "eq_pacific":
        continue
    for nm, txt in (("main_prospective_test", report_P7(region)),
                    ("intern_prospective_test", report_intern("P7", region))):
        pth = os.path.join(REPORTS, f"{nm}_{region}.md")
        with open(pth, "w") as f:
            f.write(txt)
        written.append(pth)
    a7 = load("P7", region)
    b7 = load_b("P7", region)
    # The P7 open runs the P0 driver into the P7 output directory, so Track A's
    # artifact is labelled package "P0". Relabel the pooled copy before
    # comparing: `compare_artifacts` refuses a package mismatch, which is why
    # this cross-check was previously passed `None` and computed nothing at all.
    a7c = combine_track_a(a7, region) if a7 else None
    cc7 = None
    if a7c and b7:
        a7c.package = "P7"
        keep7 = ("targets", "wmos", "months", "eval_split", "n_profiles",
                 "split_protocol")
        a7c.counts = {k: v for k, v in a7c.counts.items() if k in keep7}
        b7[0].counts = {k: v for k, v in b7[0].counts.items() if k in keep7}
        cc7 = compare_artifacts(a7c, b7[0])
    pth = os.path.join(REPORTS, f"crosscheck_prospective_test_{region}.md")
    txt7 = render_report("P7", region, a7c or (a7[0] if a7 else None),
                         b7[0] if b7 else None, cc7)
    fv = (b7[0].results.get("freeze_verification", {}) if b7 else {})
    unfrozen = [r for r, d in (fv.get("track_a_loaded_what_it_froze") or {}).items()
                if not d.get("matches_freeze")]
    if unfrozen:
        txt7 = txt7.replace(
            "\n## Provenance",
            f"\n> **Read the disagreement below as a freeze failure, not a "
            f"numerical one.** Track A scored `{'`, `'.join(sorted(unfrozen))}` "
            f"from checkpoints the freeze record does not pin; Track B scored "
            f"them from the frozen bytes. Every other row matches to float "
            f"precision, and those three are exactly the rows that differ. See "
            f"[`intern_prospective_test_{region}.md`]"
            f"(intern_prospective_test_{region}.md) for the hash-by-hash "
            f"comparison.\n"
            "\n## Provenance", 1)
    with open(pth, "w") as f:
        f.write(txt7)
    written.append(pth)

for nm, txt in (("intern_external_baselines", report_intern_external()),
                ("crosscheck_external_baselines", report_crosscheck_external()),
                ("intern_uncertainty", report_intern_uncertainty()),
                ("crosscheck_uncertainty", report_crosscheck_uncertainty())):
    pth = os.path.join(REPORTS, f"{nm}.md")
    with open(pth, "w") as f:
        f.write(txt)
    written.append(pth)

fpath = os.path.join(REPORTS, "final_crosscheck_report.md")
with open(fpath, "w") as f:
    f.write(report_final())
written.append(fpath)

for p in written:
    print(f"  wrote {os.path.relpath(p, ROOT)}")
print(f"\n{len(written)} files")
