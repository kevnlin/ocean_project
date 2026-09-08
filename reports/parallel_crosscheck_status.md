# Parallel replication & cross-check — implementation status

Track A implementation of `parallel_replication_crosscheck_plan.md`.
This document says what exists, what ran, and what is blocked and why.

---

## The change that makes this "real data"

Before this work, the project's "real data" line drew a **uniformly random
column of the GODAS reanalysis** as a profile and scored against that same
reanalysis (`godas_obs.build_sample`). That is an OSSE wearing a real dataset's
clothes, and it is circular: GODAS assimilates Argo, so predicting GODAS from
Argo-like columns is partly a test of whether a model can invert an
assimilation.

The plan's design is built on something else entirely — WMO-clustered CIs, a
held-out float cohort, same-provenance duplicates, a prospective
real-observation test. None of that had a substrate. It does now:

| | before | now |
|---|---|---|
| a "profile" | random reanalysis grid column | real QC'd Argo profile |
| platform identity | column index `0..23` | the float's **WMO** |
| observation error | pilot constant `0.08` | `TEMP/PSAL_ADJUSTED_ERROR`, per level |
| target | the same reanalysis | **held-out floats' own measurements** |
| cluster unit | none retained | WMO (primary), source-month (secondary) |

**Input**: real Argo profiles from cohort floats.
**Target**: real measurements from **WMO-disjoint held-out floats** appearing in
no training month, in any year of their life.
That is what licenses WMO as the plan's inference unit.

## Data acquired

| dataset | extent | size |
|---|---|---|
| Argo GDAC floats | 2,259 floats, 2000–2025, both regions | 9.7 GB |
| Argo QC'd cohort — Gulf Stream | 101,202 profiles, 1,154 floats, 312 months, **231 held-out floats / 19,060 profiles** | |
| Argo QC'd cohort — N. Pacific gyre | 102,756 profiles, 871 floats, 288 months, **174 held-out floats / 21,975 profiles** | |
| GODAS N. Pacific gyre (new region) | 2000–2025, identical 38×26×16 grid | 18 MB |
| EN4 (EN.4.2.2 g10) | 2000–2025, both regions | |
| ECCO V4r4 | 2000–2017, both regions | |
| NOAA weekly SST | 1981–2023, for the Senseiver reproduction | 224 MB |

Cohort quality: 86–95 % delayed-mode, 85–95 % carrying adjusted errors,
90–100 % level coverage across all 16 depths.

## Packages

| package | Track A | Track B | cross-check |
|---|---|---|---|
| **P0** real-data baseline table | 8 rows + references, 3 seeds, leads 0-3, 2 regions | run, 2 regions | **agree** |
| **P1** layout / clustering | run, 3 seeds, 2 regions. DiD with paired CI | run, 2 regions | **agree** |
| **P2** redundancy | run, 3 seeds, 2 regions; **positive control PASSES** | run, 2 regions; control passes on every seed | **agree** |
| **P3** thinning + superobbing ladder | the `thin_*` / `superob_*` rows of P0 | same rows, scored independently | **agree** |
| **P4** external baselines | EN4 **and ECCO scored**; Senseiver official example **reproduced**; rest blocked — [main_external_baselines.md](main_external_baselines.md) | provenance of the Senseiver port verified; no independent score | not comparable |
| **P5** sparsity stress | run, 3 seeds, 2 regions | run, 2 regions | **agree** |
| **P6** predictive uncertainty | module + all three required guarantees | calibration metrics recomputed by a different estimator | **agree** |
| **P7** prospective test | opened, both regions | scored independently from the frozen bytes | **RECONCILE — see below** |

The first table in this file used to say which packages were *implemented*.
That was the right question when nothing had run; it is the wrong one now, so
it says what agreed instead.

### P2 — the positive control passes, on real floats

The plan makes the whole package conditional on it: *"adding genuinely new
independent observations increases useful evidence."*

| family | k=8 evidence growth |
|---|---:|
| exact duplicates | 1.29× |
| jittered (25 km) | 1.75× |
| same-provenance | 1.29× |
| independent-provenance | 1.46× |
| **separated (control)** | **11.83×** |

Genuinely new observations grow evidence **9.2× more** than re-ingesting the
same one. Without that contrast, *"suppresses redundancy"* and *"ignores the
profile stream"* produce identical flat curves.

Real Argo also makes a distinction the synthetic version could not pose:
`same_provenance` (k cycles of **one** float) and `independent_provenance` (k
**different** floats measuring the same water) are both k observations of nearly
the same water, and only the first is one instrument re-ingested.

## Infrastructure

* `clustered_ci.py` — cluster bootstrap. Resamples **clusters** and recomputes
  the pooled ratio (not the mean of per-cluster RMSEs, which is biased when
  clusters differ in size — Argo floats differ by 20×). Methods share one draw
  of cluster indices, so difference and DiD intervals reflect the paired design.
* `protocol.py` — the frozen §2 specification as a data structure, with a
  protocol hash, manifest hashes, and the signed `ResultArtifact` of §9.
* `crosscheck.py` — §6 acceptance rules. Exact-match mismatches are errors;
  numerical differences are reported; the five stopping conditions are booleans.
* `argo_obs.py` / `argo_experiments.py` — real-observation samples and the
  P1/P2/P5 operators, all pure functions of (cohort, month, seed).
* `uncertainty.py` — post-hoc head that structurally cannot move the mean.
* `reference_adapters.py` — EN4/ECCO at real float positions.

**446 tests pass** (339 pre-existing, 107 new).

## The headline finding so far — P0, Gulf Stream, 3 seeds

| row | TEMP (mean ± sd over seeds) |
|---|---:|
| `en4` *(assimilates these floats)* | 0.429 ± 0.006 |
| `thin_expertlocal_cbottle` | 0.533 ± 0.017 |
| `dfs_expertlocal_cbottle` | 0.533 ± 0.027 |
| `uniform_expertlocal_cbottle` | 0.534 ± 0.021 |
| `count_expertlocal_cbottle` | 0.534 ± 0.024 |
| `superob_expertlocal_cbottle` | 0.535 ± 0.017 |
| `objective_interpolation` | 0.794 ± 0.014 |
| `source_persistence` | 0.802 ± 0.037 |
| `train_climatology` | 1.008 ± 0.002 |

The learned rows beat OI, persistence and climatology decisively (0.53 vs
0.79 / 0.80 / 1.01). **In the Gulf Stream every mass mode is indistinguishable
from every other** — DFS, Uniform, Count, thin and superob span 0.533–0.535,
while the seed spread is ±0.02, an order of magnitude larger.

**The second region reverses that, and this is the headline.** Running the same
experiment in the quiet North Pacific subtropical gyre gives a DFS advantage
that is consistent in sign across all three seeds and excludes zero:

| region | per-seed signs | DFS − Uniform, TEMP (seed+WMO) | excludes zero |
|---|---|---|---|
| Gulf Stream — western boundary current, high EKE | −, **+**, − | −0.0006 [−0.0078, +0.0077] | no |
| **N. Pacific gyre — quiet interior** | −, −, − | **−0.0291 [−0.0561, −0.0123]** | **yes** |

Track B reproduces both independently. The two boxes share a grid, a token
budget, a model and a protocol, and differ only in which ocean they cover — so
this is a regional effect, not a configuration difference. It is also precisely
what the plan's two-region P1 design was built to detect.

Do not over-read it yet: two regions is two samples, and the mechanism behind
the contrast (mesoscale variability swamping the weighting benefit? denser,
more clustered float coverage in the gyre where redundancy suppression pays?)
is not established by this table. The layout and redundancy packages are where
that gets tested.

**`DFS − Uniform` changes sign across seeds**, and seed 1234 alone reports a
"significant" DFS advantage that seeds 1235 and 1236 contradict:

| seed | TEMP diff | 95 % CI (WMO-clustered) | excludes zero |
|---:|---:|---|---|
| 1234 | −0.0081 | [−0.0119, −0.0048] | **yes** |
| 1235 | +0.0071 | [−0.0007, +0.0166] | no |
| 1236 | −0.0007 | [−0.0023, +0.0009] | no |

This is a plan §6 **stop-and-reconcile** condition, and it is tripping *within*
Track A before Track B exists. The cause is structural: the WMO-clustered
bootstrap resamples floats but conditions on the trained model, so it cannot see
training-seed variance — which here dominates. Reporting any single seed's
interval would manufacture a result that does not replicate.

So the honest summary is regional: **the DFS-over-Uniform claim is not
supported by the Gulf Stream P0 table, and is supported in the North Pacific
gyre.** A paper reporting only one of those boxes would be reporting a choice,
not a result.

## Track B — what an independent implementation found

Track B (`experiments/51_track_b.py`) re-implements the evaluation path: QC
re-derived from the raw GDAC files, aggregation by pandas groupby instead of
`np.bincount`, bootstrap by multinomial weights instead of index resampling,
pooled RMSE by streaming accumulation. It shares the frozen protocol, manifests,
sample construction and checkpoints, as §2 requires, and imports none of Track
A's evaluation, aggregation or CI code.

**It is written by the same author as Track A.** It can catch implementation
error; it cannot establish that a shared conception is correct. Agreement here
is evidence of arithmetic, not of science. Every artifact it writes says so.

Even so, it earned its keep — it found ten real defects. The first four came
from the P0 comparison; the last three came from extending it to P1, P2, P5 and
P7, and one of them invalidates part of a reported result:

| # | finding | resolution |
|---|---|---|
| 1 | `source_persistence` **ill-defined under ties**. Query and observation coordinates are both on a discrete grid, so exact ties for "nearest" are common (16 of 512 queries per month). Two mathematically equivalent distance formulations broke them differently: **201 of 18,432 predictions differed, RMSE 0.8075 vs 0.8046**. | Tie-breaking made explicit in the baseline's *definition* — among observations within `TIE_EPS` of the minimum, take the lowest token index. Both tracks now agree on **0/18,432**. |
| 2 | `targets` counted differently — Track A summed over channels, Track B counted TEMP only, so the two differed by **exactly 2×** and the cross-check called it a protocol problem. | A definitional mismatch, not a data one. Both now count scored target values. |
| 3 | Track B accumulated squared residuals in **float32** where Track A upcasts to float64, leaving the `DFS − Uniform` point estimate 1.5e-8 apart. | Track A was on the correct side; Track B follows it rather than the tolerance being loosened until the difference vanished. |
| 4 | Ranking compared across **different row sets** (Track B does not re-derive the shared gridded-reference adapter), and coverage was being classified as a *scientific* disagreement — the whole cross-check read `RECONCILE` when every shared number agreed. | Rankings compared on the intersection; coverage reported separately and never as a conclusion change. |
| 5 | **P7 scored three rows from models that were never frozen.** The freeze pins checkpoint hashes in `outputs/argo_P0_<region>/`; the open resolved by filename and its own output directory comes first on that search path, and that directory still holds the models the *abandoned first open* trained under the same names. In both regions the three `*_oi_expert_cbottle` rows loaded the decoys. Verified directly: the frozen `dfs_oi` checkpoint scores TEMP **0.5646**, the decoy scores **0.5637**, and 0.5637 is the number in the report. | Resolution now goes by **bytes**, not by name (`protocol.resolve_pinned_checkpoint`, 7 tests), and the open verifies every load against the record (`--freeze-record`). **The reported P7 numbers for those rows are not a registered result** and need a fresh freeze and a new cohort to correct — see below. |
| 6 | The P0 cross-check read `RECONCILE` on a **method-ranking change that was a scope difference**. Seed 1234's P0 artifact had been overwritten by a 3-row `*_oi_expert` run, so Track A's track-level seed mean for the five headline rows covered 2 seeds while Track B's covered 3. | Seed 1234 re-scored over the full row set (evaluation only, every checkpoint loaded). All four P0/P3 verdicts became `agree`. |
| 8 | **P2's accuracy half measured the query draw, not duplication.** `attacked_sample` seeded the sample RNG with `k`, and `build_argo_sample` draws the 512 held-out queries from it — so "RMSE change vs k=1" compared two unrelated draws. The ±0.008 TEMP noise that introduced was an order of magnitude larger than the effect. The tell: all five families moved in lockstep (k=2 → +0.0078, +0.0078, +0.0078, +0.0074, +0.0076) despite having completely different input tokens. | Seeded `[seed, month, lead]`, as every other evaluation in the project does. See below — the corrected table is a **result**, not just a fix. |
| 9 | **`jittered`'s coordinate offsets were not reproducible.** `duplicate_rows` seeded its RNG with `abs(hash(family))`, and Python randomizes str hashing per process, so the one redundancy family whose construction draws random numbers got a different set of offsets on every run — including from the same code on the same machine an hour later. The two tracks agreed exactly on the other four families and disagreed only on this one, which is how it surfaced. | Seeded by `REDUNDANCY_FAMILIES.index(family)`. Two regression tests, one of which runs the draw in a subprocess under three `PYTHONHASHSEED` values, because an in-process check cannot see the bug at all. |
| 10 | **P2 scores the DFS rows under a `provenance_rho` they were not trained with.** `46_argo_redundancy.py` builds them with `--rho 0.9`; P0 (which trained the checkpoints) and P1/P5/P7 all use the default `0.0`. So P2's accuracy table describes a differently configured model from the P0 table it is read against, and its k=1 baseline is not the registered model. | Both tracks now score the registered checkpoint as registered (`rho=0.0`), and `rho` is confined to the evidence half where it is applied per family as documented. P2's accuracy table and P0's baseline table describe the same model again. |
| 7 | P1, P2 and P5 were compared **seed-1234-against-three-seeds**: `combine_track_a` pooled only the P0-shaped artifacts and passed the others through as `arts[0]`. Their package-specific counts (`layout_months`, `sparsity_months`) were also being filtered out of the exact-match set, so those packages compared two empty count dicts and matched trivially. | Per-seed pooling extended to P1/P2/P5 (seed-t on the DiD, `all()` on the redundancy control); the count filter now keeps the package-specific keys. |

Two further tolerance issues the real comparison exposed: bootstrap **CI
endpoints** are Monte-Carlo estimates and cannot match to 1e-6 when two tracks
resample differently on purpose (they get their own budget; what must still
agree exactly is `excludes_zero`), and **near-zero quantities** must not be
judged by relative error alone — `DFS − Uniform` is ~6e-4, so a 1e-9 absolute
difference is a 1.5e-5 *relative* one.

**Current verdicts: 10 of 12 cross-checks `agree`.** Every shared number in
P0, P1, P2, P3 and P5 matches in both regions; the only remaining entry is
coverage (`en4`, which Track B does not re-derive). The independent QC
re-derivation matches to float32 storage precision — profiles rebuilt from raw
GDAC files, max |ΔT| = 9.5e-07 °C — and every manifest file re-hashes correctly.

The two `RECONCILE` verdicts are both P7, and both are finding #5: the three
`*_oi_expert` rows differ because Track A scored decoys and Track B scored the
frozen bytes. Every other P7 row matches to float precision.

## What finding #8 uncovered — and what it does *not* show

With the query draw held fixed across `k`, the accuracy half of P2 stops being
noise. An earlier draft of this section read the point estimates as
"DFS is flat under duplication and responds to new information, ~45x". Putting
**confidence intervals** on them (paired against each family's own k=1, WMO
clustered) does not support that, and the ratio it rested on was dividing by a
quantity indistinguishable from zero. The corrected reading:

**1. Under `exact` duplication, no learned row moves measurably — including
Uniform.** k=8, TEMP, seed 1234:

| row | Gulf Stream | N. Pacific gyre |
|---|---|---|
| `dfs_expertlocal_cbottle` | +4.5e-06 [−6.0e-05, +6.9e-05] | +9.3e-06 [−9.2e-05, +9.6e-05] |
| `uniform_expertlocal_cbottle` | −2.5e-04 [−1.2e-03, +7.0e-04] | +5.0e-04 [−6.0e-04, +1.7e-03] |
| `thin_expertlocal_cbottle` | exactly 0 | exactly 0 |

Both include zero in both regions. **The accuracy half does not distinguish DFS
from Uniform under duplication.** That distinction rests entirely on the
evidence-mass column — a property of the estimator's own weights, not of its
predictions. `thin`/`superob` are exactly zero because preprocessing deletes the
copies, so they are the ceiling on this metric and a small DFS number means
"close to what thinning already does".

**2. Under `separated`, the sign flips between regions, significantly.**

| row | Gulf Stream | N. Pacific gyre |
|---|---|---|
| `dfs_expertlocal_cbottle` | **−1.3e-03** [−2.3e-03, −3.4e-04] | **+3.1e-03** [+6.1e-04, +6.2e-03] |
| `uniform_expertlocal_cbottle` | −1.1e-03 [−1.6e-03, −4.6e-04] | +1.3e-03 [+2.0e-04, +2.8e-03] |
| `objective_interpolation` | −4.0e-02 | −2.6e-02 |
| `thin_expertlocal_cbottle` | −1.2e-03 | −1.1e-03 |

Adding eight genuinely new, well-separated profiles makes the learned rows
**significantly worse** in the gyre, while OI and `thin` improve in both regions.

**The likely mechanism is distribution shift, not redundancy.** The k-sweep
raises the input from the registered 24 profiles to 24+k−1, so at k=8 the
learned rows see 31 — outside the token budget they were trained at. OI has no
training distribution to leave; `thin` reduces back toward a fixed budget. This
is the same confound `45_argo_real_data.py` already guards against in P5 (*"a
month holds ~300 cohort profiles but the rows were trained on n_profiles = 24;
thinning the census made 100% a 12x token increase over training, so the sweep
measured distribution shift rather than sparsity"*). **P2's accuracy half has no
such guard.**

Until the k-sweep is run at *constant input count* — displacing profiles rather
than appending them — the accuracy half of P2 cannot separate "more information
helps" from "more tokens than training hurts", and the `separated` column should
not be reported as a positive control in the accuracy domain. The evidence-mass
control is unaffected: it is computed on the token geometry, where the count is
the point.

A third caution stands regardless: these intervals are WMO-clustered but
**condition on one trained model**, and P0 has already shown that seed variance
on this task can exceed the effects being measured.

## The `provenance_rho` question P2 raises (finding #10)

`46_argo_redundancy.py --rho` defaults to `0.9` and is documented as *"within-provenance
noise correlation for the **same_provenance** family"*. Two things follow from
how it is actually applied, and neither is obviously intended:

* The **evidence** loop applies it per family (`0.9` for `same_provenance` and
  `exact`, `0.0` otherwise) — which matches the documentation.
* The **model** is built once with `provenance_rho=0.9` and used for *every*
  family, including the `separated` positive control. So the accuracy rows score
  a model configured differently from the one P0 trained and reported, and
  differently from P1, P5 and P7.

**Resolved in favour of scoring the registered model as registered.** The
accuracy rows now build with the default `rho=0.0` — the configuration P0
trained and reported them under — so P2's accuracy table and P0's baseline
table sit on one ladder again. `rho` keeps its documented job in the evidence
half, applied per family. Applying it uniformly to the model, including under
the `separated` positive control where there is no shared provenance to
correlate, was never what the flag described.

## P7 — what has to happen before the holdout result can be reported

The prospective test is the one package whose validity is a claim about
*order*, and that claim is currently false for three of its eight rows. The
mechanics are now fixed, but fixing the mechanics does not fix the number: the
2025 holdout has been read, so re-running the open would be a second look at a
spent cohort, and the plan allows one.

**Corrected, by re-scoring rather than by re-opening.** `50_prospective_test.py
--stage correct` re-scores the already-opened holdout from the checkpoints the
freeze record pins. That is deliberately not a second open: same freeze record,
same checkpoints, same cohort, same registered endpoint, and nothing was
retrained or retuned in between — only which bytes got loaded changed. The
correction is recorded in the freeze record with the before/after hashes, and
the stage **refuses to run** if the recorded open already loaded what was
frozen, so it cannot become a back door to a second look.

What moved, Gulf Stream: `dfs_oi_expert_cbottle` 0.5637 → **0.5646**,
`count_oi` and `uniform_oi` unchanged to 4 dp. The five `*_expertlocal` rows
were correct all along and did not move at all. Both tracks now agree on every
row to float precision.

One thing the correction cannot repair: the holdout was read before the fault
was found, so the registration claim for those three rows is weaker than for a
never-seen test — the number is now the right one, but it is a corrected result
rather than a virgin one. **If the `*_oi_expert` rows carry weight in the paper,
they deserve a fresh freeze against the 2026 floats when the GDAC has them.**
The `*_expertlocal` rows, which is where the headline `DFS − Uniform`
comparison lives, are unaffected.

## The CI bug this run exposed, and the fix

The seed-instability reported earlier was not only a finding, it was a
**methodological defect**. A cluster bootstrap resamples *floats* but conditions
on one *trained model*, so it estimates the uncertainty in "this model's score"
when the quantity of interest is "a model trained by this recipe". On this task
the seed component dominates, so single-seed intervals were several times too
narrow.

`clustered_ci.ci_rmse_across_seeds` / `ci_difference_across_seeds` now resample
seeds and, within each, that seed's floats. `seed_t_interval` gives the
small-sample t alternative on the seed means alone. Per-cluster sufficient
statistics are persisted in every artifact so the interval can be re-derived
without re-running a model.

| quantity | before (seed 1234 only) | after (seed+WMO) | seed-t |
|---|---|---|---|
| DFS − Uniform, TEMP | −0.0081 [−0.0119, −0.0048] **excludes zero** | **−0.0006 [−0.0078, +0.0077]** | −0.0006 [−0.0194, +0.0183] |
| DFS − Uniform, SALT | +0.0016 [−0.0014, +0.0049] | **+0.0012 [−0.0027, +0.0074]** | +0.0012 [−0.0055, +0.0079] |

Both routes agree: **there is no DFS advantage over Uniform on this table.** The
apparent significance was an artifact of conditioning on one trained model.

## Known problems and limitations

1. **Track B exists, but is not a second person.** It runs every package the
   plan asks for (P0, P1, P2, P5, P7, plus P3's rows and P6's calibration
   metrics) and re-implements the whole evaluation path, and it has found real
   defects doing so. It was written by the same author as Track A. §8's
   rationale — reducing the chance one implementation is unconsciously adjusted
   to match the other — still needs a separate implementer, so agreement below
   is evidence of arithmetic, not of science.
2. **ECCO is scored, but only under a labelled secondary protocol.** No later
   ECCO Central Estimate exists (V4r4 1992–2017 is the latest; CMR has no
   V4r5), so more data does not fix it. Instead the eras are shifted inside
   V4r4's coverage — train 2000–2012 / val 2013–2014 / eval 2015–2017 — which
   is legitimate because the held-out float cohort is WMO-disjoint and
   year-independent. Never merged into the main headline table.
3. **EN4 assimilates the floats it is scored against.** It is an upper
   reference, not a peer.
4. **GODAS remains a reanalysis.** The GODAS-based results in the existing
   reports keep that circularity; only the new Argo line escapes it.
5. **FuXi-Ocean, WenHai, ADAF-Ocean cannot be run** (weights unreleased or no
   public implementation). ORCA-DL's weights need interactive OneDrive auth.
   XiHe is a different task (1/12° daily).
6. **The Senseiver's Argo adaptation is specified, not run.** Its official
   example reproduces; rule 2 remains.
7. **A flat sparsity curve is ambiguous** in exactly the way P2's control was
   designed to resolve, and the P5 report says so rather than reading a flat
   curve as robustness.
