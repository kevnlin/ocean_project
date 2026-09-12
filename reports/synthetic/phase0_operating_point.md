# Phase 0 — operating point calibration: gate result

Registered rule: [`doc/phase0_registration.md`](../../doc/phase0_registration.md),
commit `6f300e3`, committed **before** any number below was read.
Instrument: `experiments/synthetic/29_dfs_operating_point.py`.
Run record: `outputs/cache/dfs_operating_point.json`.

> **Correction notice.** An earlier revision of this report concluded the gate
> failed. That conclusion was wrong: it rested on a bug in this report's own
> measurement, described in §0. The corrected result reverses it.

> **Verdict: the gate opens, but not for the reason the audit expected.** The
> redundancy mechanism is *operative and always was* — eight exact duplicates
> collapse to 3.06× one copy's evidence (a 71% discount) at the registered
> prior, and the discount responds strongly to calibration (0.25 → 0.88 across
> the family). What Phase 0 actually refutes is the audit's own premise: the
> `s = 0.009` figure that motivated the gate is a **metric artefact**, not a
> physical operating point.

---

## 0. The bug, and what it invalidated

`duplicate_profile_attack` returns `original(664) ++ block ++ block ...`,
appending the k−1 copies **at the tail** while the attacked column stays at
`[0:16]`. The first revision measured the attacked group as `w[:16*k]` — a
contiguous prefix — which for k=8 is `w[0:128]`: the first **eight different
profile columns** of the original set, not the eight copies.

That statistic grows ~linearly in k by construction, because it sums eight
distinct columns of water and compares them to one. It measured nothing about
duplication and produced an apparent "no collapse" result (k8/k1 = 6.93,
discount 0.15).

Corrected, the group is the two disjoint ranges `[0:16] ∪ [664 : 664+16(k−1)]`,
now asserted in code to be k exact copies of one column before it is scored.
Every duplicate number below is post-fix.

## 1. What Phase 0 changed

| item | before | after |
|---|---|---|
| support measure | dimensionless `1` (`torch.ones(N,1)`) | **physical footprint in km²** |
| noise | pilot constants 0.08 / 0.35, dimensionless | **error-variance density as a noise area (km²)** |
| kernel length scales | `(0.35, 0.35, 0.25, 2.0)` normalised | **`(1544 km, 973 km, 237 m, 2 months)`**, numerically equivalent |
| random features | 32 | **256** (`N_FEATURES_LEGACY = 32` keeps the frozen record reproducible) |

The per-token operating point is now readable as
`s_token = |psi~|² = support_area / noise_area` — support divided by
error-variance density. The declared noise areas set `s_token = 1` at each
stream's nominal footprint; that is a declared normalisation, not a measured
GODAS error table.

The reparameterisation exposed the accidental anisotropy the audit predicted:
the same `0.35` is **1544 km in x but 973 km in y** (factor 1.59), purely
because the box spans 50° of longitude and 25° of latitude. (The work plan
quotes 773 × 481 km, almost exactly half; that looks like a half-width
convention. The *ratio* — the part that matters — agrees.)

## 2. The operating point was never `0.009`

6 source months, 664 live tokens each (384 profile points, 140 surface patches,
140 SSH patches).

| | legacy (F=32, unit support) | Phase 0 (F=256, km² support) |
|---|---|---|
| **`s_token`** (true per-token) | **8.34** | **0.95** |
| mean `omega` | 0.0432 | 0.0657 |
| `sum omega` | 28.68 / 32 — **89.6% of ceiling** | 43.62 / 256 — **17.0%** |
| `omega/(1-omega)` | 0.045 | 0.070 |

**`s = omega/(1-omega)` is a biased-low estimator and should not be used as the
operating point.** `omega = s/(1+s)` is the *lone-token* identity; inverting it
is valid for a set of one. With 664 tokens competing for a bounded evidence
pool, every `omega_i` sits well below `s_i/(1+s_i)`, so the inversion
understates `s` by the competition factor — here **0.95 → 0.070, a factor of
~14**. The audit's `s = 0.009` is that artefact, and the whole "capped at ~1% by
construction" argument rests on it.

Measured properly, the estimator was **never at the NEO/Berner corner**:
`s_token` was 8.34 before Phase 0 and 0.95 after.

Phase 0 did fix a real problem, just a different one. The legacy config sat at
**89.6% of the feature ceiling** — `omega` was being crushed by a 32-dimensional
basis, not by the noise. Raising `p` to 256 gave 17% headroom.

## 3. The s-family

| `c` | `s_token` | mean `omega` | `sum omega` (% of F) | duplicate discount |
|---|---|---|---|---|
| 0.001 | 950 | 0.181 | 120.4 (47.0%) | **0.876** |
| 0.01 | 95 | 0.145 | 96.5 (37.7%) | 0.847 |
| 0.1 | 9.5 | 0.107 | 70.7 (27.6%) | 0.803 |
| 0.25 | 3.8 | 0.090 | 60.0 (23.5%) | 0.777 |
| 0.5 | 1.9 | 0.078 | 51.8 (20.3%) | 0.752 |
| **1 (prior)** | **0.95** | 0.066 | 43.6 (17.0%) | **0.722** |
| 2 | 0.475 | 0.054 | 35.6 (13.9%) | 0.682 |
| 5 | 0.19 | 0.039 | 25.6 (10.0%) | 0.613 |
| 10 | 0.095 | 0.028 | 18.9 (7.4%) | 0.546 |
| 100 | 0.0095 | 0.007 | 4.6 (1.8%) | 0.248 |

The discount is **monotone in `s` across the whole family**, from 0.25 at the
noisy end to 0.88 at the quiet end. This is exactly the response the gate was
built to look for.

`sum omega` saturating near 120 (of 256) is itself informative: the 664-token
set spans roughly **120 effective dimensions**, i.e. the estimator is finding
~5× redundancy at the set level.

## 4. Duplicate collapse

`k` bit-exact copies of one biased profile column; ideal is flat in `k`,
independent-vote failure is `k8/k1 = 8`.

| `c` | k=1 | k=2 | k=4 | k=8 | k8/k1 | discount |
|---|---|---|---|---|---|---|
| 0.001 | 2.841 | 3.672 | 4.524 | 5.317 | 1.871 | **0.876** |
| 1 (prior) | 1.091 | 1.697 | 2.431 | 3.217 | 2.949 | **0.722** |
| 100 | 0.117 | 0.226 | 0.419 | 0.735 | 6.261 | 0.248 |

**Legacy vs Phase 0, same corrected measurement, at each config's own prior:**

| config | k8/k1 | discount |
|---|---|---|
| legacy — F=32, unit support, pilot noise | 3.267 | **0.676** |
| Phase 0 — F=256, km² support, `c` = 1 | 3.061 | **0.706** |

The mechanism was already working before Phase 0 (0.676) and Phase 0 improves it
only marginally (0.706). **Calibration was not the blocker.**

## 5. RFF convergence

Kernel RMSE vs the exact Gaussian, 5 basis seeds, physical length scales:

| p | 32 | 64 | 128 | **256** | 512 | 1024 |
|---|---|---|---|---|---|---|
| RMSE | 0.164 | 0.121 | 0.081 | **0.055** | 0.039 | 0.030 |

The audit's numerical premise is confirmed — 16.4% kernel error at `p = 32`
against ~1% effects — and the raise to 256 is a genuine improvement. It is not
sufficient: 256 still gives **5.5%**, and error falls only as `p^(-1/2)`, so
`p ≥ 1024` would be needed for the numerics to sit below the effect size.

## 6. Exit criteria

| # | criterion | status |
|---|---|---|
| 1 | `s = O(1)`, no degeneracy warning | **PASS** on `s_token` (0.95; 8.34 pre-Phase-0). The `omega/(1-omega)` reading of 0.070 is the biased metric of §2 and should be retired |
| 2 | `s*` registered with the rule, timestamped before the numbers | rule registered (`6f300e3`); **`s*` not selected** — see below |
| 3 | conservative refinement (P2') preserved under new units | **PASS** — mass-conservation tests green |
| 4 | RFF convergence table on record | **PASS** — §5 |

`s*` is **not selected**. The registered rule picks it on validation accuracy
`J` (one-standard-error, three seeds); `J` needs GODAS fields and training, and
no GODAS data is on this machine. The registration's fallback clause applies
verbatim: report the calibration curve, defer `s*`, do **not** substitute a
geometry-only criterion. Selecting on the duplicate discount was excluded in
advance and stays excluded — and it matters that this is still enforced now that
the discount looks *good*, not only when it looked bad.

## 7. Decision point — for the PI

Decision-point outcome **1** obtains: the duplicate discount is substantial and
rises with calibration. The plan's instruction for outcome 1 is to proceed to
Phase 1.

But the reason the gate existed is refuted, and that changes what Phase 1 is
for:

- **The premise was wrong.** "Every mechanism effect is capped at about 1% by
  construction" does not survive measurement. The mechanism delivers a 68–88%
  duplicate discount, before and after calibration.
- **The `s = 0.009` figure is an artefact** of inverting a lone-token identity
  on a 664-token set. Any argument in the theory documents that leans on
  `s << 1` should be re-checked against `s_token`, which was 8.34 at the time
  the figure was quoted.
- **So why did three independent evaluations report ~1% end-to-end effects?**
  Not because the mechanism is inert — it demonstrably is not. The remaining
  explanation is the plan's own Phase 2 diagnosis: *the evaluation contains none
  of the disease the method cures.* Profiles are drawn uniformly from a gridded
  reanalysis with no duplicate stream, no overlapping product, no clustered
  sampling. A mechanism that removes redundancy cannot show a gain on data with
  no redundancy in it.

**Recommendation: go to Phase 2, not Phase 1.** Phase 1's stated purpose is to
"make the mechanism actually operative"; §4 shows it already is for the
duplicate family. The two Phase 1 items that remain independently justified are
vertical support on profile tokens (the 2-dbar vs 10-dbar scenario genuinely
cannot be represented today) and provenance-as-noise-covariance (needed for
*near*-duplicates; only exactly-identical supports collapse now). Those are
worth doing — but as enablers of specific Phase 2 scenarios, not as repairs to a
dead mechanism.

This is a claim-scope change and is a PI decision.

## Caveats

- GODAS fields are absent, so token **geometry** is rebuilt on synthetic
  all-ocean fields. `omega` is a function of geometry, noise and variable group
  — field values never enter it — so the operating point is exact for an
  all-ocean box. Land would remove tokens, lowering N and *raising*
  `mean omega ≤ F/N`; the synthetic case is therefore the pessimistic end for
  the discount.
- Single basis seed for the operating point and sweep (5 seeds for §5). The
  three-seed rule applies to reported accuracy; no accuracy is reported here.
- The declared noise areas are a normalisation, not measured GODAS observation
  errors. Replacing them is open Phase 0 work and is what would make `c = 1` a
  physical prior rather than a convention.
- §4's legacy-vs-Phase-0 comparison uses one seed and 6 months; the gap (0.676
  vs 0.706) is small enough that it should not be quoted as an improvement
  without replication.
