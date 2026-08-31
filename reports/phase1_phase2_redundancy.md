# Phases 1 & 2 — operative mechanism, and an evaluation that contains the disease

Follows [`reports/phase0_operating_point.md`](phase0_operating_point.md).
Instruments: `experiments/39_redundancy_regimes.py`; properties asserted in
`tests/test_phase1_phase2_properties.py` (10 tests).
Run record: `outputs/cache/redundancy_regimes.json`.

> **Both phases' exit criteria are met, on geometry.** Dual-stream near-duplicates
> now collapse to **1.03x** at rho = 0.9 and track `n_eff` theory; vertical
> resampling conserves evidence to 2% across a 10x change in level spacing;
> token-count invariance is **exact to 1.00x**; and the positive control
> separates new evidence from re-ingested evidence by **2.43x**.
> The accuracy halves of both phases still require GODAS data and training.

---

## Phase 1 — making the mechanism operative

### 1a. Vertical support on profile tokens

A profile level was a point: horizontal area, **zero vertical extent**. Layer
thickness never entered the measure, so "the same profile reported at 2 dbar
versus 10 dbar" was not merely violated but *unrepresentable* — 5x the tokens,
each still a point, so 5x the evidence for the same water.

`vertical_quadrature` now integrates the basis over each layer with
Gauss-Legendre nodes, making a token a **volume**. Weights sum to the layer
thickness, so the rule integrates rather than averages.

| levels | spacing | total support (km^2.m) | total evidence |
|---|---|---|---|
| 8 | 100 m | 6.283e6 | 2.1166 |
| 16 | 50 m | 6.283e6 | 2.1482 |
| 40 | 20 m | 6.283e6 | 2.1570 |
| 80 | 10 m | 6.283e6 | 2.1583 |

Support is conserved **exactly**; evidence to **2%** across a 10x change in
resolution. The residual is the discretisation of a continuous integral, and
the quadrature converges by 3 nodes (2.1435 -> 2.1166 -> 2.1166 for n=1, 2, 3),
so `n_nodes = 3` is the default and `n_nodes = 1` recovers the old midpoint rule.

One correctness note worth recording: the noise density must be **constant per
unit volume**, because variance adds over independent sub-volumes. An earlier
draft of this measurement scaled the density with layer thickness and produced
a spurious 74% drift in evidence; the invariance above only holds with the
correct convention.

### 1b. Provenance as noise covariance

Platform / product identity now lives in the **noise model**, never the content
embedding — instance identity in the content channel would let the model
memorise platforms and leak into a held-out-float evaluation.

`whiten_provenance` implements the plan's `Cov = D + U diag(c) U^T` as block
equicorrelation over provenance groups, whose inverse square root is analytic,
so whitening stays O(N). `rho = 0` reproduces diagonal whitening exactly
(asserted by test).

**Group mass against `n_eff = k/(1+(k-1)rho)`**, predicted as
`n_eff*s/(1+n_eff*s)`, reported as measured / predicted:

| rho | k=2 | k=4 | k=8 | max rel. error |
|---|---|---|---|---|
| 0.0 | 0.674 / 0.667 | 0.805 / 0.800 | 0.892 / 0.889 | 1.0% |
| 0.5 | 0.579 / 0.571 | 0.623 / 0.615 | 0.647 / 0.640 | 1.3% |
| 0.9 | 0.521 / 0.513 | 0.527 / 0.520 | 0.531 / 0.523 | 1.5% |
| 0.99 | 0.509 / 0.501 | 0.510 / 0.502 | 0.510 / 0.502 | 1.6% |

Agreement is within 1.6% throughout — consistent with the 5.5% RFF kernel error
at p = 256. **Phase 1's exit criterion — "the measured group-mass curve matching
`n_eff` theory" — is met.**

A behaviour worth flagging because it looks wrong and is not: raising rho
*increases* total evidence over a full sample (43.8 -> 65.4 at rho = 0.9).
Correlated noise makes the common mode ill-determined but the **differences**
precise — two thermometers with a shared bias measure their difference well.
A profile's *shape* therefore gains evidence, while exact duplicates (zero
deviation from the group mean) still collapse. Both are the same algebra.

### 1c. Isolating the mechanism

The thin-stack rows (`dfs_expertlocal_cbottle` vs `..._oi_expert_...`) already
exist in `build_row`, so running DFS/uniform/Perceiver without the frozen OI
residual needs no new code. Measuring that effect size needs training and is
**not done** — see Limitations.

---

## Phase 2 — putting the pathology into the evaluation

The plan's diagnosis: *"the evaluation contains none of the disease the method
cures."* Six regimes now construct it. All are geometry, so they run without
GODAS fields or training.

### Duplication families — group mass of the attacked column

| regime | k=1 | k=2 | k=4 | k=8 | k8/k1 | discount |
|---|---|---|---|---|---|---|
| exact | 1.808 | 2.641 | 3.547 | 4.423 | 2.446 | 0.793 |
| jittered (25 km) | 1.808 | 2.640 | 3.555 | 4.519 | 2.499 | 0.786 |
| **dual_stream** (rho=0.9) | 0.913 | 0.940 | 0.954 | 0.961 | **1.053** | **0.992** |
| **separated** (control) | 1.808 | 3.187 | 6.209 | 10.764 | **5.953** | 0.292 |

**The dual-stream float — the paper's opening example — now collapses to 1.05x
at k=8.** Before Phase 1b only exactly-identical supports collapsed, and the
scenario could not be demonstrated at all.

**The positive control is what makes the rest meaningful.** `separated` — k
genuinely new columns, far apart — grows 5.95x while `exact` grows 2.45x, a
**2.43x separation**. Without that contrast, "suppresses redundancy" and
"learned to ignore the profile stream" produce identical flat curves. The
estimator is distinguishing new evidence from re-ingested evidence, not
discarding profiles.

(`separated` reaches 5.95 rather than 8.0 because the box is 4410 km wide
against a 1544 km x length scale, so even well-spread columns retain some
genuine overlap. That is physics, not saturation.)

### Dual-stream vs `n_eff`

| rho | measured ratio | `n_eff(2, rho)` |
|---|---|---|
| 0.0 | 1.461 | 2.000 |
| 0.5 | 1.162 | 1.333 |
| 0.9 | **1.030** | 1.053 |
| 0.99 | **1.004** | 1.005 |

Tracks theory closely at high correlation. At low rho the measured ratio sits
below `n_eff` because ridge saturation caps it — the Phase 0 effect, not a
Phase 1 failure.

This required getting the grouping right: correlation pairs **per level**
(real-time and delayed-mode are two reprocessings of the *same* measurement),
not per column. Grouping whole columns makes the 16 differing levels the
dominant term and their amplified deviation swamps the collapse — that draft
gave 1.39 instead of 1.03.

### Clustered vs uniform, at equal token count

| sampling | total evidence |
|---|---|
| uniform | 92.571 |
| clustered | 63.909 (**-31.0%**) |

Same number of tokens, one third less evidence. Clustering *is* redundancy and
the estimator prices it.

### Token-count invariance

| layout | tokens | evidence |
|---|---|---|
| 4x4 patches | 664 | 92.571 |
| 2x2 patches | 1504 (2.3x) | 92.719 (**1.00x**) |

Same field, same total support, 2.3x the tokens — **evidence unchanged to two
decimal places.** Token layout is a free representational choice and cannot be
used to buy influence. Asserted by test, per the plan's requirement.

---

## Status against the plan's exit criteria

| phase | criterion | status |
|---|---|---|
| 1 | dual-stream near-duplicates collapse toward one, matching `n_eff` | **PASS** (1.03x at rho=0.9; <=1.6% from theory) |
| 1 | thin-stack effect size measured | **NOT DONE** — needs training |
| 2 | redundancy regimes constructed | **PASS** — six regimes |
| 2 | token-count invariance asserted by test | **PASS** — 1.00x, in the suite |
| 2 | robustness sweep with a positive control | **PASS** — 2.43x separation |
| 2 | measurement harness (RS + reliance + RMSE triple) | **NOT DONE** — needs training |
| 2 | control ladder / property matrix filled by measurement | **PARTIAL** — geometry cells only |
| 2 | drift-law figure; pre-registered gate evaluated | **NOT DONE** — needs training |

Suite: **328 passed, 4 skipped** (10 new property tests).

## Limitations

- **No GODAS data on this machine**, so every number here is geometry:
  `omega` depends on support, noise, provenance and variable group, and field
  values never enter it. That covers the mechanism cleanly and covers *none* of
  the accuracy claims. Representation shift, the control ladder (raw mass /
  learned gain / within-modality / thinning+superobbing), the RMSE triples and
  the drift-law figure all need `13_download_godas.py` plus training.
- **Vertical levels are uniform** under the existing coordinate convention
  (`z = level index / (Z-1)`), so layer thickness is 63.3 m everywhere. Real
  GODAS levels thicken with depth; using true edges is a small follow-up once
  the data is present, and would change the *distribution* of evidence down the
  column though not the invariance.
- **Single seed, 4-6 synthetic months.** The three-seed rule applies to reported
  accuracy; no accuracy is reported. The ratios above are geometric and stable,
  but the small-sample caveat stands for anything quoted downstream.
- **rho is a free parameter.** 0.9 is a plausible value for a reprocessed
  stream, not a measured one. Estimating rho from real dual-stream Argo pairs is
  the honest version and belongs with the Phase 4 observational work.
- `jittered` at 25 km barely differs from `exact` (0.786 vs 0.793) because 25 km
  is far inside the 1544 km horizontal length scale. A jitter sweep spanning the
  length scale would be more informative and is cheap to add.
