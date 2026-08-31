# Phases 1 & 2 — making the mechanism operative, and putting the pathology into the evaluation

Follows [`reports/phase0_operating_point.md`](phase0_operating_point.md).
Instruments: `experiments/39_redundancy_regimes.py`,
`tests/test_phase1_phase2_properties.py`.
Run record: `outputs/cache/redundancy_regimes.json`.

> **Result.** Both Phase 1 mechanisms are implemented and both exit criteria are
> met: vertical resampling now conserves evidence (2% drift across a 10× change
> in level spacing, previously unrepresentable), and dual-stream near-duplicates
> collapse along `n_eff` to within 2% at high correlation. Phase 2's redundancy
> regimes are built and measured, including the **positive control** that
> separates "suppresses redundancy" from "ignores the stream" — the estimator
> passes it at 2.4×.

---

## Phase 1a — vertical support on profile tokens

**The gap.** A profile level was a point carrying a horizontal area and *zero
vertical extent*, so layer thickness never entered the measure. The plan's
"same profile at 2 dbar versus 10 dbar" scenario was not merely violated but
**unrepresentable**: 5× as many tokens, each still a point, so 5× the evidence
for the same water.

**The fix.** `vertical_quadrature` places Gauss–Legendre nodes across each
layer with weights summing to the layer thickness, so `integrate_support`
integrates rather than samples and a token becomes a *volume*.

| levels | spacing | total support (km²·m) | total evidence |
|---|---|---|---|
| 8 | 100 m | 6.283e6 | 2.1166 |
| 16 | 50 m | 6.283e6 | 2.1482 |
| 40 | 20 m | 6.283e6 | 2.1570 |
| 80 | 10 m | 6.283e6 | 2.1583 |

Support is conserved **exactly**; evidence drifts **2.0% across a 10× change in
vertical resolution**, the residual being finite quadrature of a continuous
integral. Quadrature converges by 3 nodes (1 node → 2.1435, 2 → 2.1165, 3 →
2.1166, 5 → 2.1166), so `n_vertical_nodes = 3` is the default and
`n_nodes = 1` recovers the old midpoint behaviour exactly.

**One correctness note that cost a wrong answer first time.** The noise density
must be *constant per unit volume*: for an integral over volume, variance adds
over independent sub-volumes, so `lambda = n · V` with `n` fixed. Scaling `n`
with layer thickness — which I did initially — makes evidence grow with
resolution and looks like a failed invariance.

## Phase 1b — provenance as noise correlation

**The gap.** Only *exactly identical* supports collapsed, so P1'(ii) — collapse
of correlated near-duplicates along `n_eff = k/(1+(k-1)ρ)` — was inoperative,
and the paper's opening example (one float distributed via real-time and
delayed-mode streams) could not be demonstrated.

**The fix.** `Cov = D + U diag(c) Uᵀ` as block equicorrelation over provenance
groups. Within a group of `m`, `Cov = lam·[(1-ρ)I + ρ11ᵀ]`, whose inverse square
root is analytic, so whitening stays O(N). Provenance drives the **noise model
only** — instance identity never enters the content embedding, so platforms
cannot be memorised and cannot leak into a held-out-float evaluation.

Group mass against the prediction `n_eff·s/(1+n_eff·s)`:

| ρ | k=2 | k=4 | k=8 | max rel. error |
|---|---|---|---|---|
| 0.0 | 0.674 / 0.667 | 0.805 / 0.800 | 0.892 / 0.889 | 1.0% |
| 0.5 | 0.579 / 0.571 | 0.623 / 0.615 | 0.647 / 0.640 | 1.3% |
| 0.9 | 0.521 / 0.513 | 0.527 / 0.520 | 0.531 / 0.523 | 1.5% |
| 0.99 | 0.509 / 0.501 | 0.510 / 0.502 | 0.510 / 0.502 | 1.6% |

(measured / predicted). Agreement is within 1.6% everywhere — the residual is
consistent with the 5.5% RFF kernel error at p=256. **Phase 1's exit criterion —
"the measured group-mass curve matching `n_eff` theory" — is met.**

**A subtlety worth recording.** Correlated noise *raises* total evidence for a
profile column (ρ=0.9: 65.4 vs 43.8 at ρ=0). That is correct, not a bug: with
correlated errors the common mode is ill-determined but the *differences* are
precise — two thermometers with a shared bias measure their difference well. A
profile's *shape* therefore gains evidence while exact duplicates, whose
deviation is zero, still collapse. The two behaviours are different scenarios,
not a contradiction.

**And a modelling trap it exposed.** Grouping all 16 levels of a column into one
provenance group makes dual-stream collapse *fail* (ratio 1.39 at ρ=0.9 against
`n_eff` 1.05), because the 16 differing levels dominate and their deviation is
amplified by `(1-ρ)^(-1/2)`. Real-time and delayed-mode are two reprocessings of
the *same measurement*, so the correlated pair is **level j of stream A with
level j of stream B** — pairing must be per level, not per column. With that
fixed, the ratio is 1.030 against a predicted 1.053.

## Phase 2 — the redundancy regimes

Six regimes, all geometry (no training, no GODAS fields needed).

### Duplication families — group mass of the attacked column

| regime | k=1 | k=2 | k=4 | k=8 | k8/k1 | discount |
|---|---|---|---|---|---|---|
| `exact` | 1.808 | 2.641 | 3.547 | 4.423 | 2.446 | 0.793 |
| `jittered` (25 km) | 1.808 | 2.640 | 3.555 | 4.519 | 2.499 | 0.786 |
| **`dual_stream`** (ρ=0.9) | 0.913 | 0.940 | 0.954 | 0.961 | **1.053** | **0.992** |
| **`separated`** (control) | 1.808 | 3.187 | 6.209 | 10.764 | **5.953** | 0.292 |

`dual_stream` is essentially perfect collapse — eight copies of one float carry
1.05× one float's evidence. This is the paper's opening example, now
demonstrable for the first time.

### The positive control

Without it, "suppresses redundancy" and "learned to ignore the profile stream"
produce the same flat curve and cannot be told apart. `separated` supplies `k`
genuinely new columns far apart; evidence must grow.

**Separation = 2.43×** (`separated` 5.95 vs `exact` 2.45). The estimator
distinguishes new evidence from re-ingested evidence.

`separated` reaching 5.95 rather than 8.0 is expected and physical: the box is
4411 km wide against a 1544 km x length scale, so columns spread across it are
still partially within one correlation length. Genuinely independent water would
need a larger domain.

### Clustered vs uniform, at equal token count

| sampling | total evidence |
|---|---|
| uniform | 92.57 |
| clustered (±5% of the box) | 63.91 (**−31.0%**) |

Same token count, same supports, different geometry — clustering *is* redundancy
and the measure sees it.

### Token-count invariance

| layout | tokens | total evidence |
|---|---|---|
| 4×4 patches | 664 | 92.571 |
| 2×2 patches | 1504 (2.3×) | 92.719 (**1.00×**) |

The same field at 2.3× the token count and the same total support yields
identical evidence. Token layout is a free representational choice and cannot be
used to buy influence. Asserted by test, per the plan's requirement.

## Tests

`tests/test_phase1_phase2_properties.py` — 10 tests, all passing, no data
required:

- vertical quadrature weights sum to layer thickness (integrates, not averages)
- vertical resampling conserves support exactly and evidence to <5%
- `n_nodes=1` reproduces the midpoint rule exactly
- `ρ=0` reproduces diagonal whitening exactly (strict generalisation)
- provenance group mass matches `n_eff` theory to <5% across ρ and k
- high correlation collapses duplicates toward one
- independent tokens still count (positive control)
- total evidence tracks support, not token count
- clustered sampling carries less evidence than uniform

Full suite: **328 passed, 4 skipped**.

## What is NOT done

Phase 2 has an accuracy half that needs a trained model and GODAS fields, and
neither is on this machine. Specifically **not** attempted:

- the **measurement harness**: representation shift paired with information
  reliance, reported as a triple with RMSE, in physical units (thermocline
  displacement, mixed-layer depth, spurious 0–700 m heat content)
- the **control ladder**: raw physical mass → + learned per-modality gain →
  within-modality normalisation → **thinning + superobbing**, which is the row
  that decides the paper and is count-independent by construction
- the **property matrix** filled by measurement with a run citation per cell
- the **drift-law figure** and the pre-registered gate evaluation
- Phase 1c, **isolating the mechanism on the thin stack** without the frozen OI
  residual — the rows exist (`dfs_expertlocal_cbottle` vs
  `dfs_oi_expert_cbottle`) but the effect size needs training

All of these are unblocked by `experiments/13_download_godas.py`; the geometry
harness built here is what they would plug into.

## Reading this against Phase 0

Phase 0 (corrected) found the mechanism was already operative for exact
duplicates — 0.676 discount before any Phase 0 change — and that the audit's
`s = 0.009` was a metric artefact. Phases 1 and 2 do not overturn that; they
extend the reach of the mechanism to cases it genuinely could not represent
(vertical resampling, correlated near-duplicates) and build the regimes in which
it can be *seen*.

The open question is unchanged and remains the one that matters: the mechanism
demonstrably discounts redundancy, yet three independent evaluations reported
~1% end-to-end effects. The regimes here are the instrument for testing the
remaining explanation — that the evaluation contains no redundancy to remove —
but that test needs the accuracy half, and therefore the data.

## Caveats

- All numbers are geometry on synthetic all-ocean fields. `omega` depends on
  support, noise, provenance and variable group alone, so this is exact for an
  all-ocean box; land would remove tokens and *raise* per-token leverage, making
  these the pessimistic end.
- Single basis seed; 4–6 source months. The standing three-seed rule applies to
  reported accuracy, and no accuracy is reported here.
- Profile layer thickness uses the uniform-level convention the `z` coordinate
  already assumes (`z` = level index / (Z−1), giving 63.3 m). True GODAS level
  edges are non-uniform (5 m near the surface to ~950 m) and would be more
  physical; a small follow-up once the data is present.
- ρ is a free parameter here, set to 0.9 for the dual-stream demonstration. A
  real value would come from comparing real-time and delayed-mode Argo pairs,
  which is Phase 4 data work.
