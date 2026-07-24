# Invariance Test Summary (Task 6)

*Architectural probes: random-init models, fixed seed, identical trunks — the invariances under test are structural, not learned. Trained-model sensitivity: Stage-A toy (converged models, held-out fields). Definitions: docs/token_measure_definition.md.*

**Unit-test verdict:** `89 passed in 74.75s (0:01:14)` (suites: token API, profile encoder, fusion, mbca_invariance [A+B], token_refinement [C], profile_resampling [D]).

## Architectural probes — relative output change (lower = more invariant)

| probe | Standard Perceiver | Fixed resampler | MBCA | MBCA expectation |
|---|---|---|---|---|
| Test A — exact partition x2 | 4.3e-04 | 3.8e-04 | 5.5e-08 | **exact (0)** |
| Test A — exact partition x4 | 0.0012 | 0.0011 | 5.8e-08 | **exact (0)** |
| Test A — exact partition x8 | 0.0025 | 0.0023 | 4.3e-08 | **exact (0)** |
| Test B — duplication x2 (mass split) | 0.0023 | 0.0018 | 6.1e-08 | **exact (0)** |
| Test B — duplication x4 (mass split) | 0.0048 | 0.0034 | 4.0e-08 | **exact (0)** |
| Test B — duplication x8 (mass split) | 0.0069 | 0.0045 | 5.8e-08 | **exact (0)** |
| Test C — physical 2x refinement (pred) | 0.0241 | 0.0162 | 0.0058 | smallest |
| Test C — physical 2x refinement (latent) | 0.3544 | 0.1531 | 0.0843 | smallest |

Measure contract under physical 2x refinement: grid-modality total support mass ratio refined/coarse = **0.9950** (exact conservation = 1; token count grows ~4x).

## Trained-model sensitivity (Stage-A toy, converged)

| model | held-out RMSE | dup x8 shift | dup x8 RMSE | refine 2x shift |
|---|---|---|---|---|
| Standard Perceiver | 0.0905 | 0.6218 | 0.4251 | 0.4868 |
| Fixed resampler | 0.1727 | 0.0117 | 0.1725 | 0.9094 |
| MBCA | 0.0934 | 0.0000 | 0.0934 | 0.3204 |

Trained standard attention degrades catastrophically under duplication (RMSE 0.090 -> 0.425); trained MBCA is exactly invariant at equal accuracy.

## Reading guide

- **Exact rows** hold to float32 tolerance (<1e-5) for MBCA by construction: n children (k, v, w/n) reproduce the parent's attention contribution exactly.
- **Physical refinement** genuinely changes token content (patches cover different windows), so no method is exact; MBCA conserves the modality's total attention mass while standard attention lets it grow ~4x.
- **Profile resampling** is absorbed by the encoder (fixed physical bands + span mass): token count and masses are level-count independent; residual band-mass shift <10% from boundary half-intervals (test_profile_resampling).