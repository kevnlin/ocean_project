# Invariance Test Summary (Task 6)

*Architectural probes: random-init models, fixed seed, identical trunks — the invariances under test are structural, not learned. Trained-model sensitivity: Stage-A toy (converged models, held-out fields). Definitions: docs/token_measure_definition.md.*

**Unit-test verdict:** `114 passed in 40.36s` (suites: token API, profile encoder, fusion, mbca_invariance [A+B], token_refinement [C], profile_resampling [D]).

## Architectural probes — relative output change (lower = more invariant)

| probe | Standard Perceiver | Fixed resampler | MBCA | DFS-Attention | expectation |
|---|---|---|---|---|---|
| Test A — exact partition x2 | 0.0027 | 4.5e-04 | 8.6e-08 | 7.9e-08 | **exact (0)** for MBCA + DFS |
| Test A — exact partition x4 | 0.0079 | 0.0013 | 1.2e-07 | 9.9e-08 | **exact (0)** for MBCA + DFS |
| Test A — exact partition x8 | 0.0174 | 0.0028 | 1.0e-07 | 7.6e-08 | **exact (0)** for MBCA + DFS |
| Test B — duplication x2 (mass split) | 0.0173 | 0.0022 | 1.2e-07 | 1.0e-07 | **exact (0)** for MBCA + DFS |
| Test B — duplication x4 (mass split) | 0.0345 | 0.0040 | 1.3e-07 | 1.0e-07 | **exact (0)** for MBCA + DFS |
| Test B — duplication x8 (mass split) | 0.0481 | 0.0054 | 1.0e-07 | 1.1e-07 | **exact (0)** for MBCA + DFS |
| Test C — physical 2x refinement (pred) | 0.0882 | 0.0396 | 0.0327 | 0.0381 | smallest |
| Test C — physical 2x refinement (latent) | 0.3737 | 0.1205 | 0.3561 | 0.1755 | smallest |

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