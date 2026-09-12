# DFS-Attention — Success Criterion (plan Section 12)

*Three legs, one table. A method that scores well on duplication sensitivity while ignoring the profiles fails leg 2; a method that uses the profiles but is destabilised by duplicates fails leg 1. Both must hold while accuracy stays competitive.*

Runs: prefix `fullA`, seeds [1234, 1235, 1236], protocol_v1 (276/36/12 months, anomaly target, unobserved-only RMSE). Mean ± sd over seeds.


## Leg 3 · Reconstruction accuracy (pinned test months)

| variant | TEMP RMSE (°C) | SALT RMSE (PSU) | skill vs floor (TEMP) | params | seeds |
|---|---:|---:|---:|---:|---:|
| Standard Perceiver | 0.5243 ±0.0066 | 0.1214 ±0.0007 | 5.0 ±1.2 % | 1.05 M | 3 |
| Fixed resampler | 0.5231 ±0.0022 | 0.1218 ±0.0016 | 5.3 ±0.4 % | 1.13 M | 3 |
| MBCA | 0.5221 ±0.0028 | 0.1206 ±0.0017 | 5.4 ±0.5 % | 1.05 M | 3 |
| DFS-Attention | 0.5251 ±0.0052 | 0.1202 ±0.0018 | 4.9 ±0.9 % | 1.14 M | 3 |

Train-climatology floor on the test months: TEMP 0.5521 °C.


All variants trained with identical settings (`steps`=20000, `val_every`=2500, `patience`=5, `lr`=0.0003, `warmup`=2000, `obs_query_frac`=0.25, `input_noise`=0.05, `weight_decay`=0.01, `grid_drop`=0.3, `d_model`=128, `n_latent`=128, `n_self_blocks`=4); only the fusion rule differs. DFS-Attention carries the extra parameters of its evidence resampler and background attention, noted in the params column.


## Leg 1 · Duplication sensitivity (validation months, real fields)

Relative change of the decoded field when observations are re-ingested carrying no new information. `duplicate_half` adds the copies anonymously; `duplicate_half_declared` gives them the provenance of the float cycle they came from (a real duplicated Argo record), which is the case DFS-Attention consolidates exactly.

| probe | Standard Perceiver | Fixed resampler | MBCA | DFS-Attention |
|---|---:|---:|---:|---:|
| `duplicate_half` | 0.0112 | 0.0114 | 0.0199 | 0.0014 |
| `duplicate_half_declared` | 0.0112 | 0.0114 | 0.0199 | 0.0001 |
| `patch_refine_2x` | 0.2039 | 0.0320 | 0.0146 | 0.0206 |
| `profile_resample_2x` | 0.0048 | 0.0075 | 0.0468 | 0.0061 |

## Leg 2 · Observation retention

How much the reconstruction actually depends on the profiles.

* **withheld** — RMSE penalty when the profiles are removed entirely. A model that ignores Argo pays nothing.
* **marginal** — RMSE penalty of halving the profile count from 3000 to 1500. A model that has saturated on the observations it already reads pays nothing here either, even if it does use *some* of them.

Both are penalties, so bigger is better: they measure how much of the answer is actually coming from the observations.

| variant | TEMP RMSE, all inputs | without profiles | withheld | marginal (3000→1500) | seeds |
|---|---:|---:|---:|---:|---:|
| Standard Perceiver | 0.5243 ±0.0066 | 0.5314 ±0.0103 | +1.3 % | +0.1 % | 3 |
| Fixed resampler | 0.5231 ±0.0022 | 0.5230 ±0.0040 | -0.0 % | +0.0 % | 3 |
| MBCA | 0.5221 ±0.0028 | 0.5268 ±0.0044 | +0.9 % | +0.0 % | 3 |
| DFS-Attention | 0.5251 ±0.0052 | 0.5341 ±0.0167 | +1.7 % | -0.1 % | 3 |

### Evidence budget of the observing system (DFS-Attention)

What the estimator says the month's observations are worth, before any training. `profile share` is retention measured at the evidence level rather than through the loss.

| target scale | obs tokens | DFS | DFS / token | profile share | surface share | neighbour cut |
|---|---:|---:|---:|---:|---:|---:|
| coarse | 6146 | 4293.8 ⚠ | 0.699 | 0.930 | 0.070 | 0.2531 |
| protocol | 6146 | 5829.7 | 0.949 | 0.936 | 0.064 | 0.0152 |
| fine | 6146 | 5884.1 | 0.957 | 0.935 | 0.065 | 0.0100 |

The background (WOA) contributes exactly 0 by construction — it is what the evidence is measured against. `neighbour cut` is the localisation diagnostic (docs/dfs_attention.md §7): below 0.05 the localised solve is effectively exact on this geometry.

> ⚠ At the **coarse** target the cut is 0.253: the long length scales of a coarse target make the observations correlated well beyond the 32-token neighbourhood, so that row's DFS is an **over-estimate** — the true coarse-target evidence is lower still, which only strengthens the direction of the scale trend. Truncation always biases DFS upward.


**The most informative number in this report.** The estimator says the profiles carry **94 %** of the month's independent information, yet removing them costs the trained model only the couple of percent RMSE in leg 2. The evidence is there and is correctly identified; what fails to exploit it is downstream of the evidence estimate. That localises the remaining gap to training and decoding rather than to the fusion rule — consistent with the month-identity recall overfit that caps every variant in this family at a ~0.94 validation score by ~5k steps.


## Verdict

| variant | leg 1 duplication (declared) | leg 2 retention | leg 3 accuracy vs best | all three |
|---|---|---|---|---|
| Standard Perceiver | 0.0112 fail | +1.3 % fail | +0.4 % pass | no |
| Fixed resampler | 0.0114 fail | -0.0 % fail | +0.2 % pass | no |
| MBCA | 0.0199 fail | +0.9 % fail | +0.0 % pass | no |
| DFS-Attention | 0.0001 pass | +1.7 % fail | +0.6 % pass | no |

Thresholds (fixed before the runs, stated here so they are not read off the results): leg 1 passes below 0.01 relative output change; leg 2 passes above a 2 % RMSE degradation when profiles are withheld; leg 3 passes within 3 % of the best variant's TEMP RMSE.


## Section-11 evidence probes

The mechanism-level requirements (exact duplication, depth complementarity, smooth vs thermocline density, resolution sweep, horizontal cases) are reported in full by [`dfs_evidence_probes.md`](dfs_evidence_probes.md). Headline numbers:

- exact duplication (×2/×4/×8, whole profiles, and real-time + delayed-mode copies): largest ΔDFS = **1.5e-05** on a base of 95.78;
- one column's 4 depth bands carry 3.99 DFS — different depths are complementary, not redundant;
- the same 2 dbar sampling at a 10 m target yields **19.7** DFS across a thermocline vs **10.8** in a smooth column of equal anomaly amplitude;
- the resolution sweep is monotone: 4.2 → 6.3 → 9.6 → 14.3 → 19.7 → 21.1 DFS for Δz = 250 → 100 → 50 → 25 → 10 → 5 m.
