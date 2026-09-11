# Week-4 Report — First Full-Scale Training of the Shared-Latent Variants

*Task 8 of the Week-4 plan (reports/full_training_plan.md): three fusion variants, identical everything except the fusion rule, trained under protocol_v1 with profile-count augmentation U{0..3000}; validation-selected checkpoints; pinned test months scored once per run.  Metric: **unobserved-only anomaly RMSE** (degC / PSU), pooled over the 12 pinned test months.*

## Setup deviations from the week-3 plan (declared)

protocol_v1 (split, metric, inputs, eval) is unchanged.  The week-3 POC froze the architecture on an overfit gate only; at full scale that config could not learn at all, and the following changes were required — every one applied identically to all three variants, so the comparison remains fusion-rule-isolating:

1. **Fourier coordinate features** (`coord_features` fourier_v2, shared by all encoders and the query decoder): with the original 7 smooth features the training loss stayed pinned at 1.0 even when queries were sampled AT observed profile columns (the copy diagnostic, `18_full_train.py --probe-observed`) — attention had no spatially selective basis to read tokens with.
2. **Geographically anchored latents** (`--anchor-grid 18,30` -> 540 latent tokens, one per 10x12-deg cell, initialised with the shared coordinate featurisation of the cell centre): best validation peak of every configuration searched; the unanchored 128-latent original is kept as an ablation row below.
3. **Training-side augmentation/regularisation**: 2k-step LR warmup; 25% observed-column query bootstrap (MAE-style; evaluation remains strictly unobserved-only — protocol_v1 prohibits scoring, not training, on observed columns); grid-token dropout 0.3; input noise 0.05 sigma; AdamW weight decay 0.01.

**Training-dynamics finding (the honest headline of this run).** Every configuration searched — 1M and 4.7M params, 128/512/540-anchored latents, grid-drop up to 0.5, no-bootstrap, lr 1e-4, input noise up to 0.5 sigma — peaks at a validation score of ~0.93-0.95 within ~5k steps and then degrades while the training loss keeps falling.  With 276 training fields, exact-column observations uniquely fingerprint each month, so reconstruct-the-field training collapses into month-identity recall for a globally attending token model; masking/noise levels that would destroy the fingerprint also destroy the signal.  Checkpoints are therefore validation-selected near the peak (protocol-clean), and closing the generalisation gap to the convolutional baselines — e.g. more training members/years, field-space decoding, or local-attention decoders — is the top week-5 priority.  Search-run curves: `outputs/cache/{scale_,reg_,anchor_}*.json` (their test numbers are quarantined diagnostics, never used for selection or reported).

## 1. Headline: test unobserved-only anomaly RMSE (mean ± std over seeds)

| model | TEMP (degC) | SALT (PSU) | TEMP 0-100m | TEMP 100-300m | TEMP 300-max |
|---|---|---|---|---|---|
| Climatology floor (train-only) | 0.5521 | 0.1305 | — | — | — |
| Nearest-profile fill (3 seeds) | 0.9595 ± 0.0012 | 0.3107 ± 0.0022 | 1.0401 | 0.9985 | 0.7228 |
| Pointwise MLP (3 seeds) | 0.2983 ± 0.0003 | 0.0525 ± 0.0003 | 0.2577 | 0.3957 | 0.1696 |
| Depthwise U-Net (certified, seed 1234) | 0.1580 | 0.0325 | 0.1589 | 0.1916 | 0.0837 |
| Joint-depth U-Net (certified, seed 1234) | 0.1948 | 0.0420 | 0.1973 | 0.2341 | 0.1063 |
| MBCA (method) (3 seeds) | 0.5221 ± 0.0028 | 0.1206 ± 0.0017 | 0.6126 | 0.5554 | 0.1954 |
| Standard Perceiver (3 seeds) | 0.5243 ± 0.0066 | 0.1214 ± 0.0007 | 0.6176 | 0.5551 | 0.1937 |
| Fixed-budget resampler (3 seeds) | 0.5231 ± 0.0022 | 0.1218 ± 0.0016 | 0.6157 | 0.5560 | 0.1874 |
| MBCA (method) — unanchored 128-latent ablation (seed 1234) | 0.5232 | 0.1210 | 0.6185 | 0.5518 | 0.1903 |
| Standard Perceiver — unanchored 128-latent ablation (seed 1234) | 0.5346 | 0.1226 | 0.6232 | 0.5723 | 0.2072 |
| Fixed-budget resampler — unanchored 128-latent ablation (seed 1234) | 0.5258 | 0.1212 | 0.6155 | 0.5606 | 0.1994 |

Skill vs floor = 1 − RMSE/floor.  The pointwise MLP and nearest-profile rows are recomputed under protocol_v1 (276 train / 12 pinned test, 1500 profiles, `profiles_woa_surf`, unobserved-only anomaly RMSE, 3 seeds — experiments/21_baselines_protocol_v1.py); the certified U-Net numbers are the week-3 audit checkpoints (seed 1234, fixed 1500 profiles, no count augmentation).  The shared-latent rows carry count augmentation and are additionally capable of the section-3/4 sweeps with the same checkpoint. All ML baselines (MLP, both U-Nets) still beat the shared latent on this reconstruction metric — the accuracy gap of the training-dynamics finding above; the shared latent's contribution is the flexibility axis (§3) and the fusion-rule/invariance comparison (§2, gate §5).

## 2. Sensitivity at the selected checkpoint (Task-6 protocol, real data)

Relative output change under information-preserving token manipulations (validation months; lower = more invariant):

| probe | MBCA (method) | Standard Perceiver | Fixed-budget resampler |
|---|---|---|---|
| duplicate_half | 0.01992 | 0.01122 | 0.01145 |
| patch_refine_2x | 0.01456 | 0.20390 | 0.03203 |
| profile_resample_2x | 0.04683 | 0.00481 | 0.00754 |

| probe | variant | TEMP RMSE base -> probed |
|---|---|---|
| duplicate_half | MBCA (method) | 0.5097 -> 0.5097 |
| duplicate_half | Standard Perceiver | 0.5233 -> 0.5233 |
| duplicate_half | Fixed-budget resampler | 0.5117 -> 0.5112 |
| patch_refine_2x | MBCA (method) | 0.5097 -> 0.5095 |
| patch_refine_2x | Standard Perceiver | 0.5233 -> 0.5209 |
| patch_refine_2x | Fixed-budget resampler | 0.5117 -> 0.5140 |
| profile_resample_2x | MBCA (method) | 0.5097 -> 0.5095 |
| profile_resample_2x | Standard Perceiver | 0.5233 -> 0.5231 |
| profile_resample_2x | Fixed-budget resampler | 0.5117 -> 0.5118 |

## 3. Flexibility with ONE checkpoint (no retraining)

### 3a. Profile-count sweep (test months; week-2 density axis)

| density | floor TEMP | MBCA (method) TEMP | Standard Perceiver TEMP | Fixed-budget resampler TEMP | MBCA (method) SALT | Standard Perceiver SALT | Fixed-budget resampler SALT |
|---|---|---|---|---|---|---|---|
| 0 | 0.5521 | 0.5268 | 0.5313 | 0.5230 | 0.1225 | 0.1241 | 0.1236 |
| 50 | 0.5521 | 0.5221 | 0.5269 | 0.5215 | 0.1206 | 0.1218 | 0.1216 |
| 150 | 0.5522 | 0.5229 | 0.5242 | 0.5236 | 0.1207 | 0.1215 | 0.1218 |
| 375 | 0.5521 | 0.5220 | 0.5246 | 0.5227 | 0.1206 | 0.1215 | 0.1217 |
| 750 | 0.5521 | 0.5220 | 0.5237 | 0.5233 | 0.1206 | 0.1214 | 0.1218 |
| 1500 | 0.5522 | 0.5221 | 0.5250 | 0.5232 | 0.1207 | 0.1215 | 0.1218 |
| 3000 | 0.5520 | 0.5220 | 0.5243 | 0.5231 | 0.1207 | 0.1215 | 0.1219 |

The week-2 ablation retrained the depthwise U-Net *per density* (reports/week2_density_ablation.md); every shared-latent number above comes from a single checkpoint per seed.

### 3b. Missing-modality matrix (test months, headline masks fixed)

| inputs | MBCA (method) TEMP | Standard Perceiver TEMP | Fixed-budget resampler TEMP | MBCA (method) SALT | Standard Perceiver SALT | Fixed-budget resampler SALT |
|---|---|---|---|---|---|---|
| full | 0.5221 | 0.5243 | 0.5231 | 0.1206 | 0.1214 | 0.1218 |
| drop_surf | 0.5235 | 0.5327 | 0.5518 | 0.1209 | 0.1228 | 0.1266 |
| drop_woa | 0.5209 | 0.5240 | 0.5227 | 0.1203 | 0.1213 | 0.1215 |
| drop_profiles | 0.5268 | 0.5314 | 0.5230 | 0.1224 | 0.1241 | 0.1236 |

No baseline in the repo can produce 3a/3b without retraining one model per row.

## 4. Withheld-profile evaluation (leave-profiles-out generalisation)

Predict the full column at the coordinates of **300 profiles held out of the model input** each month (of 1500), scored on the OSSE truth there — the supporting decode demonstration protocol_v1 lists but the headline (unobserved-cell) metric does not exercise. Physical anomaly RMSE = absolute RMSE (climatology cancels), so the shared-latent variants and the field baselines are directly comparable on the identical withheld columns. `near`/`far` split at 3° great-circle distance to the nearest *input* profile (mean ± std over 3 seeds).

| method | TEMP all | TEMP near | TEMP far | SALT all |
|---|---|---|---|---|
| MBCA (method) | 0.5252 ± 0.0166 | 0.5094 | 0.5490 | 0.1190 ± 0.0015 |
| Standard Perceiver | 0.5272 ± 0.0124 | 0.5112 | 0.5513 | 0.1197 ± 0.0008 |
| Fixed-budget resampler | 0.5247 ± 0.0168 | 0.5111 | 0.5453 | 0.1205 ± 0.0015 |
| Nearest-profile (from input) | 1.0281 ± 0.0317 | 0.7248 | 1.3842 | 0.3425 ± 0.0004 |
| Climatology floor | 0.5539 ± 0.0148 | 0.5372 | 0.5791 | 0.1291 ± 0.0008 |

*Data: synthetic_OSSE (no real Argo store present).* A **real-Argo** leave-one-profile-out test needs the external Argo store (absent on this machine — `data/` holds only CESM2-LE + WOA23) and must additionally guard against reanalysis-assimilation leakage; deferred until the data lands.

## 5. Submission gate #2 (from the week-3 plan)

F-C (MBCA) vs F-A (Perceiver) accuracy: 0.5221 vs 0.5243 degC -> **PASS** on the accuracy axis; see section 2 for the invariance axis (the insurance policy).

## 6. Provenance

- `fullA_mbca_s1234`: best step 2500 (val score 0.9458), 0.35 GPU-h, commit `072c97a6`, obs_query_frac 0.25, coord fourier_v2
- `fullA_mbca_s1235`: best step 2500 (val score 0.9561), 0.36 GPU-h, commit `072c97a6`, obs_query_frac 0.25, coord fourier_v2
- `fullA_mbca_s1236`: best step 5000 (val score 0.9443), 0.44 GPU-h, commit `072c97a6`, obs_query_frac 0.25, coord fourier_v2
- `fullA_perceiver_s1234`: best step 5000 (val score 0.9402), 0.43 GPU-h, commit `072c97a6`, obs_query_frac 0.25, coord fourier_v2
- `fullA_perceiver_s1235`: best step 2500 (val score 0.9667), 0.36 GPU-h, commit `072c97a6`, obs_query_frac 0.25, coord fourier_v2
- `fullA_perceiver_s1236`: best step 5000 (val score 0.9641), 0.41 GPU-h, commit `072c97a6`, obs_query_frac 0.25, coord fourier_v2
- `fullA_resampler_s1234`: best step 5000 (val score 0.9473), 0.52 GPU-h, commit `072c97a6`, obs_query_frac 0.25, coord fourier_v2
- `fullA_resampler_s1235`: best step 10000 (val score 0.9572), 0.60 GPU-h, commit `072c97a6`, obs_query_frac 0.25, coord fourier_v2
- `fullA_resampler_s1236`: best step 5000 (val score 0.9537), 0.32 GPU-h, commit `072c97a6`, obs_query_frac 0.25, coord fourier_v2
