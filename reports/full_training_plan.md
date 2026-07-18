# Plan — First Full Training Run (post-gate)

*The full-scale experiment starts only after (a) the Task-6 decisive invariance
suite passes on the MBCA module and (b) the tiny-overfit gate (Task 7) is green.
Status: Task-7 gate green (see `outputs/cache/poc_ocean.json`); Task-6 suite is
[coworker], landing on the `support_mass` seam in `token_api.py`.*

## What is trained

Three models, identical everything except the fusion rule
(`ocean_tokenizer.fusion`):

| run | variant | why |
|---|---|---|
| F-A | StandardPerceiver | the architecture control (Perceiver IO prior art) |
| F-B | FixedBudgetResampler | does a fixed token budget alone fix multiplicity? |
| F-C | MBCA | the method |

Config: d_model 128, 128 latents, 4 heads, 4 self-blocks (frozen after POC);
inputs `profiles_woa_surf` (no SSH, no points — protocol_v1).

## Data & schedule

- **protocol_v1 split**: 276 train / 36 val / 12 pinned test months.
- Per step: one random train month; 8,192 random unobserved-cell queries;
  anomaly z-space MSE. (POC showed ~1 min / 400 steps on one A100 at 17k
  tokens/month → a 100k-step run is ~4 GPU-h; budget 3 runs x 3 seeds on
  GPUs 6-7 over ~2 days.)
- Optimizer: Adam 3e-4, cosine to 1e-5, 100k steps, val every 1k steps,
  best-checkpoint on the val score (mean RMSE/floor over TEMP+SALT), early
  stop patience 10 evals.
- Profile-count augmentation **on** (uniform 0-3000 per month) so the
  one-checkpoint count sweep is in-distribution — this is the flexibility
  axis, decide BEFORE launch, do not retrofit.
- Seeds: {1234, 1235, 1236} (profile sampling + init), month split fixed.

## What is reported (in order of importance)

1. Test unobserved-only anomaly RMSE (TEMP/SALT, full + bands) vs
   climatology floor / MLP / depthwise U-Net / certified joint-depth U-Net
   (`audit_*.pt`), multi-seed mean +/- std.
2. Duplication / retokenization / profile-resampling sensitivity of F-A vs
   F-B vs F-C at the final checkpoint (Task-6 protocol, real data).
3. One-checkpoint profile-count sweep (0 -> 3000) + missing-modality matrix
   (drop surf / woa / profiles) — no retraining.
4. Gate for the paper (submission gate #2): F-C >= F-A on (1) and beats it
   decisively on (2)/(3), else the story is re-examined before writing.

## Explicitly not in this run

Forecasting/rollout, SSH, point obs, real Argo (separate later experiment),
super-resolution, uncertainty. Samudra/FuXi comparisons stay related-work.
