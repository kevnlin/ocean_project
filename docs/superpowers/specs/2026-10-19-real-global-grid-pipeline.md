# Real global Argo + satellite pipeline for the gridded model

Date: 2026-10-19. Branch: `ka-dev` (worktree `/DATA2/linke/ocean_project_ka_dev`).

## Goal

Run the gridded state model (`grid_kriging_ocean`, runner `101_grid_ka_forecast.py`) on real
Argo profiles and real satellite fields with the same inputs, the same scene keys and the same
scoring as the CESM2 runs. One model can then be trained on CESM2 and used on real data: zero
shot, or fine-tuned. That transfer is the evidence the paper needs for "pretrained on
simulations, works on observations".

## Data

| Part | Source | Use |
|---|---|---|
| Profiles | `data/argo_cohort/global_global.nc` (Argo GDAC, 20 levels to 985 m) | inputs and targets |
| Anomalies | WOA23 at the profile's own position (`global_anomx.npz`, the "exact" anomaly) | target space |
| Satellites | `data/real_obs_1deg.zarr`: monthly OISST, OISSS L4, CMEMS DUACS SLA, 1 degree, 2016-01 to 2023-12 | state inputs and query features |

Years 2016-2023 only: the satellite store covers these. 2024 has no 1-degree store yet. 2025 is
the sealed holdout of the Real-Data v1 registration and is never read.

Splits (by year, as the original real-data `OceanScenes`): train 2016-2020, validation 2021,
development 2022-2023. Floats: the cohort's stored `float_split` (inputs `cohort_float`,
queries `heldout_float`, disjoint WMO numbers, never redrawn). Normalisation: per-level mean and
spread of the training years. QC: a value more than 10 standard deviations from the mean is
treated as missing, as in `OceanScenes`.

## Inputs, built as for CESM2

* Dense surface grids: SST, SSS and SLA anomalies against their 2016-2020 monthly
  climatology, divided by the 2016-2020 spread (the recipe of `88`/`100`). NaN over land.
* Profile features (56 columns): the 44 position/season fields of `position_features`, then SST,
  SSS, SLA z-scores with validity flags (columns 44-49) and the absolute SST (column 50), sampled
  bilinearly at the profile from its own month's grids. Same layout as the CESM2 caches.
* Lagged features for leads 1, 3, 6: columns 44-50 from the input month (target month minus the
  lead), so a forecast never reads the target month's satellites.

Known differences from CESM2: real SSH is the total sea-level anomaly (DUACS), the CESM2 stand-in
is steric height; real surface fields have errors and gaps; real profile counts vary by month
(about 9,000 inputs and 2,200 held-out profiles); profile times are real days, not mid-month.

## Fixed baselines

The OI of `44_synth_argo_oi.py`, tuned the same way on the 2021 validation months (per variable
and depth band, the same grid and edge rule), scored on validation and development:
`outputs/audit/real/fixed_baselines_real_global/{summary,tuning}.json`. Climatology and nearest
profile are scored with it. The runner scores the persisted OI of the input month at every lead.

## Code

* `ocean_tokenizer/real_global.py`: splits, the 2016-2023 subset of the cohort, normalisation,
  fingerprints.
* `experiments/real_data/105_real_surface.py`: the three caches above.
* `experiments/real_data/106_real_oi_baseline.py`: the fixed baselines.
* `ocean_tokenizer/real_forecast_scenes.py`: `RealGridForecastScenes`, the same scene keys as
  `GridForecastScenes`, with variable monthly counts and real days.
* Runner `101`: `--dataset real_global`, and `--init-from` to load model weights trained on
  another dataset (weights only; the data checks of `--load-checkpoint` do not apply).

## Experiments after the pipeline

1. Scratch: the gridded model trained on real 2016-2020, 3 seeds.
2. Zero shot: the CESM2 70k `gj_flow` checkpoints scored on real data with no update.
3. Fine-tune: from the CESM2 checkpoints, a fixed short schedule, 3 seeds.

Selection on 2021 only. The 2022-2023 development years are scored once, after the settings are
fixed.

## Tests

Unit tests on small fake cohorts: the subset keeps every per-profile array aligned; features
sampled from the grids equal direct sampling; lagged features come from the input month; scenes
have the keys of `GridForecastScenes`; a month's inputs and queries are disjoint floats; the
training draw keeps both months inside the training years. A GPU smoke run of runner 101 on real
data before any campaign.
