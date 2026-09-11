# protocol_v1 — Frozen Main-Experiment Protocol

*Frozen 2026-07-17, before the Monday group meeting. Machine-readable twin:
[`configs/protocol_v1.yaml`](../configs/protocol_v1.yaml). Any deviation must be
declared and keeps a run out of the headline table.*

## Task

**Contemporaneous ocean-state reconstruction** (not forecasting): given the
heterogeneous observations of one calendar month, predict **temperature and
salinity anomalies** at unobserved coordinates:

O_t → ΔT̂(lat, lon, z, t), ΔŜ(lat, lon, z, t)

Absolute fields are recovered by adding the train-only monthly climatology:
T̂ = C_train^T + ΔT̂,  Ŝ = C_train^S + ΔŜ.

Out of scope under this protocol (do not claim, do not evaluate): forecasting /
rollout, operational data assimilation, super-resolution, "foundation model".

## Data

| | |
|---|---|
| Ground truth | CESM2-LE, **one member**, regridded to regular 1° (180×360). Member id was not recorded at regrid time and is unrecoverable — reported as "a single CESM2-LE member (historical + SSP370)" and listed as a limitation. |
| Prior | WOA23 monthly climatology, interpolated onto the analysis grid |
| Ocean mask | `MASK == 1` of the ground-truth store |
| Longitude | 0–360 convention (WOA rolled onto it) |

## Depth grid (primary)

**20 native CESM2-LE levels, 5 → 984.7 m** (indices
`[0,1,2,3,4,5,6,8,10,12,14,16,18,21,24,27,30,33,36,39]`). Native levels keep the
target lossless; WOA23 is interpolated onto them.

* The primary task has **no depth below 1000 m**. "Below 1000 m" results must
  never be reported from it (last week's mislabel, corrected).
* Depth bands for the primary task: **0–100 m / 100–300 m / 300 m–max**, plus
  per-level RMSE.
* The extended 23-level grid to 1400 m is a separate, clearly-labelled secondary
  protocol ([layered_depth_eval.md](layered_depth_eval.md)); 20- and 23-level
  numbers never share a table.

## Split (contiguous years; frozen)

| Split | Years | Months |
|---|---|---|
| Train | 1985–2007 | 276 |
| Validation | 2008–2010 | 36 |
| Test | 2011–2014 | 12 snapshots (below) |

Test months (drawn once with seed 1234, now pinned explicitly):
**2011-02, 2011-04, 2011-05, 2011-07, 2011-11, 2012-03, 2012-09, 2012-10,
2013-01, 2013-10, 2013-12, 2014-09.**

Rules:
1. All model selection (checkpoints, hyperparameters, architecture choices) uses
   the **validation** months only. The test months are touched once per final
   model.
2. Validation is the *end* of the train era (closest distribution to test
   without touching it).
3. Earlier week-1/2 runs trained on all 312 months of 1985–2010 with no
   validation split; they remain metric-comparable but are superseded by
   protocol_v1 runs.

## Inputs (primary experiment — deliberately narrow)

WOA T/S prior · dense SST/SSS · sparse synthetic T/S profiles · coordinates,
masks, time, ocean mask.
**SSH/ADT deferred** until the core model is stable *and* matched baselines
receive it too. **Point observations optional** for the MVP.

## Synthetic profiles

1500 uniform-random ocean columns per month (~3.5 % coverage), exact noise-free
model columns (stated OSSE limitation). Headline runs repeat over profile seeds
**{1234, 1235, 1236}** (mean ± std); the month split never varies with seed.
The observed-column mask marks every column containing a profile; its complement
(`unobs_mask`) is broadcast over depth.

## Target, climatology, normalization

* **Anomaly target**: `a(t) = field(t) − clim_train[month(t)]`.
* **Climatology from the 276 train months only** — validation months are now
  excluded (stricter than week-1, which used all 312; expected shift is small
  but the Task-1 rerun re-baselines all reported numbers under protocol_v1).
* Anomaly-space z-scores per variable/depth, statistics from train months only.

## Metric

**Headline: unobserved-only anomaly RMSE** — NaN-aware RMSE over ocean cells
with no profile column that month (observed columns excluded at *all* levels),
pooled over the 12 test months; per variable, full-column + per band.

* **Floor**: train-only CESM2 climatology (= zero anomaly). Skill
  `= 1 − RMSE/RMSE_floor`. The WOA prior is reported separately and is *not*
  the skill reference (bias-dominated).
* **Weighting**: primary numbers are valid-cell-weighted (matches everything
  reported to date). Area×thickness-weighted RMSE (cos-lat × layer thickness)
  is added as a secondary column when implemented — never silently substituted.
* Prohibited: scoring observed columns; ">1000 m" claims on the 20-level task;
  mixing depth grids in one table.

## Evaluation queries

Main evaluation on the fixed dense target grid (all unobserved ocean cells ×
20 levels). Withheld-profile and arbitrary-point queries are supporting
demonstrations of the same decoder interface, not the headline.
