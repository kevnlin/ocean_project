# A second simulator: iHESP 0.1-degree, with a sealed test (2026-10-21)

Date: 2026-10-21. Branch: `ka-ihesp` (from `ka-dev4`). Approved by Zihao the same day ("start iHESP now").
Written and committed before any file of the test years was downloaded.

## Why

Every simulation result of the paper comes from one CESM2 member at 1 degree. A reviewer can ask
whether KA-Ocean's gains depend on that model. iHESP CESM1.3 at 0.1 degree is a different model
version, resolution and member: an eddy-resolving ocean whose fronts and eddies the 1-degree
inputs only partly see. The question is whether the ordering of the CESM2 test holds there.

## Data

* Truth: iHESP CESM1.3 historical run, `ne120_t12` (0.1-degree POP ocean), member 002, monthly
  TEMP and SALT at the 20 levels of every cohort (5-985 m), from the Hugging Face dataset
  `SingleBicycle/iHESP-historical-3D-monthly-hr` (`51_ihesp_hr_subset.py`).
* Training and selection cohort: `data/synthetic_argo/ihesp_hr_uniform.nc` as it exists
  (`85_ihesp_argo_cohort.py`, seed 20261009): 7,600 model columns a month at area-weighted random
  ocean cells, 80 % inputs and 20 % queries, 2000-2005, anomalies against the 2000-2003 monthly
  climatology of each cell.
* Surface images: SST, SSS and steric height of the 0.1-degree fields averaged to 1-degree boxes
  (`86_ihesp_surface_cache.py`), z-scored with 2000-2003 statistics; each profile reads its box.

## Splits

| | Years | Use |
|---|---|---|
| Train | 2000-2003 | model training (the 4-year window of CESM2's `c_y4`) |
| Selection | 2004-2005 | checkpoints, spread calibration, damped-persistence factors |
| **Sealed test** | **1998-1999** (24 target months; inputs from 1997-07) | scored once |

2000-2005 were used by earlier experiments on this cohort; 1997-1999 were never downloaded before
this spec. The test lies before the training years: in a simulation the ocean's statistics do not
depend on the direction of time, and every forecast still runs forward within the test years.

## Test cohort

`ihesp_hr_test`: the rows of `ihesp_hr_uniform` unchanged, plus the months 1997-07 ... 1999-12
drawn with 85's recipe from one generator per month, `default_rng([20261009, month_index + 10000])`
(month_index counted from 2000-01, so -30 ... -1), with targets against the same 2000-2003 monthly
climatology of each drawn cell, and surface images by 86's recipe. Before any model reads it, the
builder only checks counts and that the training rows and their climatology are unchanged.

## Models (seeds 1234, 1235, 1236; CESM2's `c_y4` settings: 70,000 steps, leads 0/1/3/6,
6,080 input profiles a month, 1,024 queries a step)

| Name | Options |
|---|---|
| KA-Ocean (A) | `--kriging ka_flow` (joint background) |
| ARMOR3D-style OI | `--background prior --kriging ka_oi` |
| Aardvark-style | `--kriging none` |
| Softmax read-out | `--kriging softmax` |

Statistical baselines on the same cells: persisted OI (the frozen OI selection of
`outputs/audit/synthetic/fixed_baselines_ihesp_hr_uniform`), damped persistence (factors fitted on
2004-2005) and the climatology.

## Amendment before any scoring (2026-10-21, 03:00)

The scenes of the test cohort need an OI file computed on that cohort, as CESM2's test did
(`44 --cohort cesm2_long_test`). `44 --cohort ihesp_hr_test` tuned the OI on the test cohort's 2004
rows, which are the training cohort's: its selection and its 2004-2005 scores equal
`fixed_baselines_ihesp_hr_uniform`'s exactly, so the persisted OI is still the frozen selection. `140`
reads `outputs/audit/synthetic/fixed_baselines_ihesp_hr_test/summary.json`. No test month was scored.

## Protocol

1. Checkpoint of every run: the lowest mean standardised RMSE over leads on the fixed 8,000-cell
   draw of 2004-2005. Spread calibration: `116` on 2004-2005.
2. Every checkpoint, calibration and data file is registered by SHA-256 before the test is scored;
   the test is scored once on every held-out cell of 1998-1999, at leads 0, 1, 3 and 6.
3. Reported: RMSE by lead, variable and depth band; CRPS and 95 % coverage; mean and standard
   deviation over seeds.
4. Primary comparison, fixed now: KA-Ocean against the best end-to-end baseline (ARMOR3D-style,
   Aardvark-style) in lead-0 temperature and salinity RMSE. Every result is reported whatever it is.
