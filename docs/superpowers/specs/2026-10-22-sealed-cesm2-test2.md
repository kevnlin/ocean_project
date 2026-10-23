# Sealed CESM2 test v2 (2008-2010)

Date: 2026-10-22. Branch: `ka-dev6`. Written before any file of 2008-2010 existed.

## Why

The paper's CESM2 sealed test is 2006-2007 (`cesm2_long_test`, spec 2026-10-19). It was opened on
2026-10-20 and scored, so it can no longer decide anything about the main model. The Wang-feedback
campaign (`150_wang_campaign.py`, 100k steps, selection on 2004-2005) produced a candidate for the
main model that was trained on a 154-year window: `c_y154` (cohort `cesm2_vlong_uniform`, 1850-2003)
against the comparison basis `c_flow100` (cohort `cesm2_long_uniform`, 1964-2003, the paper's settings
at 100k steps). The 2026-10-21 spec says any change of the main method needs a new sealed CESM2 period,
2008-2010, built only when that decision is taken. This is that period.

No run has read the raw member after 2007-12: every CESM2 store on disk ends at 2005-12
(`cesm2_le_cohort_levels.zarr`, `_1850_2005.zarr`, `cesm2_le_full_standard.zarr`) or 2007-12
(`_1964_2007.zarr`); `53` was only ever run with `--last-year` 2005 or 2007. The August-era scripts
13-35 name 2008-2010 as validation years of `processed/cesm2_standard.zarr`, a single-member
placeholder source that is not on disk and is not the raw member (`data/cesm2_le_raw`,
SingleBicycle/cesm2-le-ocean-sample) every cohort here is drawn from.

## Primary comparison

`c_y154` (three seeds) against `c_flow100` (three seeds): temperature and salinity RMSE at lead 0 and
lead 6 on every held-out cell of 2008-01 .. 2010-12, each model scored on the test cohort of its own
training cohort. Leads 1 and 3, CRPS, coverage, the by-band and by-level tables, damped persistence,
persisted OI and climatology are reported alongside. Every number is reported whatever it is: if the
154-year model loses at either lead or either variable, that is the result, and the paper's main
model stays the 40-year one.

## What is built (`166`, `167`, `168`)

| File | Content |
|---|---|
| `data/cesm2_long/cesm2_le_cohort_levels_2000_2010.zarr` | `53 --first-year 2000 --last-year 2010`: the raw member at the cohort's 20 levels for 2000-01 .. 2010-12 (132 months; raw months after 2010-12 are not read). Its 2000-2007 equal the 1964-2007 store value for value, its 2000-2005 the two training stores. |
| `data/synthetic_argo/cesm2_long_test2.nc` | every row of `cesm2_long_uniform.nc` unchanged and first (3,830,400), then 2006-01 .. 2010-12 (60 months, 456,000 profiles) drawn as `87` draws a month: generator `[20261017, month_index + 10000]`, 7,600 area-uniform ocean positions, 20 % held-out floats, bilinear TEMP/SALT from the new store, anomalies against the same 2000-2003 climatology. Year split "unassigned". The 2006-2007 months equal `cesm2_long_test`'s rows exactly (same generator, same recipe, store months equal), and are kept so that a forecast into 2008-01 .. 2008-06 reads inputs from 2007-07 .. 2007-12. |
| `data/synthetic_argo/cesm2_vlong_test2.nc` | the same with every row of `cesm2_vlong_uniform.nc` (14,227,200) first. |
| `outputs/cesm2_test2_20261022/<cohort>/{satellites,surface_grids,lagged_surface}.npz` | the caches of `88`, `100`, `93`: rows and months of the training cohort copied from its registered caches value for value; 2006-2010 by the same recipes from the new store (their 2006-2007 values are checked against `outputs/cesm2_test_20261019`). |
| `outputs/audit/synthetic/fixed_baselines_<cohort>/summary.json` | `44 --cohort <cohort>`: OI tuned on 2004, scored on 2005, which are the training cohort's rows; the selection must equal the training cohort's exactly. |
| `outputs/cesm2_test2_20261022/damped_persistence.json` | `121`'s factors (least squares per lead, variable, depth band) from the persisted-OI and target arrays `c_flow100_s1234` saved on 2004-2005; `c_y154_s1234`'s arrays must give the same factors (same cells, same OI). |
| `outputs/cesm2_test2_20261022/sealed/registration.json`, `results.json` | `168`: written once each. |

The normalisation, climatology, OI selection and scale tables are the 2000-2003 / 2004 fits used in
training; the new months only add rows. A checkpoint is scored on the test cohort of its training
cohort, found by the checkpoint's `data.cohort_fingerprint` (`c_flow100` -> `cesm2_long_test2`,
`c_y154` -> `cesm2_vlong_test2`); the mapping is recorded in the registration.

## Checks (none of them reads a value of 2008-2010 as a score)

* Store: months 2000-01 .. 2007-12 equal `cesm2_le_cohort_levels_1964_2007.zarr` value for value;
  2000-2005 equal `cesm2_le_cohort_levels.zarr` and `_1850_2005.zarr`; grid, levels and MASK equal.
* Cohorts: the climatology rebuilt from the new store reproduces `CLIM_POS` of the training cohort at
  20,000 rows; the 2006-2007 draws equal `cesm2_long_test`'s rows value for value; rows stay sorted
  by month, float ids unique; the training rows are first and unchanged.
* Caches: the 2000-2003 normalisation equals the training cache's; the grid, feature and lag recipes
  on the new store reproduce the training caches on 2005-07 .. 2005-12 and `cesm2_test_20261019`'s
  on 2006-2007; the copied rows keep their registered values exactly.
* OI: `44` on each test cohort gives the training cohort's selection, per-month scores and summary
  scores exactly.
* `168 --check grid:PATH`: `c_flow100_s1234` and `c_y154_s1234` reproduce their runs' own 2004-2005
  forecast scores at every lead on their test cohorts (relative difference about 1e-5, bf16).

## Rules

1. The test is opened once. Until then no script reads 2008-2010 as a target: no training, no
   selection, no "quick look". `167` draws the rows and `168 --check` scores 2004-2005 only.
2. `168 --register grid:CHECKPOINT ...` records every checkpoint, its calibration if `116` fitted one,
   every data file of both test cohorts and the damped-persistence factors by sha256, and can only
   be written once. Register every model the comparison reports in one call: the three seeds of
   `c_flow100` and of `c_y154` (and any other 100k arm Zihao wants on this period).
3. `168 --score` verifies every file, scores 2008-2010 at leads 0, 1, 3, 6 on every held-out cell with
   the persisted OI and the climatology of the same cells, adds damped persistence on the same cells,
   and never overwrites `results.json`. It runs once, with Zihao's go-ahead.
4. Report the sealed results as they come out, including any loss to a baseline or to `c_flow100`.
