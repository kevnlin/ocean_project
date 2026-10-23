# Sealed CESM2 test (2006-2007)

Date: 2026-10-19. Branch: `ka-dev`.

## Why

Every CESM2 model so far (the gridded model of `103`, the OI-anchored local transformer of `95`) was
trained on `cesm2_long_uniform` (1964-2005) and its checkpoint chosen on 2004-2005, so its 2004-2005
scores are validation scores. The paper needs scores on months no training or selection step saw.

## What was built

| File | Content |
|---|---|
| `data/cesm2_long/cesm2_le_cohort_levels_1964_2007.zarr` | `53` with `--last-year 2007` (new file; its 2000-2005 equal the reference store value for value) |
| `data/synthetic_argo/cesm2_long_test.nc` | `cesm2_long_uniform.nc` unchanged (3,830,400 profiles, same order), then 2006-01 to 2007-12 (182,400 profiles) drawn as `87` draws a month |
| `outputs/cesm2_test_20261019/{satellites,surface_grids,lagged_surface}.npz` | the caches of `88`, `100`, `93`: old rows copied, new rows by the same recipes |
| `outputs/audit/synthetic/fixed_baselines_cesm2_long_test/summary.json` | `44 --cohort cesm2_long_test`: OI tuned on 2004, identical to the training cohort's selection |

Normalisation, climatology, OI selection and scale tables are the 2000-2003 / 2004 fits used in
training; the new months only add rows.

## Checks passed (none of them reads 2006-2007)

* The 1964-2007 store equals the training store at the checked months.
* The rebuilt 2000-2003 climatology reproduces the cohort's `CLIM_POS` exactly.
* The feature, grid and lag recipes reproduce the existing caches on 2005 months.
* The OI selection on the new cohort equals the training cohort's; its 2004 and 2005 scores match.
* `109 --check`: on the new cohort, a gridded checkpoint (`grid_ka_20261019/gj_flow_s1234`) and a
  reference checkpoint (`forecast_20261019/f_y40_s1234`) reproduce their runs' own 2004-2005 forecast
  scores at every lead (relative difference 2.4e-5 and 0).

## Rules

1. The test is opened once, after the final models are frozen. Until then no script reads 2006-2007:
   no training, no selection, no "quick look".
2. `109 --register FAMILY:CHECKPOINT ...` records every checkpoint and data file (sha256) and can
   only be written once. Register every model the paper will report in one call: the method, the
   ablations, the reference models, all seeds.
3. `109 --score` verifies every file, scores 2006-2007 at leads 0, 1, 3, 6 on every cell with the
   persisted OI of the same cells, and never overwrites `results.json`.
4. Report the sealed results as they come out, including any loss to a baseline.
