# A third simulator: GLORYS12 1/12-degree reanalysis, with a sealed test (2026-10-21)

Date: 2026-10-21. Branch: `ka-dev6`. Written before any file of the test years (2019-2020) was
downloaded; the only GLORYS month on disk at that time was 1997-01, downloaded to measure the
throughput (22 s, 1.29 GB on disk for thetao + so + zos at 36 levels, one global month).

## Why

The paper's simulation results come from two free-running climate models (CESM2 at 1 degree, iHESP
CESM1.3 at 0.1 degree). A reanalysis is a different kind of truth: an ocean model constrained by the
real observations of its years, with the real ocean's events in it. GLORYS12 (Mercator Ocean,
NEMO at 1/12 degree, assimilating satellite altimetry, SST, sea ice and in-situ profiles) is the
reanalysis ARMOR3D and the real Argo pipeline are usually compared with. The question is the same
as for iHESP: does the ordering of the CESM2 test hold on it.

## Data

* Product `GLOBAL_MULTIYEAR_PHY_001_030`, dataset `cmems_mod_glo_phy_my_0.083deg_P1M-m` (monthly
  means), variables `thetao` (potential temperature, degC), `so` (practical salinity, psu) and
  `zos` (sea surface height above geoid, m), downloaded with `copernicusmarine subset` at depths
  0-1100 m (the first 36 of the 50 levels, 0.49-1062.4 m), one global month per file. The grid is
  regular: 2041 latitudes (-80 to 90, step 1/12) by 4320 longitudes (-180 to 179.917, step 1/12).
  Values are int16 with a scale and an offset; the fill is -32767.
* Truth at the cohort's 20 levels (5, 15, 25, 35, 45, 55, 65, 85, 105, 125, 145, 165.1, 186.3,
  222.6, 267.7, 326.9, 408.8, 527.7, 707.6, 984.7 m): GLORYS levels are not these, so each 1/12-degree
  column is interpolated linearly in depth between its two bracketing GLORYS levels (every cohort
  level has two: 5 m lies between 3.82 and 5.08 m, 984.7 m between 902.3 and 1062.4 m). A cohort
  level is missing where either bracketing GLORYS level is missing, which is the case at and below
  the sea floor (the mask is the bathymetry; it does not change with the month, and every month is
  checked against the first). No extrapolation.
* A profile is a model column, as in the iHESP cohort: the 20 interpolated values of one 1/12-degree
  cell, at a uniform random point inside the cell.
* Surface images (86's recipe): the 1-degree box means of the 1/12-degree 5 m temperature and 5 m
  salinity (the cohort's first level), and the TEOS-10 steric height (reference 990 dbar) of the
  box-mean 20-level column to 985 m, computed only where the box-mean column is complete. Box
  means are area-weighted (cos latitude) over the cells of the box with a value, NaN where none.
  Boxes south of 80 S have no GLORYS cells and are NaN. The 1-degree box mean of `zos` is kept in
  the same files for a later variant (an altimeter-like field); no model of this spec reads it.

### What is kept per month, and what is deleted

Each raw month (1.29 GB) is deleted as soon as its products are written:

* `data/glorys/boxes/boxes_YYYYMM.npz`: the 20-level 1-degree box means of T and S
  (`(20, 64800)` float32 each, 10 MB) and the four surface box fields SST, SSS, steric height,
  zos (`(4, 64800)` float32);
* `data/glorys/columns/columns_YYYYMM.npz`: the 20-level T and S columns at the union of the
  cells drawn for every cohort month (training, selection and test) that shares this calendar
  month (24 years x 7,600 = up to 182,400 cells, 29 MB). These union files are how 85's
  climatology ("the training years' monthly mean of each drawn cell") is computed and later
  re-checked without the raw files: a cell's climatology is the mean over the 20 training years
  of its value in these files.

Persistent total: 288 months x (10 + 1 + 29) MB = about 11.5 GB, plus the cohort files (about
1.8 GB each). Raw files in flight: at most the number of download workers (2-3), under 5 GB.

## Calendar

The runner (`101`), the scene classes (`LongRecordScenes`: the training window is the last
`train_years` years ending 2003-12; `--val-years 2004 2005`), the cohort loader (which relabels
`year_split` from the year with the table train 2000-2003 / validation 2004 / development 2005) and
the OI selection (`44`: tuned on 2004, scored on 2005) all count month index 0 as 2000-01. The
GLORYS years are therefore RELABELLED by -13 years so that the cohort's calendar is the one the
code expects. The stored `year` and `month_index` columns are the cohort calendar; the GLORYS
calendar is stored next to them as `glorys_year` and `glorys_month`, and in the attributes
(`calendar_offset_years = 13`).

| GLORYS months | Cohort years | Month index | Use |
|---|---|---|---|
| 1997-01 .. 2016-12 | 1984-2003 | -192 .. 47 | training (`year_split = train`; `--train-years 20`) |
| 2017-01 .. 2017-12 | 2004 | 48 .. 59 | validation: checkpoint choice, OI tuning |
| 2018-01 .. 2018-12 | 2005 | 60 .. 71 | development: spread calibration, damped-persistence factors |
| **2019-01 .. 2020-12** | **2006-2007** | **72 .. 95** | **sealed test**, scored once |

A forecast at lead 6 scored in 2019-01 (month 72) reads inputs of 2018-07 (month 66), which is a
development month of the training cohort, as in the CESM2 sealed test (2006-2007, inputs from
2005-07). The test lies after the training years, as for CESM2.

Two normalisations, as for the 40-year CESM2 arms: the cohort's stored climatology (`CLIM_POS`,
`CLIM_CELL`) is the 20-year monthly mean 1997-2016 of each drawn cell; the runner's per-depth
z-scores of the anomalies (`ArgoNorm.fit` on the loader's "train" rows) cover the relabelled
2000-2003 = GLORYS 2013-2016, because the loader's split table is fixed at 2000-2003 (checked by
`44` and by the runner). The surface caches (86's recipe) use the 20 training years 1997-2016 for
their box climatology and z-scores.

## Cohorts

### `glorys12_uniform` (training and selection)

85's recipe on the 1/12-degree grid, 264 months (1997-01 .. 2018-12):

* ocean cells: cells with a value at the first GLORYS level; the cell area is proportional to
  cos(latitude) (regular grid);
* 7,600 cells a month, drawn without replacement with probability proportional to area;
  20 % of them (1,520) are queries (`heldout_float`), the rest inputs (`cohort_float`);
* position: the cell centre moved by a uniform offset in (-0.45, 0.45) cell widths (1/12 degree)
  north and east, as 85's `IN_CELL`; longitude stored in 0-360;
* one generator `numpy.random.default_rng(20261021)` consumed in month order (-192, -191, ..., 71),
  three draws a month (cells, queries, offsets), as 85 does;
* target: the column's anomaly from the 1997-2016 monthly climatology of that cell; levels at or
  below the sea floor are missing; `TEMP_ERR`/`SALT_ERR` are NaN (noise free);
* columns and attributes as 85 writes them (`grid_y`/`grid_x` the 1-degree cell, `model_j`/`model_i`
  the 1/12-degree row and column, `wmo` `S<month>...`), plus `glorys_year`, `glorys_month`.

Checks before the file is written: 7,600 profiles and 1,520 queries in every month; no value is a
fill; the finite fraction by level is printed; the climatology recomputed from the union files at
4,000 random rows equals the stored one.

### `glorys12_test` (sealed)

The rows of `glorys12_uniform` unchanged, plus the 24 months 2019-01 .. 2020-12 (month index 72-95)
drawn with the same recipe from one generator per month, `default_rng([20261021, month_index + 10000])`
(138's convention), with targets against the same 1997-2016 climatology of each drawn cell, and the
same `TEMP_ERR`/`SALT_ERR`, `wmo` `T<month>...`, `year_split = unassigned`. The positions of the
test months are drawn (they depend on the seed and the ocean mask only) before the training pass so
that the climatology at their cells is read from the training months into the union files; no
test-year value is read before `157_glorys_test_cohort.py` runs, and that script prints only counts
and the training-row checks (training rows unchanged, climatology reproduced, no fill), never a
statistic of a test value.

## Caches (outputs/glorys_20261021/<cohort>/)

`156_glorys_surface_caches.py <cohort>` writes, in the layouts of 86 and 136:

* `satellites.npz`: the 56 profile features (44 positional + value/presence of SST, SSS, steric
  height z-scores + the absolute SST), `normalization_scope = "training years only"`;
* `surface_grids.npz`: `[M, 3, 180, 360]` float16 z-scores of the 1-degree boxes for every cohort
  month;
* `lagged_surface.npz`: `[P, 3, 7]` the surface columns of month m - lead at each profile's box,
  leads 1, 3, 6.

The z-scores use the monthly climatology, mean and spread of the box fields over the 240 training
months. For the test cohort the training rows must give the training cohort's features back
exactly (139's check).

OI: `44 --cohort glorys12_uniform` (tuned on 2004 = GLORYS 2017, scored on 2005 = GLORYS 2018)
writes `outputs/audit/synthetic/fixed_baselines_glorys12_uniform/summary.json`; `44 --cohort
glorys12_test` writes the test cohort's own file, which the scorer reads (its selection equals the
training cohort's, as the iHESP amendment found; it touches no test row).

## Models (seeds 1234, 1235, 1236; 100,000 steps as the 2026-10-21 campaign; leads 0/1/3/6;
6,080 input profiles a month, 1,024 queries a step, `--train-years 20`, `--val-years 2004 2005`)

| Arm | Options |
|---|---|
| `g_ka` KA-Ocean | `--kriging ka_flow` (joint background) |
| `g_armor` ARMOR3D-style OI | `--background prior --kriging ka_oi` |
| `g_none` Aardvark-style | `--kriging none` |
| `g_softmax` softmax read-out | `--kriging softmax` |

Queue `158_glorys_queue.py` (150's pattern): GPUs 2, 3, 4, two runs each, a launch needs 60 GB
free (those GPUs carry another campaign until about 00:00 Oct 22; the gate makes the queue wait),
finished runs skipped, failed runs resumed from their last checkpoint; a 40-step smoke of one arm
on GPU 7 first. Runs under `outputs/glorys_20261021/runs/`.

Statistical baselines on the test cells: persisted OI (the frozen selection), damped persistence
(factors fitted on 2004-2005 = GLORYS 2017-2018 from `g_ka_s1234`'s saved predictions, as 140), the
climatology.

## Protocol

1. Checkpoint of every run: the lowest mean standardised RMSE over leads on the fixed 8,000-cell
   draw of 2004-2005 (GLORYS 2017-2018). Spread calibration: `116` on the same years.
2. `159_sealed_glorys_test.py` (140's pattern on `109`'s scorer): `--check` reproduces a run's own
   2004-2005 scores on the test cohort; `--register` records every checkpoint, calibration and data
   file by SHA-256; `--score` scores every registered checkpoint once on every held-out cell of
   2019-2020 (cohort 2006-2007) at leads 0, 1, 3, 6 and writes `results.json`, never twice.
3. Reported: RMSE by lead, variable and depth band; CRPS and 95 % coverage; mean and standard
   deviation over seeds.
4. Primary comparison, fixed now: KA-Ocean against the best end-to-end baseline (ARMOR3D-style,
   Aardvark-style) in lead-0 temperature and salinity RMSE. Every result is reported whatever it is.

## Scripts

| | |
|---|---|
| `src/ocean_tokenizer/glorys_sources.py` | grid, calendar mapping, depth interpolation, box means, the draws |
| `experiments/data/155_glorys_months.py` | download a month, write its boxes and union columns, delete the raw file |
| `experiments/synthetic/155_glorys_cohort.py` | `glorys12_uniform.nc` from the per-month products |
| `experiments/synthetic/156_glorys_surface_caches.py` | satellites, surface grids, lagged surface of a cohort |
| `experiments/synthetic/157_glorys_test_cohort.py` | `glorys12_test.nc` |
| `experiments/synthetic/158_glorys_queue.py` | the training queue |
| `experiments/synthetic/159_sealed_glorys_test.py` | the sealed scorer |
