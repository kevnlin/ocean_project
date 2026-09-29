---
license: other
pretty_name: ocean_project data
---

# ocean_project data

Data backup for [kevnlin/ocean_project](https://github.com/kevnlin/ocean_project):
sparse-profile reconstruction of subsurface ocean temperature and salinity.
Everything the training, audit and report scripts read is here, laid out
exactly where the code expects it. Clone the code, download this repository
into its root, and the scripts run unchanged.

```bash
git clone https://github.com/kevnlin/ocean_project && cd ocean_project
hf download klin2323/ocean_project-data --repo-type dataset --local-dir .
# -> data/, outputs/, checkpoints/ land next to src/ and experiments/
```

## Layout

| Path | What it is | Made by |
|---|---|---|
| `data/argo_cohort/global_global.nc` | **Processed real Argo, global.** 1,791,944 QC'd profiles, 2002-2025, on 20 levels (5-985 m) and the 1° 180×360 grid. Raw values, plus `year`, `year_split`, `float_split`, `wmo`, `grid_y/x`, `data_mode`, per-level errors. | `44_build_argo_cohort.py --regions global --grid global --levels protocol --suffix _global` |
| `data/argo_cohort/global_anomcell.npz` | Global anomaly, **WOA23 at the 1° cell centre** (registered target) | `audit_tools.load_cohort(anomaly="cell")`, cached |
| `data/argo_cohort/global_anomx.npz` | Global anomaly, **WOA23 bilinearly interpolated to each profile's position** (the audit's fix) | `audit_tools.load_cohort(anomaly="exact")`, cached |
| `data/argo_cohort/{gulfstream,npac_gyre}*.nc` | Regional cohorts (Gulf Stream, N. Pacific gyre): `_ext` = 23 levels, `_anom` = anomaly, `_ecco*` = ECCO in-situ constraints, `_sim` = CESM2-sampled | `44_build_argo_cohort.py` |
| `data/argo_cohort/manifest.json`, `data/argo/manifest.json`, `data/argo/ar_index_global_prof.txt.gz` | Float selection, file hashes, GDAC index | `42`, `44` |
| `data/synthetic_argo/cesm2_uniform.nc` | **Synthetic Argo for the synthetic audit.** 547,200 profiles (7,600/month × 72) at uniform random ocean positions, CESM2 interpolated to each; both climatologies stored (`CLIM_POS_*`, `CLIM_CELL_*`) | `experiments/synthetic/41_synth_argo_cohort.py` |
| `data/cesm2_le_full_standard.zarr` | CESM2-LE 1° monthly TEMP/SALT/SST/SSS, 2000-2005, 60 levels | `experiments/data/standardize.py` |
| `data/woa23_standard.zarr`, `data/woa23/` | **WOA23 monthly climatology** (the anomaly reference for real data): standardized store and the raw NetCDF | `standardize.py` |
| `data/real_obs_1deg.zarr`, `data/real_obs/` | Satellite SST (OISST), SLA (DUACS), SSS (OISSS), 2016-2023: 1° model-ready store and the native files | `experiments/data/41_download_real_obs.py` |
| `data/reference/en4/`, `data/reference/ecco/` | EN4.2.2 g10 and ECCO V4r4 gridded reference products | `43_download_reference_products.py` |
| `data/godas_gulfstream/`, `data/godas_npac_gyre/` | GODAS regional subsets | `13_download_godas.py` |
| `data/senseiver/` | Inputs for the Senseiver reproduction | `48_senseiver.py` |
| `outputs/audit/global/<arm>/` | Real-data audit runs: `summary_seed*.json` (scores), `history.jsonl` (training curves), `model_seed*.pt` (weights) | `experiments/real_data/62_sanity_train.py` |
| `outputs/audit/synthetic/<arm>/` | Synthetic audit runs, same layout, plus `data_audit.json` | `62_sanity_train.py --region synthetic`, `42_synth_argo_audit.py` |
| `outputs/ckpt/`, `outputs/cache/`, `outputs/argo_*/`, `outputs/godas_*` | Earlier checkpoints, cached evaluation JSON, signed real-Argo artifacts | see `experiments/README.md` |
| `checkpoints/` | Small trained baselines | `src/baselines/` |

**Not included (re-download):** the raw GDAC float files (`data/argo/dac/`,
66 GB) and the raw ECCO in-situ constraint files (`data/reference/ecco_obs/`,
36 GB, ECCO V4r4's own observation files from PO.DAAC). Every processed
cohort built from them is included (`global_global.nc`, `*_ecco*.nc`). To
rebuild the Argo side: `42_download_argo.py --regions global --start-year 2002`,
then `44_build_argo_cohort.py`. W&B offline run directories are also left out:
the same curves are in each run's `history.jsonl`.

## Splits

**Real Argo (global audit).** Two independent splits:

- *By year* (`year_split` is recomputed from `year` by the driver's split
  table): train 2016-2020, validation 2021, test ("development") 2022-2023.
  Validation picks checkpoints and settings; test is only reported.
- *By float* (`float_split`, fixed at cohort build, seed 20260905): about 20 %
  of floats (WMOs) are `heldout_float` and never appear as input in any month.
  Inputs are `cohort_float` profiles; scores are at `heldout_float` profiles
  in the same month. During training, 30 % of the cohort floats in a month are
  drawn as targets each step, so held-out floats are never trained on.

**Synthetic.** Train 2000-2003, validation 2004, test 2005. 20 % of each
month's profiles were marked as queries once (seed 20260926); all runs share
them.

## Preprocessing (real Argo)

1. Argo QC (`44`): D/A profiles use `*_ADJUSTED` with adjusted QC, R profiles
   raw; keep position/time QC 1-2, value QC 1-2 (plus 5, 8 when adjusted);
   pressure must be positive and increasing.
2. Pressure → depth with TEOS-10 `gsw.z_from_p`; linear interpolation onto the
   20 protocol levels, no extrapolation; profiles need ≥ 8 valid levels.
3. Each profile is assigned to its 1° grid cell (`grid_y`, `grid_x`).

## Target

Anomaly against the **WOA23 monthly climatology**, in the profile's month:

- `cell` (registered): `x − C(cell centre)` — `global_anomcell.npz`
- `exact` (audit fix, `--ablation anomaly_exact`): `x − C(profile position)`,
  bilinear in lat/lon — `global_anomx.npz`

Synthetic data uses CESM2's own train-year 1° climatology instead of WOA23.

## Normalization

`ArgoNorm`: per channel and per level, mean and standard deviation of the
anomaly over the **training years only**, computed at load time
(`argo_obs.ArgoNorm.fit(c, "train")`). No file is stored for it; it is
recomputed from the cohort. Optional input QC (`--ablation qc`) removes values
beyond 8σ of the same training statistics.

## Paths the code expects

All scripts resolve paths from the repository root (`OCEAN_ROOT` overrides it):

- `audit_tools.cohort_path` → `data/argo_cohort/global_global.nc`
  (`--region global`), `data/synthetic_argo/cesm2_uniform.nc`
  (`--region synthetic`), `data/argo_cohort/<region>_ext.nc` otherwise
- anomaly caches → `data/argo_cohort/global_anom{cell,x}.npz`
- climatology → `data/woa23_standard.zarr`
- satellite fields → `data/real_obs_1deg.zarr`
- run outputs → `outputs/audit/<region>/<tag>/`

## Sources and acknowledgements

Argo data were collected and made freely available by the International Argo
Program and the national programs that contribute to it (https://argo.ucsd.edu),
part of the Global Ocean Observing System. Also used: WOA23 (NOAA NCEI), CESM2
Large Ensemble (NCAR), NOAA OISST v2.1, Copernicus Marine DUACS, OISSS
(PO.DAAC), EN4 (Met Office Hadley Centre), ECCO V4r4 (NASA/JPL), GODAS (NOAA
PSL). Each keeps its own terms of use.
