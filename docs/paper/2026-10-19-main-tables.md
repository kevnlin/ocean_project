# Main tables of the CVPR 2027 paper (frozen 2026-10-19)

Method (frozen): gridded ocean state from rasterised Argo profiles and dense SST/SSS/SSH grids
(U-Net), lead evolution (U-Net conditioned on the lead, zero-initialised), background read at the
query, Kriging Attention over the 32 nearest profiles' innovations with the flow-dependent
(Paciorek-Schervish) kernel and a learned lead decay. Background "joint". No frozen OI.
Code: branch `ka-pilot` (frozen) and `ka-dev`. All numbers: RMSE, temperature degC / salinity PSU,
mean ± standard deviation over seeds 1234-1236, on every scored cell.

## Table 1. CESM2 OSSE, sealed test 2006-2007

Leads 0 (reconstruction), 1, 3, 6 months. Trained on 1964-2003, checkpoint chosen on 2004-2005,
scored once with `109` (`docs/superpowers/specs/2026-10-19-sealed-cesm2-test.md`).

| Row | Run | Status |
|---|---|---|
| Climatology | zero anomaly | scored by 109 |
| Persisted OI | tuned OI of the input month (lead 0: OI) | scored by 109 |
| ARMOR3D-style: satellite first guess + OI of in-situ innovations | 111 `c_gp_oi` | running |
| Aardvark-style encoder-processor-decoder (no exact-value read-out) | 103 `gj_none` | done |
| Anchored transformer: OI anchor + local refinement (strongest point model) | 95 `f_y40` | 2 seeds done, 3rd running |
| Ours | 103 `gj_flow` | done |

## Table 2. Real Argo + OISST/OISSS/DUACS SLA, test 2022-2023

Trained on 2016-2020, checkpoint chosen on 2021, 2022-2023 scored once (a real version of `109`,
to be written before scoring; same rules).

| Row | Run | Status |
|---|---|---|
| Climatology (WOA23) | zero anomaly | from 106 / scorer |
| OI (tuned on 2021) and persisted OI | 106 selection | done |
| ARMOR3D-style | 111 `r_gp_oi` | running |
| Aardvark-style | 111 `r_none` | running |
| Ours, trained on real data only | 107 `scratch` | done |
| Ours, CESM2-pretrained and fine-tuned | 107 `finetune` (or `r_finetune_noise` if it is better on 2021) | done / running |

## Table 3. Ablations (CESM2; reported on 2004-2005 and on the sealed test)

| Group | Rows | Runs |
|---|---|---|
| Read-out | none / softmax / Kriging Attention with the OI kernel / learned stationary kernel / flow kernel | 103 `gj_none`, `gj_softmax`, `gj_oi`, `gj_stat`, `gj_flow` (done) |
| Background | joint / satellite-only prior | `gj_flow`, 111 `c_gp_flow` |
| Lead evolution | with / without | `gj_flow`, 111 `c_noevo` |
| Lead decay | with / without | `gj_flow`, 111 `c_nodecay` |
| Simulation length | 4 / 40 training years | 111 `c_y4`, `gj_flow` |

## Table 4. Transfer from simulation to observations (real, 2021 and 2022-2023)

Real training years 1 (2020), 2 (2019-2020), 5 (2016-2020) x {from scratch, CESM2-pretrained
fine-tune}, plus zero shot; the same with noise-augmented pretraining (111 `c_noise`). Shown as a
small table and a label-efficiency curve.

## Figures and extra columns

* Kriging Attention against softmax: duplicate profiles, screening, negative weights; learned
  anisotropic length scales on a map (flow dependence).
* Skill against lead (CESM2 and real); error by depth band; one cross-section.
* Uncertainty: CRPS and 68/95 % coverage from the predicted standard deviation (to add to the
  scorers before the test is opened).
* Cost: parameters, training hours, inference time per month.

## Before any test is opened

1. All runs above finished; the reference's third seed finished.
2. The scorers compute RMSE, J, per-depth-band RMSE, CRPS and coverage.
3. One registration per test (CESM2, real) listing every row above, all seeds.
