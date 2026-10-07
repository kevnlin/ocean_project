# Multi-modal comparison on the synthetic cohort, with 4DVarNet as a baseline — design

Date: 2026-10-06
Branch: `multimodal-4dvarnet`
Source: request of 2026-10-06 — research baselines with code for a multi-modal
comparison, build the comparison table this week, and adapt 4DVarNet to our
input data as a baseline. Decisions taken with the user the same day: data =
synthetic CESM2; baseline this week = 4DVarNet only; codebase =
`4dvarnet-starter`; design approved as below.

---

## 1. Purpose

Every synthetic result so far is Argo-only. This adds satellite-type inputs to
the synthetic cohort, runs our model and 4DVarNet with and without them, and
puts all methods in one table.

## 2. Scope

**In scope**

- A surface store for the synthetic cohort: SST, SSS and sea level from CESM2.
- 4DVarNet, adapted from `4dvarnet-starter`, Argo-only and Argo + satellite.
- Our model with the satellite fields.
- One comparison table in the synthetic report.
- The baseline survey as a document in the repo.

**Out of scope**

- Real Argo and real satellite data.
- The other candidates of the survey (ARMOR3D-style, OSnet / NeSPReSO /
  Nardelli, ConvNP): listed, not run.
- A 4DVarNet trained on the full CESM2 field.
- Pushing: commits stay local until asked.

## 3. Inputs, identical for every multi-modal method

For month *m* a multi-modal method receives:

1. all 6,080 `cohort_float` profiles of month *m*, as now;
2. three 1° monthly fields of month *m*: `SST`, `SSS`, `SLA`.

The fields come from `data/cesm2_le_full_standard.zarr` (2000-01 … 2005-12):

- `SST`, `SSS`: the store's own variables.
- `SLA`: the store has no sea-level variable. It is the repo's existing
  substitute, steric height relative to 990 dbar computed with TEOS-10 from
  CESM2's TEMP and SALT on the 20 analysis levels
  (`ocean_tokenizer.ssh.steric_height_columns`), NaN where the column does not
  reach the reference level.
- Each field is stored as an anomaly against its own train-year (2000-2003)
  monthly climatology, the way the T/S target is defined.

Written by a new script to `data/synthetic_argo/cesm2_surface_1deg.zarr` with
dimensions `(time, lat, lon)` and the variable names of the real satellite
store, so `62_sanity_train.py` reads it through its existing `--surface` path.

**Limitation, to be stated wherever these numbers appear:** steric height is a
vertical integral of the T and S being reconstructed. It is an optimistic
stand-in for altimetry for every method that uses it.

## 4. 4DVarNet

### 4.1 Code

`https://github.com/CIA-Oceanix/4dvarnet-starter` at commit `20f1b5f`, cloned
into the git-ignored `external/4dvarnet-starter` and left unmodified (CeCILL-C).
The driver imports its `GradSolver`, `BilinAEPriorCost`, `ConvLstmGradModel` and
`BaseObsCost` from `src/models.py`. `kornia` is installed because that module
imports it. The starter's data pipeline and Lightning module are not used: they
are built for one gridded SSH variable with dense targets.

### 4.2 Formulation on our data

| Element | Choice |
|---|---|
| State | TEMP and SALT z-anomalies on 20 levels: 40 channels on the 180 × 360 global 1° grid, one month at a time |
| Observations | the input profiles binned to their 1° cell (`grid_y`, `grid_x`), mean per cell and level, NaN elsewhere |
| Initial state | observations with 0 (climatology) elsewhere, as the starter's `init_state` |
| Prior, gradient model, solver | the starter's classes with the hidden sizes, step count and step size of its `config/xp/base.yaml`; channel count 40 instead of its time window |
| Observation term, Argo-only | `BaseObsCost` unchanged |
| Observation term, Argo + satellite | `BaseObsCost` plus a term on the surface fields that follows the starter's `contrib/multimodal.MultiModalObsCost`: MSE between a learned 3 × 3 convolution of the state and one of the fields. Rewritten in our driver because the original assumes state and field have the same channel count |
| Longitude | fields are padded circularly by 8 cells in longitude before the solver and cropped after, so there is no seam at 0° E |

### 4.3 Training, with Argo targets only

The starter trains on a dense target and adds a Sobel-gradient loss. Neither
exists here: a method on this cohort sees Argo profiles only.

- Training months 2000-2003, `cohort_float` profiles only.
- Each time a month is drawn, 30 % of its input profiles are held out and
  gridded as the target; the other 70 % are gridded as the observations. This
  is the rule of `62_sanity_train.py` and of the pointwise MLP.
- Loss = `50 × MSE(output, target)` on the cells that have a target, plus the
  prior cost, with the starter's weights. The gradient loss is dropped.
- Optimiser, learning-rate schedule, epochs (150) and batch size (4 months) as
  in the starter's base configuration.
- After every epoch the validation year is scored the way the test year will
  be; the weights with the lowest validation macro z are kept. Test is scored
  once with them.
- One fixed run at seed 1234 per variant, reported as a single number.

### 4.4 Scoring

At evaluation the observations are all 6,080 input profiles of the month. The
40-channel output is sampled bilinearly (periodic in longitude) at the 1,520
query positions and scored with `point_baselines.Scores` on the evaluation sets
of `synth_argo_eval`: the same 351,895 test values, after the input-parity
assertion and the zero-prediction identity check.

## 5. Our model with satellite fields

`62_sanity_train.py --region synthetic --surface` with the current
configuration (at-position target, 500 km refiner at gate 1.0, 64 slots,
15,000 steps), seeds 1234-1236. Two changes to the driver: the surface store is
chosen by region, and the "synthetic task is Argo-only" exit is removed. A new
queue `syn_surface_k15_l64` in `run_audit_queue.py` runs it. Its Argo-only
counterpart is the existing `k15_l64_syn_r500_g1`.

## 6. Code and outputs

| File | Change |
|---|---|
| `experiments/synthetic/46_synth_surface_fields.py` | New. Builds the surface store of §3. |
| `src/ocean_tokenizer/gridded_obs.py` | New. `bin_profiles` (profiles → gridded mean and mask) and `sample_bilinear` (grid → points, periodic in longitude). |
| `tests/test_gridded_obs.py` | New. |
| `experiments/synthetic/47_synth_argo_4dvarnet.py` | New. The 4DVarNet driver of §4; `--surface` selects the multi-modal variant. Writes `outputs/audit/synthetic/fourdvarnet{,_surface}/summary_seed1234.json`. |
| `experiments/real_data/62_sanity_train.py` | Surface store by region; synthetic no longer refuses `--surface`. |
| `experiments/real_data/run_audit_queue.py` | Queue `syn_surface_k15_l64`. |
| `experiments/synthetic/43_synth_argo_report.py` | New section with the comparison table; the parity check covers the new runs. |
| `docs/baseline_survey.md` | New. The candidate baselines, their papers and code. |
| `README.md` | Commands, and the `external/` clone step. |

## 7. The table

One row per method, test year 2005, TEMP (°C), SALT (PSU) and J:

| Method | Inputs | Source |
|---|---|---|
| Climatology, nearest profile, pointwise MLP, OI | Argo | existing |
| 4DVarNet | Argo | new |
| 4DVarNet | Argo + SST, SSS, SLA | new |
| Our model | Argo | existing `k15_l64_syn_r500_g1` |
| Our model | Argo + SST, SSS, SLA | new |

Every row uses the same 6,080 input profiles a month and the same test values;
the report refuses to render otherwise.

## 8. Verification

- `pytest -q` passes.
- The surface store has 72 months, finite SST and SSS on wet cells, and an SLA
  whose anomaly correlates with the 100-300 m temperature anomaly (the check
  `28_make_ssh.py` already reports for this field).
- Both 4DVarNet runs pass the parity and identity checks; their training loss
  falls and their validation error ends below climatology.
- The regenerated report changes only by the new section, finding and table.

## 9. Known risks

- 4DVarNet is trained here without the dense truth and gradient loss it is
  normally trained with, so it may do worse than its published form. The table
  labels how it was trained.
- The Perceiver backbone with satellite tokens has not been run on a global
  domain; if it does not fit in memory or time, the satellite patch size is
  enlarged and the change is recorded.
- Our model has lost to OI on this cohort. The satellite fields may or may not
  change that; the report states what is measured.
