# Fixed baselines on the synthetic Argo cohort, and the todo status table — design

Date: 2026-10-04
Branch: `synthetic-oi-baseline`
Source: the five-item todo of 2026-10-04 (all on CESM2 synthetic data)

---

## 1. Purpose

The todo has five items. Four already have results from the pre-shutdown
synthetic audit (`outputs/audit/synthetic/`, written up in
`reports/synthetic/synth_argo_audit.md`):

| Todo item | Existing arms | Report section |
|---|---|---|
| Anomaly target at the Argo position | `syn_reg_cell` vs `syn_reg` | §5 |
| Local refiner sweep, 3 seeds each | `syn_reg`, `syn_reg_g1`, `syn_r{150,500,1500}_{g005,g1}` | §4 |
| Slots 32 → 64 | `syn_r500_g1`, `syn_fix_l64`, `syn_fix_l128` | §7 |
| DFS vs uniform vs count | `syn_r500_g1`, `syn_fix_uniform`, `syn_fix_count` | §8 |
| **Compare with a fixed baseline such as OI** | **none** | **none** |

This work fills the one gap and puts all five items in one table. No model is
retrained.

## 2. Scope

**In scope**

- Three fixed (no trainable parameters) baselines scored on the synthetic
  cohort's fixed queries: climatology, nearest profile, optimal interpolation.
- OI hyperparameters tuned on the validation year only.
- A "Fixed baselines" section and a todo status table in the generated
  synthetic report.
- Installing the pinned environment into `.venv` so tests and the report run.

**Out of scope**

- Retraining or re-scoring any model. The model side of every comparison comes
  from the `summary_seed*.json` files already on disk.
- Real Argo, and any new synthetic cohort (for example clustered positions).
- The kernel-weighted `objective_interpolation.ObjectiveInterpolation` of the
  real-data regional study. It has no covariance solve and its frozen settings
  were chosen for a 25° box.
- Selecting anything on the test year.

## 3. The comparison

"Our setup" is the validated configuration `syn_r500_g1`: at-position target,
local refiner initialised at 500 km / 100 m with gate 1.0, Perceiver-IO fuse,
DFS mass, 32 slots, 12 k steps, seeds 1234-1236. Its scores are read from the
existing summaries and reported as mean ± sd over the three seeds.

Every baseline gets the same information and is scored on the same cells:

| | Model (`62_sanity_train.py --region synthetic`) | Baselines (this work) |
|---|---|---|
| Cohort | `load_cohort(ROOT, "synthetic", SPLITS, anomaly="exact")` | same call |
| Normalization | `ArgoNorm.fit(c, "train")` | same call |
| Splits | train 2000-2003, validation 2004, development (test) 2005 | same |
| Inputs for month *m* | every `cohort_float` profile of month *m* (6,080) | same, month *m* only |
| Queries for month *m* | every `heldout_float` profile of month *m* (1,520) × 20 levels | same |
| Validation cells | 8,000 per month, drawn by `default_rng([20260918, m]).choice(R*L, 8000, replace=False)` | same draw |
| Test cells | all | all |
| Metric | pooled RMSE in z and physical units, J = RMSE / RMS of target, overall and per depth band | same formulas |

**Input parity (requirement: no method gets fewer Argo profiles than
another).** Every month of the cohort has exactly 6,080 input profiles and
1,520 query profiles, and all 54 existing training runs (18 arms) were run
with `n_profiles = 0` (every input profile), 12 k steps, batch 1, the same
splits, 351,895 test cells and 92,517 validation cells. This is enforced, not
assumed:

- `44_synth_argo_oi.py` hands every baseline all `cohort_float` profiles of the
  month, with no cap and no subsampling. It asserts 6,080 inputs and 1,520
  queries for every evaluated month and writes the per-month counts to
  `summary.json`.
- OI's `k` nearest observations and the nearest-profile baseline choose among
  all 6,080 profiles; `k` is localisation of the solve, not a smaller input
  set. A level that is non-finite in a profile (below the sea floor) is
  missing for every method alike.
- `43_synth_argo_report.py` reads `n_profiles`, `steps`, `batch`, `splits` and
  the test and validation cell counts from every summary it cites in the todo
  table and in §9, and exits with an error if any arm differs from
  `syn_r500_g1`. The tables print the input profiles per month.
- The earlier gridded OI result (1,500 profiles a month, `oi_baseline.md`) is
  a different experiment and appears in no table of this work.

**Identity check.** Before any baseline number is written, a zero prediction
is scored. Its cell count `n` and `climatology_z` must equal the values stored
in `outputs/audit/synthetic/syn_r500_g1/summary_seed1234.json` for both splits
and both channels (test: 351,895 cells, 1.6557 TEMP / 1.6358 SALT; validation:
92,517 cells, 1.9145 / 1.7182): the cell counts exactly, the climatology error
to a relative tolerance of 1e-5 (the model's sums were accumulated in float32).
If any value differs the script exits with an error and writes nothing.

## 4. Baselines

All three predict the z-scored anomaly at each query cell.

1. **Climatology.** Prediction 0. This is the J = 1 floor.
2. **Nearest profile.** The value of the nearest input profile (great-circle
   distance) that is finite at the same level.
3. **Optimal interpolation.** `ocean_tokenizer.oi` (Bretherton, Davis and
   Fandry 1976), unchanged: level by level and per variable, zero background
   and unit background variance in z space, covariance
   `C(r) = exp(-r² / 2L²)` with `r` the great-circle distance in km, the `k`
   nearest observations, and `gamma` the observation-to-background error
   variance ratio. `oi.LevelSweep` already accepts arbitrary query arrays with a
   mask, so the queries are passed as 1-D arrays with an all-true mask.

## 5. OI tuning

- **Grid.** `L_km` ∈ {150, 250, 400, 600, 900, 1500}, `gamma` ∈
  {0.01, 0.03, 0.1, 0.3}, `k` ∈ {10, 20, 40}: 72 settings.
- **Selection.** One setting per variable and depth band (the four bands of
  `62_sanity_train.py`: 0-100, 100-300, 300-700, 700-1400 m), eight selections
  in all, each the minimum pooled validation RMSE in z over that band's
  validation cells.
- **Edge rule.** If a selected value sits on an end of its axis, the next
  value of that side's ladder is added and the selection is repeated: `L_km`
  down 100 then 60, up 2500 then 4000; `gamma` down 0.003 then 0.001, up 1.0
  then 3.0; `k` down 5, up 80. A selection still on an end when its ladder is
  used up is reported as such.
- **Test.** The test year is scored once, with the eight selected settings.
- **Also recorded.** The best single setting per variable (no band split) and
  its test score, so the gain from per-band tuning is visible.

Tuning per band gives OI its strongest form, so a model that beats it has
beaten a properly tuned baseline.

## 6. Code

| File | Change |
|---|---|
| `src/ocean_tokenizer/point_baselines.py` | New. `nearest_profile(...)`, `point_sweep(...)` / `oi_points(...)` (thin wrappers over `oi.LevelSweep` for scattered queries), and `band_of_levels(...)` / `Scores`, which reproduce the arithmetic of `score()` in `62_sanity_train.py` on numpy arrays. |
| `tests/test_point_baselines.py` | New. `oi_points` equals `oi.oi_level` on the same points laid out as a grid; `oi_points` returns an observation's own value as `gamma → 0`; `nearest_profile` skips non-finite levels and crosses the date line; `Scores` matches a hand-computed case, including J and the per-band split. |
| `experiments/synthetic/44_synth_argo_oi.py` | New. Loads the cohort, builds the evaluation sets of §3, runs the identity check, the tuning of §5 and the three baselines, and writes the outputs of §7. `--smoke` runs one validation month on a 2 × 2 × 1 grid and only prints. Months are spread over a process pool. |
| `experiments/synthetic/43_synth_argo_report.py` | Add §9 and the todo status table (§8 below). Sections 1-8 are not touched. |
| `README.md` | One line for `44_synth_argo_oi.py` in the synthetic audit commands. |

## 7. Outputs

`outputs/audit/synthetic/fixed_baselines/`:

- `summary.json` — for each of `climatology`, `nearest_profile`, `oi` and
  `oi_single` (one OI setting per variable): the
  `scores` block in the schema `62_sanity_train.py` writes (`rmse_z`,
  `rmse_physical`, `unit`, `J`, `climatology_z`, `n`, `by_band_z`,
  `by_band_physical`, `by_band_J`, `macro_z`) for `validation` and
  `development`; the selected OI settings; the identity-check values; the
  cohort path and its SHA-256; the git commit.
- `tuning.json` — validation RMSE in z for every grid setting, per variable and
  band, including any edge extensions.

Both files are small and are committed, like the run summaries.

## 8. Report

`43_synth_argo_report.py` regenerates `reports/synthetic/synth_argo_audit.md`
with two additions:

- **Todo status table**, after the introduction and before the findings: the
  five items, the arms that answer each, the input profiles per month and the
  headline test numbers.
- **§9 Fixed baselines.** A table of test TEMP (°C), SALT (PSU) and J for
  climatology, nearest profile, OI and the model (mean ± sd, 3 seeds), with
  model − OI as a difference and a percentage; a per-band table in z units
  (the model summaries store `by_band_z`); and the selected OI settings.
- **Finding 9** in the findings list, worded from the numbers: the model
  "beats OI" if its 3-seed mean is lower than OI's by more than 2 seed sd for
  both variables, "loses to OI" if higher by more than 2 sd for both,
  otherwise the per-variable result is stated.

## 9. Verification

- `pytest -q` passes in `.venv` after `pip install -r requirements-lock.txt`.
- The input-parity assertions and the identity check of §3 pass.
- After regeneration, `git diff` of `synth_argo_audit.md` shows only the todo
  table, finding 9 and §9; sections 1-8 are unchanged. This also confirms the
  downloaded outputs reproduce the committed report.

## 10. Known risk

OI may match or beat the model on this cohort. The earlier gridded experiment
(`reports/synthetic/oi_baseline.md`) had OI at 0.229 °C with 1,500 profiles a
month; this cohort has 6,080 profiles a month and the model scores 0.230 °C.
The two cohorts differ (years, target, scored cells), so this is not a
prediction. The report states the measured result either way; OI is not
detuned and the model is not retuned in response.

## 11. Cost

CPU only. About 35,000 small OI analyses for tuning (12 months × 2 variables ×
20 levels × 72 settings) plus the test year: a few minutes on 12 processes.
