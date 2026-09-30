# Ocean project — pre-shutdown summary (2026-09-29)

What was done before the server shut down, the key numbers, and where
everything lives. Detailed, regenerable reports are linked in each section.

## Where everything is

| What | Where |
|---|---|
| Data, outputs, checkpoints | Hugging Face `klin2323/ocean_project-data` (private dataset), revision `6836f29`: https://huggingface.co/datasets/klin2323/ocean_project-data |
| Data description (splits, preprocessing, target, normalization, paths) | [`docs/DATA.md`](../docs/DATA.md) (also the HF dataset card) |
| Code | GitHub `kevnlin/ocean_project`, branch `preshutdown-backup`: https://github.com/kevnlin/ocean_project/tree/preshutdown-backup · PR to open: https://github.com/kevnlin/ocean_project/compare/main...preshutdown-backup |
| Setup and reproduction commands | [`README.md`](../README.md), "Quick start on a new machine" |
| Real-data validation (A-D) | [`reports/real_data/preshutdown_validation.md`](real_data/preshutdown_validation.md), [`target_qc_audit.md`](real_data/target_qc_audit.md) |
| Synthetic audit | [`reports/synthetic/synth_argo_audit.md`](synthetic/synth_argo_audit.md) |
| Run results | `outputs/audit/global/<arm>/` (real) and `outputs/audit/synthetic/<arm>/`: `summary_seed*.json` (scores, config), `summary_eval_seed*.json` (per-depth re-score), `history.jsonl` (curves), `model_seed*.pt` (selected weights) |

Recover on a new machine:

```bash
git clone -b preshutdown-backup https://github.com/kevnlin/ocean_project && cd ocean_project
python -m venv .venv && .venv/bin/pip install -r requirements-lock.txt
hf download klin2323/ocean_project-data --repo-type dataset --local-dir .
.venv/bin/python -m pytest -q
```

The HF backup leaves out only the raw GDAC float files (66 GB) and raw ECCO
in-situ files (36 GB); both re-download, and every cohort built from them is
included.

## Experiments completed

**Real Argo** (global cohort, train 2016-20 / val 2021 / test 2022-23, scored
at WMO-held-out floats, 405 K model, 12 k steps, seeds 1234-1236; selection on
validation, test reported once):

| | Result | Numbers (test, paired over seeds) |
|---|---|---|
| **A. Target at the profile's position** | **Small, consistent gain** | Artifact C(position) − C(cell centre) is 2 % of the target on real Argo (12-77 % on synthetic). At the local refiner: −0.0082 ± 0.0018 °C, −0.0012 ± 0.0002 PSU, better on all 3 seeds (about 0.8 %). Better at 0-300 m and 700-985 m, slightly worse at 300-700 m (+0.004 °C). At the registered refiner the comparison is swamped by collapse (−0.006 ± 0.018 °C over 3 seeds; −0.016 ± 0.007 °C on the 2 seeds where neither arm collapsed). |
| **B. Local refiner (500 km / 100 m / 1 month / gate 1.0)** | **Improves, and removes the collapse** | The registered refiner ended at climatology on seed 1235 (test J 0.97); the local one never did. Seed-to-seed sd of test TEMP 0.079 → 0.001 °C. On the 2 non-collapsed seeds: −0.014 ± 0.007 °C, −0.0052 ± 0.0008 PSU. Best configuration (both fixes): 1.026 °C / 0.188 PSU, J 0.859 / 0.891. |
| **C. Overfit one fixed real month, 20 k steps** | **Fits far better than before, but not to 0.01** | Full model 0.050 z TEMP (0.059 °C), 0.021 z SALT (0.004 PSU); macro 0.087 z at 8 k → 0.035 z at 20 k, still falling when the schedule ends. Latent-only 0.035, raw-tokens-only 0.025 macro z. The earlier 0.068 z (8 k steps) was under-training. Real targets carry fine-scale structure (a float's cycles in one month, 28 km apart, differ by 0.5 z) that synthetic CESM2 at 1° does not, which is consistent with it fitting ~30× slower than synthetic (0.001 z). |
| **D. Extreme z-scores and QC** | **Salinity extremes are data errors** | 603 salinity values beyond \|z\| = 100: whole profiles at 10-15 PSU near 1000 m, mostly delayed-mode floats that passed Argo QC, carrying 53 % of the salinity Σz². Of the 25 largest temperature values, 21 are impossible for their location (e.g. −1.9 °C at the tropical surface). The 8σ QC removes 45 k TEMP / 73 k SALT values with median \|z\| ≈ 6, so it also removes many valid values (on synthetic, where every value is true, the same rule removed 92 k TEMP / 34 k SALT). |

**Synthetic CESM2** (known truth; uniform random positions; 63 runs):
overfit passes on every decoder (≤ 0.0056 z at 20 k steps); the at-position
target is the largest effect (−0.074 °C, −24 %); refiner 500 km / gate 1.0
selected on validation (−0.0075 °C); 32/64/128 slots, DFS/uniform/count mass
and Perceiver-IO vs PhCA-style fuse all tie within ~0.003 °C.

## What improved

- **Local refiner start** (B): removes the seed-dependent collapse that made
  every single-seed ranking in the 09-17 audit unreadable, and lowers error.
- **Target at the profile's position** (A): consistent but small on real Argo
  (~0.8 %); large on synthetic because the synthetic anomaly is small.
- **Longer training** (C): real memorisation 0.068 → 0.035 z.

## What did not improve (or was not tested on real data)

- Latent capacity, mass rule and backbone: ties on synthetic; not re-run on
  real data (lower priority in the plan).
- Real-data overfit did not reach 0.01 z within 20 k steps.

## Open issues

1. **Salinity QC.** The errors are real, but blanket 8σ clipping also removes
   valid values. A profile-level screen for impossible values (anomaly beyond
   ~5 PSU / 15 °C, or a profile whose median \|z\| exceeds ~10) is the obvious
   next test. Any QC comparison must hold the scored target set fixed: QC also
   removes values from the evaluation targets, so a QC run's RMSE (e.g.
   `fixed_stack`, 0.154 PSU) is not directly comparable with an unfiltered one.
2. **Real overfit floor.** Run past 20 k steps to see where it levels off.
   Query coordinates carry the month but not the day, so a float's repeated
   cycles in one month are separable only by position.
3. **Normalization.** One std per level pools regions of very different
   variability; a location-dependent scale was proposed but not tested.
4. **Real-data DFS vs uniform.** Synthetic positions have no clustering, so
   the synthetic tie says nothing about DFS on clustered real floats.
5. **Housekeeping.** The PR from `preshutdown-backup` to `main` still has to be
   opened (no GitHub CLI on the server). Uncommitted edits to
   `14_godas_dfs_d4rt.py`, `godas_model.py` and `godas_obs.py` were left out of
   the repository on purpose; they exist only on the server.

## Experiment log

Each experiment's card (hypothesis, baseline, change, data, seeds,
checkpoints, TEMP/SALT RMSE, conclusion) is in
[`preshutdown_validation.md`](real_data/preshutdown_validation.md). Every run
records its configuration and seed in `summary_seed*.json`; the code is the
`preshutdown-backup` branch; the data version is HF revision `6836f29`;
training commands are in the README and in `run_audit_queue.py`
(`real_final`, `real_overfit`, `syn_*`).
