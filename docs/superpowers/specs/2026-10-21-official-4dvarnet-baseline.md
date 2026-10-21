# The official 4DVarNet as a published-code baseline (2026-10-21)

Date: 2026-10-21. Branch: `ka-4dvar` (from `ka-dev4` 51a2f25). Approved by Zihao the same day
("跑一个已发表方法的官方代码做对比"). Written and committed before any training run.

## Why

Every learned baseline of the paper except Poseidon-T is our own re-implementation ("-style"). A
reviewer can ask for a published method run from its authors' code. 4DVarNet (Fablet et al. 2021;
the CIA-Oceanix `4dvarnet-starter`, Beauchamp et al. 2023) is the standard learned interpolation
method of ocean data assimilation with maintained public code. The repository already pins it
(`external/4dvarnet-starter`, commit 20f1b5f) and runs its classes unchanged
(`fourdvar_synthetic.py`). It was run once in early October on the older 2000-2005 CESM2 cohort
(one seed, another adapter, `outputs/audit/synthetic/fourdvarnet*`); those runs are not reused.

## Method: official, unchanged

* `GradSolver`, `ConvLstmGradModel`, `BilinAEPriorCost` executed from the pinned source (its hash
  is checked; the class code is not edited).
* Hyperparameters of `config/xp/base.yaml`: 10 solver steps, solver rate 1e3, prior 32 channels at
  half resolution without the quadratic term, ConvLSTM gradient model 48 channels, dropout 0.1.
* Objective of `Lit4dVarNet.step`: 50 x reconstruction MSE + 1 x prior cost of the output. Its
  Sobel term needs dense truth and is dropped.
* Optimiser of `cosanneal_lr_adam`: Adam 1e-3 (prior 5e-4), cosine annealing over the run,
  gradient norm clipped at 0.5 (the trainer's `gradient_clip_val`).
* The official evaluation output ends with the prior autoencoder's projection; the state before it
  is the training-mode output. Which one is used is chosen on the selection years (below).

## Task adapter (ours)

* State: the global 1-degree grid, T at 20 levels, S at 20 levels, SST, SSS, SSH (43 channels; the
  time-window channels of the original SSH task become depth and variable channels).
* Observation cost: the state read bilinearly at every input profile's exact position against its
  values, plus the surface channels against the surface grids where they exist.
* Initial state: profile cell means and the surface grids, zero elsewhere (the official rule).
* Supervision: the held-out cells of KA-Ocean's training draws, channel-balanced MSE.
* Forecast leads: the input month's reconstruction read at the target month (persisted
  reconstruction, like the persisted OI).
* Code: `src/ocean_tokenizer/fourdvar_grid.py` (tests: `tests/test_fourdvar_grid.py`), runner
  `experiments/synthetic/160_official_4dvarnet.py`, scorer `experiments/real_data/161_fourdvarnet_sealed.py`.

## Data and selection: those of KA-Ocean (method A)

| | CESM2 | Real Argo |
|---|---|---|
| Training | 1964-2003 (`cesm2_long_uniform`, 40 years) | 2016-2020 |
| Selection | 2004-2005, the fixed 8,000-cell draw | 2021, the fixed 8,000-cell draw |
| Inputs a month | 6,080 profiles + SST/SSS/SSH grids | every input float + OISST/OISSS/DUACS grids |
| Training draws | lead 0, 30 % of the floats held out, 1,024 cells a step | same |
| Steps | 70,000 (selection every 2,000) | 30,000 (selection every 1,000) |
| Seeds | 1234, 1235, 1236 | 1234, 1235, 1236 |

The checkpoint and the output path are the ones with the lowest lead-0 mean standardised RMSE on
the selection draw. The chosen checkpoint is then scored on every cell of the selection years.

## Tests: one declared additional pass

The three tests (CESM2 2006-2007, real 2022-2023, real 2024) were opened before this baseline
existed. They are scored once more, for 4DVarNet only, like the CLOINet-style pass:

1. `161 --check` for each run: its selection-year scores must come back through the tests' scenes
   (relative 1e-3); this reads no test year.
2. `161 --register`: the six chosen checkpoints, every test file and the main-pass results, by
   sha256, once.
3. `161 --score cesm2`, `--score real`, `--score real2024`: once each, every held-out cell, leads
   0, 1, 3, 6; the persisted OI must reproduce the main pass (relative 1e-4); each
   `results_<test>.json` is written once and never overwritten.

Nothing is tuned on test data, and every result is reported whatever it is. The real 2025 holdout
is never read.

## Reported

Temperature and salinity RMSE by lead (leads 1-6 as persisted reconstruction) and depth band; MAE,
which equals the CRPS of a deterministic forecast; mean and standard deviation over seeds; the
output path each run chose.
