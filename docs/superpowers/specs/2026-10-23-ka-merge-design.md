# Kriging attention into main

Date: 2026-10-23. Branch `ka-merge` (from `main` 08d0faa plus the forecast commit b7984b1).

## Why

Zihao's kriging-attention model (KA-Ocean: U-Net state, lead evolution, kriging read-out; "method A")
and the experiments answering Wang's 2026-10-21 feedback exist only on the `ka-*` branches. Main keeps
the OI-anchored transformer as its model; KA comes in beside it as the comparison arm, with its
history, runners, specs and reports. No new model code is written.

What Wang asked and where the branches answer it is `docs/superpowers/specs/2026-10-21-wang-feedback-experiments.md`
(learned covariances, multivariate KA, 100k steps, 154 years, augmentation, GLORYS12); results are
`reports/paper/wang_feedback_results_20261022.md`. The selected model has no transformer layers.

## What is merged

The seven leaf branches, in this order: `ka-dev3`, `ka-dev6`, `ka-dev8`, `ka-dev7`, `ka-ihesp`,
`ka-4dvar`, `ka-dev5`. Every other `ka-*` branch is an ancestor of one of them. A trial merge onto
08d0faa added 112 files with four conflicts:

| file | resolution |
|---|---|
| `.gitignore` | union of both sides |
| `experiments/synthetic/95_forecast_campaign.py` | main's version, plus any change of `ka-dev3`'s that main lacks |
| `experiments/synthetic/151_synthetic_gp_nongaussian.py` | `ka-dev8`'s (adds `--kinds`), after checking it contains `ka-dev6`'s |
| `src/ocean_tokenizer/kriging_attention.py` | both: `ka-dev5`'s read-out mode beside `ka_deep`/`ka_product`/`ka_multiscale`; `MODES` and the per-mode tables are the union, existing modes unchanged |

Two scripts numbered 97 (`97_forecast_curves.py`, `97_kriging_pilot.py`) stay as they are; the repo
already has duplicate numbers.

## What is not touched

The `ka-*` branches, their worktrees under `/DATA2/linke/ocean_project_ka*`, and any queue running in
them. Nothing is pushed or deleted. Later commits on those branches can be merged again.

## Checks

* The full test suite on main before the merge (baseline) and on `ka-merge` after it; no test that
  passed before may fail, and every KA test passes.
* Every runner the merge adds imports (`python -c` on its module; nothing trains).
* Kevin approves before `ka-merge` goes into `main`.

## Known consequence

The runners hash their sources and refuse to resume when they changed. The merged
`kriging_attention.py` equals no single branch's copy, so a branch checkpoint cannot be resumed from
main; finished checkpoints still evaluate. Running queues stay in their worktrees.
