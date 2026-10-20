# Choosing the final method (2026-10-20)

Date: 2026-10-20. Branch: `ka-dev3` (from `ka-dev2`). Approved by Zihao the same day ("按建议做").

## Why

Three variants of the gridded model are close on the selection years, and the paper must report one
method. The choice is made by a rule written down before the deciding runs finished, on validation
data only, so that the sealed tests (CESM2 2006-2007, real 2022-2023) stay untouched and stay the
honest test of whatever is chosen.

## Candidates (simplest first)

| | Method | CESM2 runs (70k steps, chosen on 2004-2005) | Real runs (30k steps, chosen on 2021) |
|---|---|---|---|
| A | U-Net + kriging attention (flow kernel), joint background | `103` gj_flow (ka-pilot) | `107` scratch (ka-dev) |
| B | U-Net + kriging attention, prior background: the background sees only the surface grids, position and season; the month's profiles enter only through the kriging innovations | `111` c_gp_flow (ka-dev) | `114` rb_prior (ka-dev3) |
| C | Poseidon-T (public weights) + kriging attention, joint background | `112` p_ka (ka-dev2) | `114` rc_poseidon (ka-dev3) |

Poseidon-T supports only the joint background (the backbone is lead-conditioned itself), so there is
no "Poseidon-T + prior background" candidate. Seeds 1234, 1235, 1236 for every run.

## Rule (`experiments/real_data/115_method_selection.py`)

1. Only the selection years are read: CESM2 2004-2005 and real 2021.
2. A run's score is the score it chose its checkpoint by: the mean over leads 0, 1, 3, 6 of the
   per-variable standardized RMSE on the fixed 8000-cell validation draw
   (`summary["validation"]["scores"]["macro_z"]`).
3. A candidate's score on a dataset is the mean over its three seeds. Its overall score is the mean,
   over CESM2 and real data, of that score divided by A's score on the same dataset.
4. The chosen method is the simplest candidate (A, then B, then C) whose overall score is within
   0.5 % (0.005) of the lowest overall score. 0.5 % is about the seed-to-seed spread of the score.
5. `115` refuses to decide while any of the 18 runs is unfinished and writes `decision.json` once.

## After the decision

* The losing candidates stay in the paper: C as "kriging attention on another backbone", B as the
  background ablation.
* `114` pr_random (C with randomly initialised Poseidon-T weights, CESM2, 3 seeds) separates the
  public pretraining from the architecture; it is an ablation, not part of the rule.
* The chosen method is frozen; then both sealed tests are registered and scored once (`109`, `113`),
  after Zihao confirms.
