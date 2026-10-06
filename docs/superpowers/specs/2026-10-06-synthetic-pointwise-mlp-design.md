# Pointwise MLP baseline on the synthetic Argo cohort — design

Date: 2026-10-06
Branch: `synthetic-pointwise-mlp`
Source: request of 2026-10-06, "for the fixed baselines, run pointwise MLP as
another baseline ... just the Argo profiles, run it the same way it was
originally", on the setup of the other synthetic results.

---

## 1. Purpose

Section 9 of `reports/synthetic/synth_argo_audit.md` compares the trained
model with climatology, nearest profile and optimal interpolation. This adds
the repo's pointwise MLP (`ocean_tokenizer.baselines`, the gridded line's
`mlp` method) in its profiles-only form as a fourth reference. It is a trained
baseline, not a fixed one, and the report says so.

## 2. The method, kept as it was

`baselines.MLP` and its settings in `config.py` are used unchanged: hidden
layers 256-256-256 with SiLU, two outputs (TEMP and SALT in z units), Adam at
learning rate 1e-3, MSE loss, 30 epochs, batches of 65,536, 120,000 training
points per month, the weights of the last epoch (no early stopping, no
selection on validation).

Features are those of `baselines._point_features` with the `profiles` input
only, nine per query cell:

| # | Feature | As in the original |
|---|---|---|
| 1 | latitude ÷ the 1° grid's latitude std (51.96°) | `(lat − mean) / std` of the grid latitudes, whose mean is 0 |
| 2-3 | sin and cos of longitude | same |
| 4 | depth, standardized over the 20 levels | same |
| 5-6 | sin and cos of 2π · calendar month / 12 | same |
| 7-8 | the nearest input profile's TEMP and SALT z-anomaly at that level, 0 where it has none | `near`, `nan_to_num` |
| 9 | chord distance on the unit sphere to that profile | `nn_dist` |

"Nearest" is the single horizontally nearest input profile, the same one for
every level, as in the original.

## 3. What has to differ from the gridded original

The original trained on the full CESM2 field at random grid points. On this
cohort a method sees Argo profiles only, so the training targets are profiles,
under the rule the trained model follows (`62_sanity_train.py`):

- training months are 2000-2003; only `cohort_float` profiles are used;
- in a month, 30 % of them are drawn as targets and the other 70 % are the
  inputs the features are computed from;
- four such draws per month, from which 120,000 cells with finite TEMP and
  SALT targets are kept (the original's points per month).

The target is the at-position anomaly in z units (`ArgoNorm` fitted on the
training years), the target of every other synthetic result.

## 4. Evaluation, identical to the other baselines

- Inputs at evaluation: all 6,080 `cohort_float` profiles of the month.
- Queries: the 1,520 `heldout_float` profiles; validation 2004 on the capped
  8,000 cells a month, test 2005 on all cells (351,895 per variable).
- The input-parity assertion and the zero-prediction identity check of
  `44_synth_argo_oi.py` run before any number is written.
- Seeds 1234, 1235, 1236; a seed fixes the target draws, the point subsample,
  the initial weights and the batch order.

The evaluation sets move from `44_synth_argo_oi.py` into a shared module so
both drivers build them from one definition; `44` must reproduce its committed
scores exactly after the move.

## 5. Code and outputs

| File | Change |
|---|---|
| `src/ocean_tokenizer/synth_argo_eval.py` | New. Cohort loading, the fixed evaluation sets, the parity assertion and the identity check, taken from `44`. |
| `src/ocean_tokenizer/point_baselines.py` | Add `mlp_point_features(...)`. |
| `tests/test_synth_argo_eval.py`, `tests/test_point_baselines.py` | Tests for the two additions. |
| `experiments/synthetic/44_synth_argo_oi.py` | Use the shared module; no change in results. |
| `experiments/synthetic/45_synth_argo_mlp.py` | New driver: training set, training, scoring, `outputs/audit/synthetic/pointwise_mlp/summary_seed<seed>.json` in the run-summary schema. |
| `experiments/synthetic/43_synth_argo_report.py` | Section 9: an MLP row in the table and the band table, a sentence in the text, finding 9 and the todo row; the report's parity check covers the MLP summaries. |
| `README.md` | One command line. |

## 6. Verification

- `pytest -q` passes.
- `44_synth_argo_oi.py` re-run after the refactor gives the committed
  `scores`, selections and tuning grid, value for value.
- The MLP driver's parity and identity checks pass on all three seeds.
- The regenerated report changes only section 9, finding 9 and the todo row.
