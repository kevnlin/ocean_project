# GODAS registered rows — first real run (mentor doc §§2.1–2.6, §4, §6)

Run date: 2026-08-17 | Branch: `intern/d4rt-query-decoder`
Source: `dfs_d4rt_intern_plan.md` §§2.1–2.6, §4, §6

---

## 0. Provenance

`dfs_d4rt_intern_plan.md` is written in the past tense as a finished audit but
**was never run** (confirmed with its author 2026-08-16). Its §7/§8/§9 tables,
§6.3 checkpoint hashes, §4 manifest SHA and "106 passed" are targets, not
measurements. **These are the first actual numbers for those sections.** They
are not a reproduction of anything and must not be merged with the document's
own tables.

## 1. What was built and run

Everything in §§2.1–2.6 and §4 was built from scratch; §6.1's `cnp_cbottle` was
**not** built, because the document specifies only "width 168, 247,088
parameters" with no architecture, and inventing one would misattribute it.

| Piece | Module |
|---|---|
| §2.1 observation contract, 664-token budget, five masks | `godas_obs.py` |
| §2.2 set-level DFS + conservative transport | `batched_dfs.py` |
| §2.5 causal local OI + superobservations | `objective_interpolation.py` |
| §2.6 cBottle masked loss | `losses.py` |
| §2.6 frozen-OI residual, 8 lead/channel gates | `oi_residual.py` |
| §2.3/§2.4 query decoder, local refiner, channel experts | `query_decoder.py` |
| §4 GODAS download + checked loader | `13_download_godas.py`, `godas.py` |
| row assembly | `godas_model.py` |
| §6 driver | `14_godas_dfs_d4rt.py` |

Data: GODAS 25–50 N, 280–331 E, 2000–2025, 16 levels 5–949 m, **38 × 26** grid,
78 files, 154 MB via OPeNDAP server-side subsetting. Splits reproduce the
document's eligible-source counts exactly: **224 / 32 / 32 / 8**.

Training follows §6.2: 5000 steps, 512 queries, AdamW 3e-4, validation every
500, selection = mean over channels of RMSE / validation-climatology RMSE.
Seeds 1234/1235/1236. All values below are anomaly z units.

## 2. Validation (2019–2021, 32 source months, 3 seeds)

| Row | Selection score | TEMP | SALT |
|---|---|---|---|
| `count_oi_expert_cbottle` | **0.6520 ± 0.0201** | 0.7398 ± 0.0167 | 0.8569 ± 0.0352 |
| `uniform_oi_expert_cbottle` | 0.6523 ± 0.0222 | 0.7467 ± 0.0198 | 0.8500 ± 0.0359 |
| `dfs_oi_expert_cbottle` | 0.6577 ± 0.0226 | 0.7499 ± 0.0135 | 0.8604 ± 0.0438 |
| `uniform_expertlocal_cbottle` | 0.6643 ± 0.0155 | 0.7824 ± 0.0225 | 0.8409 ± 0.0154 |
| `dfs_expertlocal_cbottle` | 0.6657 ± 0.0188 | 0.7832 ± 0.0197 | 0.8437 ± 0.0272 |
| `count_expertlocal_cbottle` | 0.6684 ± 0.0173 | 0.7901 ± 0.0236 | 0.8431 ± 0.0185 |
| `objective_interpolation` | 0.7090 ± 0.0228 | 0.7966 ± 0.0104 | 0.9407 ± 0.0494 |

## 3. 2025 holdout (8 source months, 3 seeds)

> **Superseded — see §8 below.** The table in this section predates the SSH
> co-location fix (commit `425b486`). The DFS rows in it are affected; the
> controls are not. The corrected all-rows-one-code-state numbers are in §8.


| Row | Score | TEMP | SALT |
|---|---|---|---|
| **`objective_interpolation`** | **0.7159 ± 0.0076** | 0.7945 ± 0.0142 | 0.9508 ± 0.0264 |
| `dfs_expertlocal_cbottle` | 0.7264 ± 0.0476 | 0.8258 ± 0.0494 | 0.9375 ± 0.0683 |
| `uniform_expertlocal_cbottle` | 0.7346 ± 0.0390 | 0.8232 ± 0.0370 | 0.9645 ± 0.0609 |
| `count_expertlocal_cbottle` | 0.7387 ± 0.0366 | 0.8350 ± 0.0369 | 0.9598 ± 0.0538 |
| `count_oi_expert_cbottle` | 0.7421 ± 0.0412 | 0.8063 ± 0.0458 | 1.0099 ± 0.0624 |
| `dfs_oi_expert_cbottle` | 0.7506 ± 0.0547 | 0.8202 ± 0.0450 | 1.0148 ± 0.0994 |
| `uniform_oi_expert_cbottle` | 0.7537 ± 0.0422 | 0.8189 ± 0.0385 | 1.0253 ± 0.0708 |

## 4. Findings

**The ordering inverts between validation and holdout.** Standalone OI is last
on validation (0.7090) and **first on the holdout (0.7159)**. Every learned row
generalises worse to 2025 than the deterministic baseline it was built to
improve on, and the OI-residual rows — best on validation by a clear margin —
are the *worst* on holdout. On 32 validation months that is the signature of
selection overfitting, and it is the single most important thing in this
report.

**DFS is not better than its matched controls.** Paired per-seed differences on
the holdout score (negative = DFS better):

| Comparison | seed 1234 | 1235 | 1236 | mean | verdict |
|---|---|---|---|---|---|
| `dfs_oi` − `uniform_oi` | −0.0015 | −0.0166 | +0.0089 | −0.0031 | mixed |
| `dfs_oi` − `count_oi` | −0.0030 | −0.0065 | +0.0349 | +0.0085 | mixed |
| `dfs_expert` − `uniform_expert` | −0.0036 | −0.0226 | +0.0017 | −0.0082 | mixed |
| `dfs_expert` − `count_expert` | −0.0002 | −0.0223 | −0.0142 | −0.0123 | DFS better in all 3 |

Only one of four comparisons favours DFS consistently, and it is against the
*conventional* control rather than the matched-mechanism one. **No
DFS-superiority claim is supportable from these runs**, which is what the
document's own §8 anticipated.

**Every row selected its final step** (5000, or 4500 for two). As on the
CESM2 line, §6.2's budget truncates rather than converges here, so all
conclusions above are provisional on undertrained models.

## 5. A bug worth recording

The first pass of this run produced `count` and `uniform` scores agreeing to
four decimals across all three seeds — 0.7824 ± 0.0225 TEMP for both. They were
the same model: `count` fed unit masses to the *conservative* transport, which
is exactly `uniform`. §2.2 requires `count` to use "a conventional fixed-query
Perceiver resampler" whose unit ω "does not drive its global resampler" — a
different mechanism, normalising over **tokens** rather than **slots**, so token
multiplicity feeds through instead of being conserved away.

This mattered: §2.4 calls `count_oi_expert_cbottle` "required to separate an
OI-residual gain from a DFS gain", which a duplicate row cannot do. Fixed
(`PerceiverResampler`), the whole 3-seed run repeated, and a regression test
now asserts the two rows use different resamplers and produce different
predictions. All tables above are from the corrected run.

## 6. Limitations

1. **Undertrained** — every row selected its final step.
2. **No bootstrap intervals.** Aggregate errors are not independent; per-source
   statistics must be saved before intervals can be computed. Same gap the
   document's §10 flags.
3. **`cnp_cbottle` absent** — under-specified in the source.
4. **The holdout was evaluated in the same pass as validation.** There has been
   no selection pressure on it yet, so it is chronologically untouched, but it
   is not a locked-checkpoint evaluation in the §3.8 sense. Re-running the
   evaluation from the saved checkpoints would make it one.
5. **GODAS is a reanalysis**, so sparse inputs and dense labels come from one
   product. Even a clean 2025 result is a reanalysis-OSSE, not independent
   observational validation.
6. **Batch size 1, not §6.2's 2**, and plain masked MSE is used for the loss
   where a row name says `cbottle` only in the sense that `CBottleMaskedLoss`
   is the loss applied — the group-balanced variant is deliberately absent.

## 7. Reproduce

```bash
.venv/bin/python experiments/13_download_godas.py --start-year 2000 --end-year 2025
ALL=objective_interpolation,count_expertlocal_cbottle,uniform_expertlocal_cbottle,\
dfs_expertlocal_cbottle,count_oi_expert_cbottle,uniform_oi_expert_cbottle,dfs_oi_expert_cbottle
for s in 1234 1235 1236; do
  CUDA_VISIBLE_DEVICES=$((s-1234)) .venv/bin/python experiments/14_godas_dfs_d4rt.py \
      --configs "$ALL" --seed $s --steps 5000 --validation-interval 500 \
      --queries 512 --eval-queries 1024 --output outputs/godas_v2_s$s
done
```

Artifacts: `outputs/godas_v2_s{1234,1235,1236}/metrics_seed*.json` and the
per-row checkpoints beside them. Wall clock ~24 min per seed on one GPU at
0.023 s/step.


---

# 8. Corrected results after the SSH co-location fix (commit `425b486`)

A bug found after the first run: SSH and surface T/S patch tokens had
**bit-identical coordinates** (same x, y, both z=0, same t), so the purely
geometric §2.2 estimator consolidated them as exact duplicates. Adding SSH
*deleted* evidence from the stream it shadowed — surface omega 5.197 -> 3.499,
with `surf` and `ssh` summing to exactly the same value. Fixed by making the
kernel variable-group aware (profiles and surface both carry T/S and still
compete; SSH does not consolidate with them).

All six trainable rows re-run under one code state, 3 seeds.

## 8.1 Holdout 2025

| Row | Score | TEMP (z) | SALT (z) |
|---|---|---|---|
| `objective_interpolation` * | **0.7159 ± 0.0076** | 0.7945 | 0.9508 |
| `dfs_expertlocal_cbottle` | 0.7225 ± 0.0474 | 0.8254 | 0.9266 |
| `dfs_oi_expert_cbottle` | 0.7332 ± 0.0318 | 0.8067 | 0.9836 |
| `uniform_expertlocal_cbottle` | 0.7346 ± 0.0390 | 0.8232 | 0.9645 |
| `count_expertlocal_cbottle` | 0.7387 ± 0.0366 | 0.8355 | 0.9591 |
| `count_oi_expert_cbottle` | 0.7396 ± 0.0384 | 0.8048 | 1.0045 |
| `uniform_oi_expert_cbottle` | 0.7532 ± 0.0455 | 0.8161 | 1.0279 |

\* deterministic, unaffected by the fix; carried from §3.

## 8.2 DFS vs matched controls — the conclusion changes

| Comparison | per seed | mean | verdict |
|---|---|---|---|
| `dfs_oi` − `uniform_oi` | −0.0203 / −0.0046 / −0.0349 | **−0.0200** | DFS better ×3 |
| `dfs_expert` − `uniform_expert` | −0.0032 / −0.0200 / −0.0131 | **−0.0121** | DFS better ×3 |
| `dfs_oi` − `count_oi` | −0.0117 / +0.0012 / −0.0085 | −0.0064 | mixed |
| `dfs_expert` − `count_expert` | +0.0001 / −0.0198 / −0.0290 | −0.0162 | mixed |

**§4's "no DFS-superiority claim is supportable" applied to the pre-fix run and
no longer describes the evidence.** DFS now beats `uniform` — the matched
control with an identical parameter set — in all three seeds in both variants.

Three reasons that is still not a claim:

1. **DFS remains last on validation** (0.6590 vs 0.6516/0.6538 for the OI rows,
   0.6727 vs 0.6643/0.6686 for the expert rows). The validation-to-holdout
   inversion now runs in DFS's favour, which warrants more suspicion than
   celebration, not less.
2. **n = 3, no bootstrap.** Three same-signed paired differences is p ≈ 0.25
   under a sign test. §10 requires a paired 95 % temporal interval; this does
   not meet it.
3. **OI still wins outright** (0.7159). No learned row beats the deterministic
   baseline on the holdout.

## 8.3 Robustness, unchanged by the fix

> **Superseded by §9.** The duplicate figures once reported here compared a
> measurement against a definition and confounded duplicate resistance with
> profile weighting. §9 replaces them with a controlled measure.

SSH removal still helps **every** row, including `uniform` and `count` where
unit mass makes the co-location bug impossible as a cause. A second mechanism
is therefore still unidentified — the channel-0 collision (SSH occupies the
TEMP value slot) and attention dilution are the open suspects.

Artifacts: `outputs/godas_final_s{1234,1235,1236}/metrics_seed*.json`.
Note these ran the audit over **6** holdout months; §9 uses all 8, so absolute
scores there are not comparable cell-for-cell with §8.1.

---

# 9. Robustness audits (mentor doc §9)

Run `godas_v3`, commit `5700706`, 6 trainable rows × 3 seeds, evaluated on
**all 8** eligible 2025 holdout source months (§8 used 6, so absolute scores
below are not comparable cell-for-cell with §8.1). Lower is better throughout.

This supersedes the audit reported in §8.3 and in the previous §9. The earlier
duplicate table had two defects, both fixed here:

1. **It compared a measurement against a definition.** `uniform` mass *is* the
   token count, so its "8.00× growth" is arithmetic, not an experimental
   result — it has zero variance across every seed by construction. And the
   mass column measured no model at all: it was identical for the two DFS rows
   because `observation_mass` has no learned parameters.
2. **The output shift confounded two effects.** DFS assigns the copied group
   mass 0.62 where the controls assign 16.0, so a DFS row responds less to
   *any* perturbation of that column, duplicated or not. Half the reported
   advantage was simply lower profile weighting.

The corrected audit runs three conditions on one **pinned** profile column
(cell 19, 13 — previously a fresh random column each month, whose variance
swamped the effect):

| | condition | measures |
|---|---|---|
| A | k=1, unbiased | baseline prediction |
| B | k=1, biased +2 z | **profile sensitivity** = ‖B−A‖ |
| C | k=8, biased +2 z | **duplicate shift** = ‖C−B‖ |

The reported statistic is **duplicate-over-sensitivity = shift / sensitivity**:
how much extra the model moves for eight bit-exact copies, *per unit of how
much it moves for real information in the same place*. Weighting cancels, so
what remains is duplicate resistance alone.

## 9.1 Duplicate over-sensitivity — the headline

Mean over 3 seeds; sd(seed) across seeds, sd(month) across the 8 holdout
months within a seed.

| Row | TEMP | sd(seed) | SALT | sd(seed) |
|---|---|---|---|---|
| `dfs_oi_expert_cbottle` | **0.70** | 0.23 | **1.88** | 0.53 |
| `uniform_oi_expert_cbottle` | 1.34 | 0.36 | 3.74 | 1.34 |
| `count_oi_expert_cbottle` | 1.33 | 0.46 | 4.27 | 2.70 |
| `dfs_expertlocal_cbottle` | **3.17** | 0.28 | **3.34** | 0.81 |
| `uniform_expertlocal_cbottle` | 5.19 | 0.55 | 5.49 | 0.76 |
| `count_expertlocal_cbottle` | 5.20 | 0.54 | 5.41 | 1.07 |

**DFS is lower in 12 of 12 paired comparisons** — 2 variants × 2 channels ×
{uniform, count} × 3 seeds, every seed individually, with no exceptions. The
reduction is 39–48 % against `uniform` on TEMP and 39–50 % on SALT.

The `oi_expert` DFS row is the only row anywhere in the build with a TEMP
ratio **below 1**: eight redundant copies move it *less* than the single
genuine observation does. That is the mechanism doing exactly what §2.2 says
it should.

Interpreting the two variants: the `expertlocal` rows sit 3–5× higher than the
`oi_expert` rows across the board, because the frozen OI background anchors
the prediction and the gated residual can only move it so far. The absolute
level is a property of the variant; the DFS-vs-control gap is what is being
tested, and it holds in both.

## 9.2 Mass growth — reported, not headlined

| Row | copied-group mass k=1 | k=8 | growth |
|---|---|---|---|
| `dfs_*` (both variants) | 0.624 | 2.359 | **3.84× ± 0.08** |
| `uniform_*`, `count_*` | 16.000 | 128.000 | 8.00× ± 0.00 |

DFS grows sublinearly where the controls grow exactly linearly. But the
control column is a definition, so this is a **sanity check that the estimator
is wired in**, not a comparative result — and the honest reading is that 3.84×
is far weaker than the mechanism achieves in isolation.

**The open problem.** The same estimator measured on an isolated duplicate
pair gives 1.08× for eight copies; deployed here it gives 3.84×. Per-token
leverage is ≈0.05 in the live 523-token set against ≈12.5 for a lone token,
and consolidation only bites at high leverage. The likely cause is
**32 random features for ~664 tokens — roughly 20 tokens per feature
dimension**, so the whitened Gram matrix cannot resolve them as distinct.
`N_FEATURES = 32` is fixed by doc §2.2, so this is a design question for
review rather than something to change unilaterally.

## 9.3 Missing-modality — absolute score

| Row | full | no profiles | no surface T/S | no SSH | profiles only |
|---|---|---|---|---|---|
| `dfs_oi_expert_cbottle` | 0.7227 | 0.8874 | 0.7778 | 0.6975 | 0.7410 |
| `uniform_oi_expert_cbottle` | 0.7488 | 0.9453 | 0.8225 | 0.6889 | 0.7285 |
| `count_oi_expert_cbottle` | 0.7413 | 0.9245 | 0.8340 | 0.6920 | 0.7372 |
| `dfs_expertlocal_cbottle` | 0.7172 | 0.8000 | 0.7788 | 0.7033 | 0.7345 |
| `uniform_expertlocal_cbottle` | 0.7346 | 0.8474 | 0.7970 | 0.7023 | 0.7298 |
| `count_expertlocal_cbottle` | 0.7385 | 0.8490 | 0.8070 | 0.7045 | 0.7334 |

## 9.4 Missing-modality — change from `full`

Negative means **removing that stream improved the model**.

| Row | no profiles | no surface T/S | no SSH | profiles only |
|---|---|---|---|---|
| `dfs_oi_expert_cbottle` | +22.8 % | +7.6 % | **−3.5 %** | +2.5 % |
| `uniform_oi_expert_cbottle` | +26.2 % | +9.8 % | **−8.0 %** | −2.7 % |
| `count_oi_expert_cbottle` | +24.7 % | +12.5 % | **−6.6 %** | −0.6 % |
| `dfs_expertlocal_cbottle` | +11.5 % | +8.6 % | **−1.9 %** | +2.4 % |
| `uniform_expertlocal_cbottle` | +15.4 % | +8.5 % | **−4.4 %** | −0.7 % |
| `count_expertlocal_cbottle` | +15.0 % | +9.3 % | **−4.6 %** | −0.7 % |

**Profiles dominate.** Withholding them costs +11 % to +26 %, by far the
largest dependency, and `profiles_only` scores within a couple of percent of
`full` — the gridded streams add little once profiles are present.

**SSH is still a liability, but DFS is hurt least.** Removing SSH improves
every row; the doc expected removal to *cost* about +2 %, so the sign is
opposite for all six. The new detail is the ordering: the DFS rows lose the
least by keeping SSH (−1.9 %, −3.5 %) and the controls lose the most (−4.4 %
to −8.0 %), consistently in both variants. That is what the variable-group fix
(§8) predicts — a mass estimator that can tell SSH from surface T/S
down-weights the redundant stream instead of letting it dilute the pool — but
it is a partial mitigation, not a cure. The residual is unexplained; the open
suspects remain the channel-0 collision (SSH occupies the TEMP value slot,
distinguished only by the mask channel and modality embedding) and plain
attention dilution from 140 extra tokens.

## 9.5 Holdout scores from this run

Mean ± sd over 3 seeds, 8 months.

| Row | Score | TEMP (z) | SALT (z) |
|---|---|---|---|
| `dfs_expertlocal_cbottle` | **0.7172 ± 0.0500** | 0.8206 | 0.9180 |
| `dfs_oi_expert_cbottle` | 0.7227 ± 0.0226 | 0.7918 | 0.9740 |
| `uniform_expertlocal_cbottle` | 0.7346 ± 0.0390 | 0.8232 | 0.9645 |
| `count_expertlocal_cbottle` | 0.7385 ± 0.0369 | 0.8354 | 0.9587 |
| `count_oi_expert_cbottle` | 0.7413 ± 0.0399 | 0.8064 | 1.0073 |
| `uniform_oi_expert_cbottle` | 0.7488 ± 0.0380 | 0.8137 | 1.0186 |

Paired differences (same seed, identical parameter set, mass mode the only
difference):

| Comparison | per seed | mean | verdict |
|---|---|---|---|
| `dfs_oi` − `uniform_oi` | −0.0231 / −0.0047 / −0.0505 | **−0.0261** | DFS better ×3 |
| `dfs_expert` − `uniform_expert` | −0.0033 / −0.0199 / −0.0292 | **−0.0175** | DFS better ×3 |
| `dfs_oi` − `count_oi` | −0.0227 / +0.0048 / −0.0380 | −0.0186 | 2 of 3 |
| `dfs_expert` − `count_expert` | +0.0001 / −0.0191 / −0.0451 | −0.0213 | 2 of 3 |

Consistent with §8.2 and slightly larger with the two extra months. The
caveats there still bind, and one is worth restating: **all six rows selected
step 5000, in all three seeds** (one exception, `count_oi` seed 1235 at 4500).
Validation score was still falling when the budget ran out, so every row is
undertrained and this is a comparison between equally-undertrained models.
5000 steps is fixed by doc §6.2.

Note also that seed spread (sd up to 0.050) is roughly twice the DFS-vs-control
effect (0.018–0.026). Only the **paired** differences are informative here; an
unpaired reading of §9.5's first table would be noise.

## 9.6 Against the doc's §9 gate

The gate asks for duplicate amplification improved by a registered factor
versus both controls, **with a paired interval excluding parity**.

- **Direction and consistency: met.** 12 of 12 paired comparisons favour DFS,
  every seed individually, both channels, both variants.
- **Interval: not met.** n = 3 seeds gives p ≈ 0.125 at best under a sign
  test over seeds. The audit now stores `per_month` values for all 8 months,
  which is the sufficient statistic a paired block bootstrap needs, so §10 can
  compute the interval without re-running training — but it has not been run,
  and no interval is claimed here.

## 9.7 Reproduce

```bash
ROWS=dfs_oi_expert_cbottle,uniform_oi_expert_cbottle,count_oi_expert_cbottle,\
dfs_expertlocal_cbottle,uniform_expertlocal_cbottle,count_expertlocal_cbottle
for s in 1234 1235 1236; do
  CUDA_VISIBLE_DEVICES=$((s-1234)) .venv/bin/python experiments/14_godas_dfs_d4rt.py \
      --configs "$ROWS" --seed $s --steps 5000 --validation-interval 500 \
      --queries 512 --eval-queries 1024 --audit --audit-months 8 \
      --output outputs/godas_v3_s$s
done
```

~48 min per seed on one GPU (3 seeds in parallel on GPUs 0–2). Scores
reproduce to ~2.4e-3 rather than bit-exactly — GPU kernel nondeterminism.

Source: `outputs/godas_v3_s{1234,1235,1236}/metrics_seed*.json`, keys
`missing_inputs` and `duplicate_attack` (the latter now carrying
`profile_sensitivity`, `duplicate_shift`, `duplicate_over_sensitivity`,
`duplicate_over_sensitivity_std`, `pin_cell`, and `per_month`).
Superseded predecessors: `outputs/godas_final_s*/` (§8, 6 months),
`outputs/godas_audit_s*/` (pre-SSH-fix).
