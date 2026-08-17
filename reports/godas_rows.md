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
