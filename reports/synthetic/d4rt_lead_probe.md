# D4RT causal space-time query decoder — lead-0..3 probe on protocol_v1

Run date: 2026-08-16 | Branch: `intern/d4rt-query-decoder`
Spec: [docs/superpowers/specs/2026-08-16-d4rt-query-decoder-design.md](../docs/superpowers/specs/2026-08-16-d4rt-query-decoder-design.md)
Mentor source: `dfs_d4rt_intern_plan.md` §2.3, §2.4, §6.2

---

## 0. Provenance warning

`dfs_d4rt_intern_plan.md` is written in the past tense as a finished audit but
**was never run** (confirmed with its owner 2026-08-16). Its §7/§8/§9 tables,
§6.3 checkpoint hashes, §4 manifest SHA and "106 passed in 3.08s" are targets,
not measurements. **Nothing in this report reproduces them and no number here
may be compared against them.** The GODAS system those sections describe does
not exist in this repository, on any branch, in any commit.

Everything below is a fresh measurement on the CESM2-LE / protocol_v1 line.

## 1. What was run

Mentor §2.3/§2.4 built at the document's stated dimensions, on the existing
protocol_v1 OSSE rather than GODAS. Out of scope and not built: §2.2 set-level
DFS with random Fourier features, §2.5 causal OI, §2.6 OI residual and cBottle
loss, §§4–6 GODAS, §9 audits, §11 raw Argo.

| | |
|---|---|
| Architecture | width 64, 4 heads, 2 latent + 2 decoder blocks, 32 latents, 32 obs + 8 reference slots (§2.3 verbatim) |
| Query | `(lat, lon, depth, t_src, t_tgt)`; lead added via zero-init embedding with `padding_idx=0` |
| Local refiner | length scales `(0.35, 0.35, 0.30, 3 mo)`, mass exponent 1, gate init 0.05, chunk 128 (§2.4 verbatim) |
| Parameters | 403,015 |
| Context | `[t_src-1, t_src]` surface/SSH, profiles at `t_src` only |
| Training | 5000 steps, 512 queries/sample, AdamW 3e-4 (wd 0.01), validation every 500 (§6.2) |
| Selection | mean over lead 0..3 × {TEMP,SALT} of RMSE / val climatology RMSE (§6.2) |
| Splits | protocol_v1: 276 train / 36 val / 12 pinned test months |
| Seeds | 1234, 1235, 1236 |
| Evaluation | 12 pinned test months, full unobserved pool — **9.30 M queries per lead** |

Recorded deviations from §6.2, also emitted into each run JSON as
`deviations_from_mentor_6_2`: **batch size 1, not 2** (the `fullrun`
observation path is batch-1 and widening it interacts with the DFS evidence
solve); **plain masked MSE, not cBottle loss** (§2.6 is out of scope).

## 2. Training: §6.2's budget undertrains this domain

| Seed | Selected step | Val score |
|---|---|---|
| 1234 | 5000 | 0.7640 |
| 1235 | 5000 | 0.7547 |
| 1236 | 5000 | 0.7967 |
| **mean ± std** | | **0.772 ± 0.022** |

**All three seeds selected the final step, and every one of the 10 validations
in every seed set a new best.** The runs stopped because they hit the step
budget, not because they converged.

The mechanism is domain size. §6.2's budget is 512 queries × 5000 steps =
2.6 M query-samples. On GODAS's 38 × 26 × 16 ≈ 15.8 k-cell regional grid that
is ~165 samples per cell; on protocol_v1's 180 × 360 × 20 ≈ 1.3 M-cell global
grid it is **2 samples per cell**. The domain is ~82× larger and the budget was
not scaled for it.

For calibration, a discarded run at 8192 queries × 22000 steps (~70× the
query-samples) reached val score **0.5774** before it was stopped — versus
0.772 here. That run is not otherwise usable (it predated the §6.2 config fix
and saved no checkpoint), but it bounds what the same architecture reaches with
more compute.

## 3. Lead evaluation — 12 pinned test months, 3 seeds

Unobserved-only anomaly RMSE in physical units, mean ± sample std over seeds
1234/1235/1236. Observation draw is identical across leads (fixed by the
`t_src` profile sample), so all four leads score on the same cells.

### TEMP (°C)

| Lead | Model | Persistence | Climatology | vs clim | vs pers |
|---|---|---|---|---|---|
| 0 | 0.3692 ± 0.0191 | 0.3692 ± 0.0191 | 0.5521 ± 0.0001 | **−33.1 %** | — |
| 1 | 0.3887 ± 0.0171 | 0.3889 ± 0.0171 | 0.5611 ± 0.0001 | **−30.7 %** | −0.07 % |
| 2 | 0.4068 ± 0.0136 | 0.4080 ± 0.0136 | 0.5594 ± 0.0001 | **−27.3 %** | −0.29 % |
| 3 | 0.4308 ± 0.0083 | 0.4344 ± 0.0090 | 0.5589 ± 0.0002 | **−22.9 %** | −0.83 % |

### SALT (PSU)

| Lead | Model | Persistence | Climatology | vs clim | vs pers |
|---|---|---|---|---|---|
| 0 | 0.0932 ± 0.0024 | 0.0932 ± 0.0024 | 0.1305 ± 0.0000 | **−28.6 %** | — |
| 1 | 0.0931 ± 0.0023 | 0.0932 ± 0.0022 | 0.1302 ± 0.0000 | **−28.5 %** | −0.06 % |
| 2 | 0.0937 ± 0.0020 | 0.0939 ± 0.0019 | 0.1292 ± 0.0000 | **−27.5 %** | −0.29 % |
| 3 | 0.0958 ± 0.0018 | 0.0962 ± 0.0016 | 0.1299 ± 0.0000 | **−26.3 %** | −0.40 % |

Baselines: **persistence** = the model's own lead-0 reconstruction reused as
the forecast; **climatology** = zero anomaly, floor recomputed per target
month. The lead-0 climatology floor of 0.5521 °C / 0.1305 PSU matches
protocol_v1's documented 0.552 °C / 0.131 PSU, which is a useful check that the
evaluation is scoring in the right space.

## 4. Result

**The model clearly beats climatology at every lead** — 33 % better at lead 0
falling to 23 % at lead 3 for temperature, 29 % to 26 % for salinity — with
clean monotone degradation in lead. Reconstruction (lead 0) at 0.369 °C is
consistent with a 403 k-parameter model trained for 5000 steps.

**The model does not meaningfully beat persistence.** Spec §6 set the probe's
positive result as "a lead-1 forecast that beats persistence". The sign is
consistent — the paired per-seed difference is negative at every lead in all
three seeds:

| Lead | seed 1234 | 1235 | 1236 |
|---|---|---|---|
| 1 | −0.00059 | −0.00002 | −0.00021 |
| 2 | −0.00210 | −0.00058 | −0.00085 |
| 3 | −0.00481 | −0.00217 | −0.00385 |

— but the magnitude is **0.07 % at lead 1**, roughly 300× smaller than the
between-seed spread. Read plainly: the lead-conditioned query has learned
almost nothing beyond re-emitting its own lead-0 reconstruction. The margin
grows with lead (0.07 % → 0.83 %), which is the direction a real forecast
signal would move, but it is far too small to call a positive result.

Following mentor §10's own rule, no significance language is attached to that
ordering: there is no paired month-block bootstrap, and the saved JSONs hold
aggregate error sums rather than per-month sufficient statistics, so one cannot
be computed from them retrospectively.

**Verdict: the capability exists and is correct; the forecast skill is not yet
demonstrated.** Leads 1–3 are decodable from one checkpoint by changing a
single number in the query, with query independence guaranteed
architecturally. Whether that capability carries genuine forecast information
beyond persistence is unresolved, and §6.2's budget — which all three seeds
show is truncated on this domain — is the first thing to rule out.

## 5. Limitations

1. **Undertrained.** All three seeds selected their final step. Any conclusion
   about forecast skill is provisional until a converged run exists.
2. **No bootstrap intervals.** Aggregate query errors are not independent;
   per-source/lead/level error sums must be saved before intervals can be
   computed. This is the same gap mentor §10 flags.
3. **Batch size 1, not §6.2's 2.**
4. **Loss is masked MSE, not cBottle** (§2.6 out of scope).
5. **OSSE only.** CESM2-LE plays the role of truth; this is not real-ocean
   validation.
6. **DFS evidence is not lead-aware.** An observation at `t_src` is genuinely
   weaker evidence about `t_src+3`, but `dfs.py` was deliberately left
   untouched to avoid entangling this work with the paper line. A plausible
   contributor to the flat persistence margin.

## 6. Reproduce

```bash
# train (per seed)
CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python experiments/34_d4rt_lead_train.py \
    --seed <1234|1235|1236> --steps 5000 --queries 512 --val-every 500 \
    --min-steps 0 --tag d4rt_m62_s<seed>

# evaluate (per seed)
CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python experiments/35_d4rt_lead_eval.py \
    --ckpt outputs/ckpt/d4rt_m62_s<seed>.pt --seed <seed> \
    --n-profiles 1500 --tag d4rt_lead_eval_m62_s<seed>
```

Artifacts: `outputs/ckpt/d4rt_m62_s{1234,1235,1236}.pt`,
`outputs/cache/d4rt_m62_s*.json` (training curves + selection),
`outputs/cache/d4rt_lead_eval_m62_s*.json` (lead results).
Wall clock: ~18 min training per seed, ~108 min evaluation per seed.

## 7. Next

1. **Settle the budget question.** Rerun at a query/step budget scaled to the
   global grid and re-evaluate. If the persistence margin stays ~0 at
   convergence, that is a real negative result about query-side lead
   conditioning and worth reporting as such.
2. **Save per-source sufficient statistics** so paired month-block bootstrap
   intervals can be computed; without them no ordering here can carry
   significance.
3. **Consider lead-aware evidence** (§5.6) if the margin stays flat.
