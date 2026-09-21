# Performance gap, novelty risk, and where to submit — global reconstruction

*Prepared for the group after the 2026-09-17 meeting. Setup: global ocean state
reconstruction — one domain over the whole ocean, every Argo profile a month
delivers (~8,900 in the test years), 20 levels to 985 m, WOA23 monthly anomaly
target, train 2016-2020 / validate 2021 / test 2022-23, truth = held-out
WMO-disjoint floats. Every number comes from `reports/real_data/pipeline_audit.md`,
`overfit_sanity.md` and `ablation_ladder.md`; two seeds per trained arm.*

---

## 0. Five sentences

1. **The model can fit.** Memorising one global month reaches **0.068 z = 0.080 °C
   / 0.010 PSU** (8 k steps), and the copy test 0.086-0.092 z. Capacity,
   tokenisation and the decoder work, which is also why the bigger model did not
   help.
2. **The "numbers ≈ 1" symptom is seed-dependent.** On seed 1235, 6 of the 12
   arms that keep the registered refiner initialisation (a 3,500 km Gaussian with
   a 0.05 gate) end near the climatology (J 0.96-0.97), including the baseline and
   the PhCA-style backbone; the same arms reach J 0.86-0.88 on seed 1234. The
   arms that start the refiner local or open its gate reach J 0.86-0.87 on both
   seeds.
3. **DFS does not beat uniform mass in the Perceiver** (−0.0002 z on the seed
   where neither run collapsed); count mass is worse (+0.020 z).
4. **The PhCA-style (LNO) fuse ties the Perceiver-IO fuse** (−0.0013 z where
   neither collapsed) and collapses on the same seed. The fuse stage is not the
   bottleneck.
5. **The reachable target is a skill score, not 0.1 °C.** Point-scale variance
   no other float sees puts a floor of J ≈ 0.55-0.69 on any method; 1 z is
   1.24 °C / 0.26 PSU in the upper 100 m.

---

## 1. The performance gap

### 1.1 Where the model stands

| | TEMP z | TEMP J | TEMP (°C) | SALT J |
|---|---:|---:|---:|---:|
| climatology | 1.0638 | 1.000 | 1.216 | 1.000 |
| **baseline** (Perceiver-IO + DFS + D4RT), seed 1234 / seed 1235 | 0.9264 / 1.0172 | 0.871 / 0.956 | 1.105 (mean) | 0.906 / 0.989 |
| Perceiver-IO + uniform mass, seed 1234 / seed 1235 | 0.9262 / 0.9584 | 0.871 / 0.901 | 1.067 (mean) | 0.905 / 0.946 |
| PhCA-style fuse + DFS, seed 1234 / seed 1235 | 0.9251 / 1.0365 | 0.870 / 0.974 | 1.115 (mean) | 0.905 / 1.004 |
| PhCA-style fuse + uniform mass, seed 1234 / seed 1235 | 0.9756 / 0.9988 | 0.917 / 0.939 | 1.123 (mean) | 0.974 / 0.916 |
| baseline + refiner gate 1.0 (both seeds) | **0.9149** | **0.860** | 1.035 | 0.900 |
| baseline + audit fixes stacked (both seeds) | 0.9154 | 0.865 | **1.003** | **0.868** |

### 1.2 Why "≈ 1", physically

One z unit is the anomaly's own standard deviation: **1.24 °C / 0.26 PSU** in
the upper 100 m, 0.84 °C / 0.10 PSU at 300-700 m. A model at 1 z predicts the
training climatology. The 0.1-0.2 °C target came from the synthetic CESM2 era,
where truth was a smooth 1° model field; at a real held-out float it would need
J ≈ 0.1. Measured on the cohort:

| abs(lat) band | mesoscale length L1 | nugget (TEMP) | J floor |
|---|---:|---:|---:|
| 0-20° | 125 km | 0.44 | 0.67 |
| 20-45° | 75 km | 0.33 | 0.58 |
| 45-90° | 75 km | 0.31 | 0.55 |

With every profile the median held-out target sits **125 km** from its nearest
input (only 18 % within 50 km). **Report J against the floor, retire the
absolute target.**

### 1.3 What the pipeline gave away (audit, `pipeline_audit.md`)

1. **The refiner's initialisation** — 3,500 km length scale, 0.05 gate. The arms
   that change it (`refiner_local`, `refiner_gate1`, `fixed_stack`) did not
   collapse on either seed.
2. **Bad salinity passes Argo QC**: 3,290 profiles beyond 10σ (max |z| 685);
   test-year input salinity z-RMS 1.55 with them, 0.99 without. The robust QC arm
   improves salinity J 0.906 → 0.876 on the seed where neither run collapsed.
3. **A per-month cap costs skill globally**: the trained arm at cap 1000 is
   worse by 0.039 z.
4. **Four depth-band tokens, mean-pooled**, discard 19-20 % of a profile's
   vertical variance.
5. **Climatology at the 1° cell centre** adds 0.20 °C of spurious anomaly near
   the surface (1.7 % of variance).
6. **Batch size 1** — the training loss swings; validation was still falling at
   12 k steps in the arms that did not collapse.

### 1.4 Why DFS vs uniform "flipped"

Here the ranking is decided by which run collapsed, not by the mass rule: on
seed 1235 the DFS baseline collapsed and the uniform run did not. On the seed
where neither collapsed, DFS and uniform are identical in the Perceiver
(−0.0002 z). Inside the PhCA-style fuse, DFS beat uniform by 0.05 z on seed
1234 — but that rests on one seed, because the DFS run collapsed on seed 1235.
**Stabilise the refiner first, then re-ask the mass-rule question with more
seeds.**

---

## 2. Novelty risk against prior work

| candidate contribution | prior art | status |
|---|---|---|
| Shared latent + coordinate query decoding | Perceiver IO; ADAF-Ocean (arXiv 2511.06041); GraphDOP | **Blocked.** Cite, do not claim. |
| Position-only cross-attention into a fixed latent, decoding at arbitrary coordinates | **LNO** (Wang & Wang, NeurIPS 2024), **Transolver** (ICML 2024), Slot Attention (NeurIPS 2020) | **Blocked.** Our fuse is PhCA-*style*: LNO's position-only attention with Slot-Attention normalisation. Say so. |
| Sparse obs → dense ocean field (CNN/U-Net/attention) | CLOINet, ReconMOST, TS-Cast (2026), 3-D U-Net++ (ESSD 2026) | **Blocked.** Crowded. |
| DFS evidence conserved through slot competition | DFS is standard in variational DA; slot-softmax conservation is new as a combination | **Novel as a property, unsupported as a benefit** — the one positive sign (PhCA + DFS vs uniform) is a single seed. |
| Token-partition invariance (MBCA) | no direct competitor found | **Novel as a property**, no measurable benefit at Argo density. |
| **Evaluation protocol**: WMO-disjoint held-out floats, nugget floor, two-seed paired deltas | most of the field scores against gridded reanalyses | **The strongest asset.** |

**LNO claims, checked against the paper** (arXiv 2406.03923): it "reduces the
parameter count by an average of 30% and memory consumption by 50% compared to
Transolver across all benchmarks, also with a 1.8× training acceleration" —
verbatim. Its Darcy relative L2 is **0.49 vs Transolver 0.58** (×10⁻², Transolver
reproduced by the authors), and it is best on 4 of 6 forward benchmarks, not all.
None of that transfers here: our wall time is dominated by the DFS neighbour
search over ~30 k tokens, and the PhCA-style runs took 4.1 h vs 3.9 h.

---

## 3. Venue recommendations

1. **JAMES (AGU) or AIES (AMS) — recommended.** The audit, the floor
   measurement, the protocol and the paired ablations are a complete, honest
   evaluation paper. Neither venue requires beating the state of the art.
2. **NeurIPS 2027 Datasets & Benchmarks (~June 2027).** The global cohort, the
   WMO-disjoint split, the climatology reference and the paired-seed harness are
   a benchmark others could use.
3. **ICML 2027 — abstract 16 Jan, paper 22 Jan 2027.** Only with a positive
   methods result by mid-January: DFS or the PhCA-style fuse ahead of its
   reference (uniform mass, the Perceiver fuse) stably across ≥ 3 seeds. Not
   supported by current evidence.
4. **Nature Communications / Nature Machine Intelligence — not yet.**

ICLR 2027 closed with its abstract deadline on 18 September 2026 (papers are due
25 September, but only for registered abstracts).

---

## 4. What to do next, in order

1. **Make the refiner local by default** (length scales ~150 km / 100 m, or gate
   1.0) and rerun the main tables with ≥ 3 seeds and paired deltas.
2. **Report J against the nugget floor**, with RMSE in °C / PSU beside z.
3. **Robust input QC** — salinity is where it pays globally.
4. **Use every profile**; drop the cap.
5. Only then return to the mass rule (DFS vs uniform) and the fuse stage
   (Perceiver vs PhCA-style), with enough seeds to resolve differences of 0.01 z.
6. For headroom beyond interpolation, the information has to come from somewhere
   the profiles do not reach. That is the open question, and it is outside this
   comparison, which is strictly the two fuse stages and the mass rule.
