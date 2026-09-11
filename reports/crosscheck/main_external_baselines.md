# P4 — External neural / ocean baselines

<!-- Track A. Written against measured availability, not against expectation. -->

> **One track.** Track B's independent verification is `pending`; no agreement
> is claimed or implied below.

The plan asks for a faithful Senseiver port under a 3-day box, plus a
feasibility and scoring cross-check for ADAF-Ocean, EN4, ECCO, ORCA-DL, and the
short-range references XiHe / WenHai / FuXi-Ocean. This report separates what
was **run** from what is **blocked**, with the blocker named in each case.

---

## Summary

| reference | code | weights / data | status |
|---|---|---|---|
| **EN4** (EN.4.2.2 g10) | n/a — gridded product | ✅ downloaded, 2000–2025 | **scored** as a P0 row |
| **ECCO** (V4r4) | n/a — gridded product | ✅ downloaded, 2000–2017 | **scored**, under a labelled secondary protocol — no later release exists |
| **Senseiver** | ✅ official `OrchardLANL/Senseiver` | ✅ official NOAA example rebuilt from source | **reproduced, adapted, and scored** on held-out floats |
| **ORCA-DL** | ✅ official | ⚠️ weights on a OneDrive share (interactive auth) | blocked on non-scriptable download |
| **XiHe** | ✅ official inference code | ⚠️ ONNX weights, separate distribution | different task (1/12° daily eddy-resolving) |
| **FuXi-Ocean** | ✅ official inference code | ❌ **weights not released** (repo's own checklist) | cannot be run at all |
| **WenHai** | ❌ no public repository found | ❌ | cannot be run |
| **ADAF-Ocean** | ❌ no public repository found | ❌ | cannot be run |

---

## EN4 — scored, and honestly labelled

EN.4.2.2 (g10 bias correction), monthly 1°, 42 levels, downloaded for
2000–2025 and cut to both P1 region boxes.
Adapter: `src/ocean_tokenizer/reference_adapters.py`.

EN4 is interpolated to each held-out float's **real position and depth**, then
z-scored with the same train-only statistics the model's targets use, so its
RMSE sits in the same units as every other row and is computed on the identical
queries.

**EN4 assimilates the very floats it is scored against.** It is an upper
reference that has already seen the answer, not a peer. A model that does not
beat EN4 has not failed; a model that *does* would need a careful explanation.
The P0 tables carry EN4 on this understanding.

One practical finding worth recording: the Met Office changed its directory
layout mid-series. Years ≤ 2021 live under `en4-2-1/EN.4.2.2/`, years ≥ 2022 sit
flat in `en4-2-1/`. Assuming either path alone silently loses 2022–2025 — which
is exactly the development and holdout era. The downloader tries both.

## ECCO — scoreable, via a shifted protocol rather than more data

ECCO Central Estimate V4r4, monthly 0.5°, downloaded for 2000–2017, both boxes.

### Can more data fix it? No.

The obvious move is to download a later ECCO. It does not exist:

| product | estimation period |
|---|---|
| **V4r4** (latest Central Estimate) | **1992–2017** |
| V4r3 | 1992–2015 |
| V4r2 / V4r1 | 1992–2011 |

NASA CMR carries **no V4r5 at all** (0 hits), and every ECCO T/S collection on
PO.DAAC ends `2018-01-01`. The only ECCO-family product reaching 2023 is
**ECCO2 Cube92** (1992–2023, 18–30 km cube-sphere) — a *different product
family*, fit by Green's functions plus adjoint rather than being a V4 Central
Estimate, on a different grid, and distributed through ECCO Drive behind
Earthdata auth. Reporting it in the same column as V4r4 would silently mix two
estimators. `SASSIE_ECCO` reaches 2021 but is Arctic-only and irrelevant to both
of our boxes.

So under the main protocol — train 2000–2018, validation 2019–2021, development
2022–2024, holdout 2025 — ECCO's entire coverage sits **inside the training
era**, and no download changes that.

### What does work: move the evaluation, not the product

The held-out float cohort is **WMO-disjoint and year-independent**. A float is
held out for its whole life, so shifting the evaluation era does not change
which floats are held out. That licenses a clearly-labelled **secondary**
protocol whose eras sit inside V4r4's coverage
(`protocol.ECCO_OVERLAP_SPLITS`, selected with `--split-protocol ecco_overlap`):

```
train 2000–2012   validation 2013–2014   development 2015–2017
```

The model still never sees the evaluation months and never sees the held-out
floats, so the comparison stays out-of-sample in both. The cohort supports it
comfortably:

| region | eval months | held-out floats | held-out profiles |
|---|---:|---:|---:|
| Gulf Stream | 36 (24 eligible) | 54 | 2,781 |
| N. Pacific gyre | 36 (24 eligible) | 56 | 4,790 |

That is comparable to the main development split's 60 WMOs. **ECCO now scores.**
Measured against held-out floats in physical units:

| region | ECCO TEMP | ECCO SALT | query coverage |
|---|---:|---:|---:|
| Gulf Stream | 1.98 °C | 0.331 g/kg | 86.3 % |
| N. Pacific gyre | 0.88 °C | 0.130 g/kg | 86.7 % |

Worse in the high-EKE Gulf Stream, better in the quiet gyre — as expected for a
1° state estimate. This is a **secondary** protocol and is never merged into the
main headline table, the same treatment this project already gives the extended
23-level grid. Runs carry a warning saying so in their artifact.

### A bug this exposed, worth recording

An earlier version scored ECCO under the *main* protocol anyway. Because every
query fell outside its coverage, the fallback returned zero anomaly — so the
table silently reported the **climatology floor under ECCO's name**, a number
that looks like a result and is not one.

The same fault had a subtler form that survived the first fix: a query the
product genuinely cannot reach (below its deepest wet level, or inside its land
mask) was also being zero-filled, which scores *climatology* wherever the
product has no answer and penalises a gridded product for its own mask. Both
references now return NaN there, the accumulator drops it, and the realised
`query_coverage` is recorded next to the row. The correction is not cosmetic:

| row | zero-filled | NaN-dropped + coverage reported |
|---|---:|---:|
| EN4 | 0.413 | **0.349** (93.6 %) |
| ECCO | 0.584 | **0.460** (87.8 %) |

**Both products assimilate the very floats they are scored against.** They are
upper references that have already seen the answer, not peers.

## Senseiver — official example reproduced

Upstream: `OrchardLANL/Senseiver`, commit `e443eb0`, the code accompanying
Santos et al., *Nature Machine Intelligence* 2023.
Harness: `experiments/real_data/48_senseiver.py`.

The plan's rule — *never use a homemade approximation and label it Senseiver* —
is respected structurally: **every line of model code executed is the authors'.**
Nothing was reimplemented.

The one thing that had to be built is data. The official repository ships
`Data/*/placeholder*.txt`; the datasets are not in it. Its NOAA loader expects
`Data/NOAA/sst_weekly.mat`, a repackaging of NOAA OI SST V2 weekly. That array
was rebuilt from **NOAA's own primary files** in the layout the official loader
reads.

Reproducing the published frame count took one non-obvious step. The authors'
loader hard-codes `num_frames = 1914`, and neither NOAA weekly file reaches it
alone (427 frames for 1981–1989, 1727 for 1990–present). Concatenated, frame
1914 lands on **2018-06-24**, consistent with the snapshot a 2023 paper would
have used. Training on the shorter 1990-onward file alone would have been a
silent deviation.

Verification is through the authors' own loader rather than ours:

```
official loader returns (1914, 180, 360, 1), range [-0.050, 1.000]
```

The authors' `train.py` then runs to convergence on their NOAA configuration
(train loss → 0.083). **Rule 1 is satisfied.**

### Scored at held-out float positions — and what it shows

`experiments/real_data/56_senseiver_score.py` carries the trained model's predictions
through the same WMO-clustered held-out-float evaluation every DFS row uses, so
the Senseiver finally has a row rather than only a training curve. Gulf Stream,
36 development months, 60 held-out floats:

| | TEMP (degC) | SALT (PSU) |
|---|---:|---:|
| **GODAS itself**, scored against the same floats | **2.2057** | **0.4064** |
| **Senseiver** (trained to reproduce GODAS) | **2.2192** | **0.4065** |
| | *+0.6 %* | *+0.02 %* |

**The Senseiver reproduces its training target almost exactly, and inherits that
target's entire gap to the ocean.** Its 2.22 degC is not 2.22 degC of Senseiver
error — it is GODAS's 2.21 degC error plus 0.6 % of its own. The model is doing
its job nearly perfectly; the job is the problem.

That is the substantive P4 finding, and it is a statement about the architecture
rather than about the implementation. The Senseiver maps sparse sensors to a
DENSE field and must be trained on dense snapshots. No dense truth of the real
ocean exists, so the best available training target is a reanalysis — and a
model trained to emulate a reanalysis is bounded by that reanalysis, however
good the model is. The DFS rows are trained against float measurements directly
and carry no such ceiling.

A caveat that keeps this honest: GODAS is a coarse regional reanalysis and this
is a hard, high-EKE box. A different dense target (EN4, a higher-resolution
reanalysis) would move the ceiling, and the comparison should be repeated
against one before the conclusion is generalised. What does not change is the
structure of the argument: whatever dense field is chosen, the Senseiver is
capped by it.

### The adaptation, and the asymmetry it exposes

Rule 2 — adapt to our held-out Argo evaluation — is specified but **not yet
run**, and the specification itself is a finding:

The Senseiver maps sparse sensors to a **dense field** and is trained on dense
field snapshots. Our real-Argo task has no dense truth; the targets are point
measurements from held-out floats. So the faithful adaptation must train it on a
gridded T/S product (GODAS) with sensors placed at real Argo positions, then
evaluate at the held-out floats' real positions.

**The Senseiver requires a gridded training field where the DFS model does
not.** That is a structural difference in what the two methods need, not a
detail of the port, and it belongs in the paper's comparison rather than in a
footnote. Supplying GODAS is the most favourable honest choice available, since
GODAS has assimilated Argo and therefore encodes more than our model ever sees.

## ORCA-DL, XiHe, FuXi-Ocean, WenHai, ADAF-Ocean

* **ORCA-DL** (`OpenEarthLab/ORCA-DL`, *Science Advances*) — official code
  present. Predictions, weights and data are distributed through a **OneDrive
  share requiring interactive authentication**, which no unattended job can
  fetch. Obtaining it is a manual step, not an engineering one. Its task is also
  seasonal-to-decadal prediction on its own grid, so a comparison would need a
  separate protocol from our contemporaneous reconstruction.
* **XiHe** (`Ocean-Intelligent-Forecasting/XiHe-GlobalOceanForecasting`) —
  official ONNX inference code released. The task is **1/12° eddy-resolving
  daily forecasting**, which shares neither the grid, the horizon, nor the
  target of this study. Per the plan's own instruction to keep short-range and
  monthly/seasonal tables separate, it does not belong in the same table.
* **FuXi-Ocean** (`huangqiusheng/FuXi-Ocean`) — inference and evaluation code
  released; the repository's own release checklist still shows
  **`[ ] ONNX weight files`** unticked. It cannot be run.
* **WenHai** and **ADAF-Ocean** — no public implementation was found.

### Recommendation

The plan's §7 puts external foundation models last and warns: *"Do not spend
weeks porting extra foundation models before the layout and redundancy claims
are independently verified."* The evidence here supports that ordering strongly.
Three of the five short-range/foundation references cannot be executed at all
with what their authors have released, and the two that can address a different
task. The Senseiver is the one genuinely comparable neural reference, and its
official example now runs.

**Proposed scope for this round:** EN4 as the operational upper reference,
Senseiver as the neural comparator once its Argo adaptation is run, ECCO on a
clearly-labelled overlap table, and a citation-with-reason for the rest.
