# Monday 2026-08-11 group meeting — slide content

*One slide per phase, following the plan's §1.1 checklist. Written 2026-08-09.
Every number below is **3 seeds (1234/1235/1236), mean ± std**, under
**protocol_v1**, unless the slide says otherwise — so the plan's honesty rule
("label every number 1-seed") no longer applies to anything here.*

**Where the numbers live** is stated on each slide, so any question can be
answered by opening one file.

---

## Slide 0 — Phase 0: environment, data, reproduction

**Say:** the qualification bar is met, in a stronger form than the plan asked.

* Data verified in place: `cesm2_le_full_standard.zarr`, dims `time 3012 ×
  depth 60 × 180 × 360`, TEMP/SALT/SST/SSS/MASK present.
  *(No 42 GB transfer was needed — we are already on the box that hosts it.)*
* `config.ROOT` is now overridable via `OCEAN_ROOT`, so the tree runs on another
  machine without editing source.
* **Reproduction:** the certified U-Net checkpoint reproduces its cached test
  RMSE to **1.0e-06 (TEMP) / 3.1e-07 (SALT)**.

**The one thing to flag:** the plan's target was "reproduce 0.1492 / 0.0312
±0.005". Those are **week-2** numbers. Under the frozen **protocol_v1** the
certified model is **0.1580 / 0.0325** — and we match it to 1e-06, not ±0.005.

| | |
|---|---|
| Numbers from | `outputs/cache/oi_vs_unet_seed1234.json` → `certified_reproduction` |
| Also | `outputs/cache/audit_depthwise_e40.json` → `test` (the cached reference) |
| Test suite | 160 passed (`python -m pytest tests/ -q`) |

---

## Slide 1 — Phase 1 【M1】: Optimal Interpolation. **Does our method beat OI?**

**Say:** yes — and it still wins when the information is made identical.

The repo had **no OI**. `predict_nearest` is a distance-gated fill with no
covariance model, so it cannot handle observation redundancy. `oi.py` implements
Bretherton et al. (1976) level-wise in z-scored anomaly space, sharing the
anomaly target, normalisation and scoring mask with the U-Net.

Tuned on **validation** months (protocol_v1 rule): **L = 500 km, γ = 0.1,
k = 10** — interior optimum on a smooth convex surface. Repeating the sweep on
*training* months returns the **identical** optimum → not fitted to one split.

| method | TEMP (°C) | SALT (PSU) | N. Atlantic TEMP |
|---|---|---|---|
| Climatology floor | 0.5520 | 0.1305 | 0.6459 |
| Nearest-profile fill | 0.9670 | 0.3119 | 1.3265 |
| **Optimal interpolation** | **0.2287** | **0.0528** | **0.3139** |
| U-Net, `profiles_only` (like-for-like) | 0.1829 | 0.0489 | — |
| U-Net, full (certified) | 0.1580 | 0.0325 | 0.1938 |

* **Full system beats OI by +30.9 % (TEMP) / +38.4 % (SALT).**
* **Like-for-like, +20.0 %** — same information, nothing more. So ~⅔ of the
  margin is a better interpolator, ~⅓ is the extra modalities.
* Margin is **larger** in the North Atlantic (+38.3 %) than globally → not a
  global average hiding a regional failure.
* **Every method sees byte-identical profiles**: we replay the certified run's
  RNG (312 discarded draws) rather than merely matching settings.

**The finding to say out loud:** *OI beats the pointwise MLP* (0.2287 vs
0.2983). A properly tuned classical baseline outperforms a learned pointwise
model — and the shared-latent variants (~0.52) sit **well below OI**. An OI row
belongs in any table claiming a learned method is useful, including the paper's.

| | |
|---|---|
| Report | `reports/oi_baseline.md` (headline, bands, region, figures) |
| Tuning | `reports/oi_tuning.md` (4×3 grid, k-sweep, stability check) |
| Numbers | `outputs/cache/oi_vs_unet_seed1234.json`, `oi_tuning_val.json`, `oi_tuning_train.json` |
| `profiles_only` row | `outputs/cache/ssh_ablation_s*.json` → `results.profiles_only` (3 seeds: 0.1814 ± 0.0021) |
| Figures | `reports/fig_oi_rmse_bars.png`, `reports/fig_oi_vs_unet_error_map.png` |

---

## Slide 2 — Phase 2: profile density. **Where does TEMP cross 0.1 °C?**

**Say:** measured, not extrapolated — **≈3900 profiles/month**.

| profiles/month | TEMP (°C) |
|---:|---|
| 1500 | 0.1492 ± 0.0017 |
| 3000 | 0.1122 ± 0.0010 |
| **4000** | **0.0991 ± 0.0012** ← already below target |
| **6000** | **0.0829 ± 0.0016** |

* Crossing at **N ≈ 3919**; the seed spread is ~12× smaller than the gap between
  density points, so it is well resolved, not noise.
* SALT was already inside the advisor's 0.04–0.05 target at *every* density.

**Include the correction** (it is the intellectually honest part): before
running, I fitted the whole 100–3000 curve (α = 0.269) and called it the
*conservative* reading against the plan's tail slope (α = 0.410), predicting a
crossing at 6143. **The plan was right and I was wrong.** Local α keeps rising —
0.151, 0.254, 0.340, 0.410, 0.432, **0.441** — so the last two segments are the
steepest of the whole curve; returns are still accelerating at 6000. The lesson:
with monotonic local slopes, use the tail slope; a global fit is not
"conservative", it is biased by a regime the question is not about.

**Reality anchor:** real Argo ≈ 4000 floats → ~12 000 profiles/month, but very
unevenly distributed. Our sampling is uniform-random, so this is **not** "real
Argo already delivers 0.1 °C" — the uniform requirement sits at roughly ⅓ of
today's real volume, and the gap is about **coverage**, not count.

| | |
|---|---|
| Report | `reports/density_4000_6000.md` (§2 answer, §3 the correction) |
| Numbers | `outputs/cache/density_powerlaw.json` (fit, local slopes, crossings) |
| Raw | `outputs/cache/density_ablation_seed{1234,1235,1236}.json` |
| Figure | `reports/fig_density_powerlaw.png` |

---

## Slide 3 — Phase 3: architecture / normalization / tokenization spec

**Say:** this is the document the advisor asked for, and the paper's method
section should be lifted from it rather than re-derived.

`doc/architecture_spec.md` — every claim carries a `file:line` link:

* **A. Data pipeline** — raw → zarr → `CommonGrid` → `prepare_month`, with a
  dataflow diagram and the meaning of every key in the sample dict.
* **B. Normalization (the core section)** — physical clipping → train-only
  monthly climatology → anomaly → **per-variable, per-depth** z-scores, with
  every formula. Key consequence: `z = 0` recovers the climatology, which is
  *why* the floor is a principled reference. NaN → value 0 **+ a finite flag**,
  never a silent zero.
* **C. Tokenization** — 68-dim `coord_features` shared verbatim by encoders and
  the query decoder; per-modality encoders and their token counts.
* **D. Model & loss** — depthwise U-Net (469 858 params), masked MSE in
  z-anomaly space, training hyperparameters.
* **E. Final vision** — modality encoders → shared latent → coordinate query
  decoder, marked *exists* vs *missing*.
* **F. Four transferable ideas from cBottle** (the assigned reference).

**The number worth putting on the slide:** at 1500 profiles the WOA prior
contributes **~10 800 tokens** against **6 000** profile tokens. Under plain
softmax attention the prior can outvote the observations — that is precisely
what the fusion-rule comparison exists to study.

| | |
|---|---|
| Spec | `doc/architecture_spec.md` (§A–F + an appendix of corrections to the plan) |
| cBottle note | `reports/reading_cbottle.md` (grounded in the fetched abstract) |

---

## Slide 4a — Phase 4a: the SSH modality. **Pre-registered, and confirmed.**

**Say:** there was no SSH anywhere in the pipeline. We built one, wrote the
hypothesis down *before* training anything, and both parts held.

Built as **TEOS-10 steric (dynamic) height** from the T/S fields, referenced to
990 dbar; 360 monthly fields. Characterised *before* any model ran:

* coverage **86.7 %** of ocean · anomaly std **2.89 cm**
* corr with **SST** anomaly **0.470** · corr with **100–300 m T** anomaly
  **0.708** → it is a **thermocline proxy**, not a restatement of SST.

| TEMP (°C), 3 seeds | full | 0–100 m | **100–300 m** | 300–max |
|---|---|---|---|---|
| control `profiles_woa_surf` | 0.1572 ± 0.0010 | 0.1566 | 0.1923 | 0.0830 |
| **+ SSH** | **0.1368 ± 0.0002** | 0.1405 | **0.1632** | 0.0722 |
| relative gain | **+13.0 %** | +10.3 ± 1.0 % | **+15.1 ± 0.8 %** | +13.0 ± 2.0 % |

* **H1 confirmed** (+13.0 %). Per-seed deltas −0.0220 / −0.0191 / −0.0200 —
  every seed shows it, the smallest ~16× the control's own spread.
* **H2 confirmed** — the 100–300 m thermocline gains most, on both the absolute
  and relative reading, and leads at *every individual seed*.
* **For scale:** one derived channel buys what ~1500 → ~2700 profiles/month
  would. A satellite field is far cheaper than doubling the float array.

**Do not overclaim.** The pseudo-SSH is computed **from** the same T/S the model
reconstructs. 13.0 % is an **upper bound** on real altimetry. The defensible
claim is *"a vertically integrated surface constraint helps, most in the
thermocline"* — testing real ADT is a Phase-5 job.

*(Unplanned, flagged as an observation not a finding: the SSH arm's seed spread
±0.0002 is 5× tighter than the control's ±0.0010. Three seeds is thin evidence
for a variance claim.)*

| | |
|---|---|
| Report | `reports/ssh_ablation.md` (§3 hypothesis, unedited since before the run; §5 results) |
| Numbers | `outputs/cache/ssh_ablation_s{1234,1235,1236}.json` |
| Modality diagnostics | `outputs/cache/ssh_dyn_meta.json` |
| Code | `src/ocean_tokenizer/ssh.py`, `experiments/{28_make_ssh,29_ssh_ablation}.py` |

---

## Slide 4b — Phase 4b: the fusion core. **⚠️ Not what the plan expected**

**Say plainly:** the plan has this slide as "design on paper, training not
feasible in 3 days, lands in W2–W3". **That is out of date.** It already exists,
was trained at full scale in week 4, and **missed its acceptance bar**.

* `src/ocean_tokenizer/fusion.py` — four comparable fusion variants (standard
  Perceiver / fixed-budget resampler / MBCA / DFS-Attention).
* Plan's bar: *"pass = beat the joint U-Net (0.1787)"*. Reality under
  protocol_v1: shared latent **~0.52** vs joint U-Net **0.1948** — i.e. barely
  below the climatology floor.
* **Diagnosed cause** (week-4 report): with only 276 training fields, exact-column
  observations uniquely fingerprint each month, so reconstruct-the-field training
  collapses into **month-identity recall**. Every masking level that destroys the
  fingerprint also destroys the signal. Checkpoints are validation-selected near
  the peak, which is protocol-clean but not competitive.
* So the open task is **not** "implement the fusion core" — it is **"fix the
  training dynamics"**: more training members/years, field-space decoding, or
  local-attention decoders.

**Why it still matters:** the shared latent's reason to exist is the flexibility
axis — one checkpoint across observation densities, off-grid query coordinates
(needed for real Argo in Phase 5), and a natural home for evidence weighting. It
has to earn that by first matching the U-Net's accuracy. Today it does not.

**Concrete next step on this slide:** add the SSH channel as a third
`GridPatchEncoder` stream (+540 tokens/month) — on Slide 4a's evidence, the
fusion core is where a thermocline constraint should pay off most.

| | |
|---|---|
| Week-4 results | `reports/full_training_report.md` (§1 headline, training-dynamics finding) |
| Code | `src/ocean_tokenizer/fusion.py` |
| Context in spec | `doc/architecture_spec.md` §E (exists-vs-missing table) |

---

## Slide 6.1 — Phase 6.1: future prediction via D4RT-style space-time queries

**Say:** this is the *roadmap*, not a result. No numbers claimed.

Our decoder already takes `(lat, lon, depth, month)`. D4RT (arXiv:2512.08924,
CVPR 2026 best paper) uses a query that can probe *any point in space and time*.
Adding a **target time** upgrades ours from 3-D to 4-D, at which point three
tasks become the same operation with a different query:

| query | task |
|---|---|
| `t_tgt = t` | reconstruction / reanalysis (today) |
| `t_tgt > t` | **forecast — with no separate forecasting head** |
| off-grid `(lat, lon, depth)` | **super-resolution — with no second model** |

Minimal and backward-compatible: extend `coord_features` with a normalised lead
`Δt`, held **exactly zero** for all current training so existing behaviour is
bit-identical. Then mix 70 % `t_tgt = t` / 30 % future queries per step.

**The honest bar:** the probe is positive only if lead-1 beats **persistence**
(reusing this month's reconstruction) — not merely climatology.

Note the competing design worth naming: cBottle-style **masked-frame
conditioning** puts time on the *input* side; the D4RT format puts it on the
*query* side. Both avoid autoregressive rollout drift. The choice is empirical.

*(Stays within "query-side conditioning" — does not touch the current paper's
claims; the claim surgery deleted "temporal" and it stays deleted.)*

| | |
|---|---|
| Design | `doc/architecture_spec.md` §E.1 |
| Competing design | `reports/reading_cbottle.md` §3.4 |

---

## Backup slide — corrections to the plan (have this ready, don't lead with it)

The advisor may be working from the plan's numbers, which describe a **week-2
snapshot**. The tree is at week 4 with DFS-Attention merged. Eleven conflicts
are tabulated in `reports/intern_week1.md` §0. The four most likely to come up:

| Plan says | Reality |
|---|---|
| Split 312/12, floor **0.535 / 0.125** | protocol_v1: 276/36/12 pinned, floor **0.5523 / 0.1305** |
| Best = **0.1492 / 0.0312** | Certified: **0.1580 / 0.0325** |
| `fusion.py` to be created | Exists, trained, **missed its bar** (Slide 4b) |
| Tune OI on training months | protocol_v1: hyperparameter selection uses **validation** |

| | |
|---|---|
| Full table | `reports/intern_week1.md` §0 (all 11) |
| Spec appendix | `doc/architecture_spec.md` — final section |

---

## Two process points to raise, not bury

1. **The Sunday check-in did not happen in the intended order.** The plan
   (§3, Task 1.4) says: show the tuning table + unit tests to the senior student
   *before* touching the 12 test months. The test months were scored before that
   review took place. Nothing leaked — tuning used validation months only, and
   the test months were touched exactly once — but the gate was not observed as
   written, and it is better to say so than to let it pass silently.

2. **What is still 1-seed or missing.** Everything on these slides is 3 seeds
   except: the OI row itself (deterministic given its samples; only the profile
   draw varies), and the `mlp` / `unet_joint` rows at densities 4000/6000, which
   were not run — the density curve is fitted on `unet_depthwise`.

---

## Slide count

Seven core slides (0, 1, 2, 3, 4a, 4b, 6.1) + one backup = **8**, inside the
plan's "5–8 slides" budget. If it needs cutting to 5: keep **1, 2, 4a, 4b** and
fold Phase 0 into a single line on Slide 1; Slide 3 becomes "spec is written,
here is the link"; Slide 6.1 becomes a closing bullet.
