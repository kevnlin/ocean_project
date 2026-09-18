# Performance gap, novelty risk, and where to submit

*Prepared for the group after the 2026-09-17 meeting. Every number comes from a
file in this repo: `reports/real_data/pipeline_audit.md` (the audit),
`overfit_sanity.md` (the sanity test), `ablation_ladder.md` (one switch at a
time, plus the satellite-era arms), and the published `main_tables.md` /
`main_table2.md`.*

---

## 0. The four sentences

1. **The model can fit.** Memorising one month reaches 0.017 z = **0.025 °C**
   and the copy test 0.013 z, both across two seeds and two regions. Capacity,
   tokenisation and the decoder work — which is also why the bigger model did
   not help.
2. **A zero-parameter kriging OI matches or beats it** on the identical held-out
   floats, and beats every learned row once it may use the profiles the
   128-cap discards. On profiles alone there is almost nothing left for a
   network to add.
3. **The reason is measurable, not mysterious.** The anomaly decorrelates in
   50-150 km; the median held-out target sits 148 km from its nearest input; and
   the irreducible point-scale variance puts a floor of **J ≈ 0.5-0.65** on any
   method. RMSE ≈ 1 z is not a bug, it is the unit: 1 z = 1.6-2.1 °C.
4. **The one configuration that clearly beats the interpolator adds satellite
   data**: J **0.813** (all surface fields) and **0.838** (altimetry alone)
   against **0.916** for the best profiles-only method on the same split. The
   gain is largest at depth and survives with altimetry only, which never sees a
   float. That is the paper.

---

## 1. The performance gap

### 1.1 The baseline that was missing

A kriging OI — covariance fitted on the training years, nothing trained — run on
the identical held-out floats, months and input draws as Table 1:

| region | method | TEMP RMSE (z) | TEMP J | SALT RMSE (z) | SALT J |
|---|---|---:|---:|---:|---:|
| gulfstream | DFS (ours, GAOT, 406 K, 12 k steps) | 1.1624 | 0.961 | 1.1540 | 0.991 |
| gulfstream | **kriging OI, same 128 profiles** | **1.1560** | **0.956** | **1.1363** | **0.976** |
| gulfstream | **kriging OI, all profiles of the month** | **1.1197** | **0.926** | **1.1060** | **0.950** |
| npac_gyre | DFS (ours, GAOT) | 0.8814 | 0.849 | 0.8153 | 0.875 |
| npac_gyre | **kriging OI, same 128 profiles** | **0.8451** | **0.814** | **0.8199** | 0.880 |
| npac_gyre | **kriging OI, all profiles** | **0.8261** | **0.796** | **0.8067** | **0.866** |

The climatology reference reproduces the published value exactly (1.2098 /
1.1644 Gulf Stream, 1.0384 / 0.9314 gyre), so this is the same evaluation.

**Every reviewer in this field will ask for this baseline**, and the one
currently in the tables does not stand in for it: `Causal OI` is a
kernel-weighted mean with no covariance inversion, GODAS-era length scales
(≈330 × 220 km) and a background shrinkage switched off by the ~10⁻⁶ noise
values it is fed — which is why it scores J = 1.004, worse than climatology, and
flatters every learned row above it.

### 1.2 What the numbers mean physically

| region | channel | 0-100 m | 100-300 m | 300-700 m | 700-1400 m |
|---|---|---:|---:|---:|---:|
| gulfstream | TEMP | 1.89 °C | 1.66 °C | 1.66 °C | 0.80 °C |
| gulfstream | SALT | 0.457 PSU | 0.270 | 0.234 | 0.083 |
| npac_gyre | TEMP | 1.03 °C | 0.95 °C | 0.49 °C | 0.12 °C |
| npac_gyre | SALT | 0.208 PSU | 0.151 | 0.041 | 0.032 |

The current model is at **≈1.96 °C / 0.38 PSU** (Gulf Stream) and **≈0.68 °C /
0.10 PSU** (gyre) against climatology at 2.05 °C / 0.39 PSU and 0.88 °C /
0.14 PSU.

**Retire the 0.1-0.2 °C target for held-out floats.** It came from the synthetic
CESM2 era, where truth was a smooth 1° model field (0.158 °C against a 0.55 °C
floor). Measured on the real cohort, two *different* floats in the same month at
the same place correlate only 0.62-0.79 in the upper ocean (and 0.38 at
700 m in the gyre), so even a perfect co-located
observation leaves ~25-50 % of the variance unexplained. The fitted nugget puts
the floor at:

| region | channel | mean nugget | J floor |
|---|---|---:|---:|
| gulfstream | TEMP | 0.36 | **0.60** |
| gulfstream | SALT | 0.37 | 0.61 |
| npac_gyre | TEMP | 0.40 | 0.63 |
| npac_gyre | SALT | 0.37 | 0.61 |

Report skill against that floor, not an absolute RMSE.

### 1.3 Two different ceilings, and which one binds

* **What the given profiles allow.** The optimal linear estimate from the floats
  the model is handed is J ≈ 0.956 (cap 128) / 0.926 (all profiles). The model
  is at 0.961. **Headroom on profiles alone: ~0.03 J.**
* **What the ocean allows.** J ≈ 0.60. The difference between 0.93 and 0.60 is
  information the profiles do not contain.

So: better architecture buys at most a few hundredths; more information buys
tenths. The experiment below confirms it.

### 1.4 The experiment that changes the outlook

Secondary split (train 2016-2020, validate 2021, test 2022-23 — the years
`data/real_obs_1deg.zarr` covers), Gulf Stream, 2 seeds, all arms size-matched
at ~409 K and trained identically:

| arm | test TEMP (z) | test TEMP (°C) | test J TEMP | test J SALT |
|---|---:|---:|---:|---:|
| Perceiver-IO, profiles only | 1.0840 | 1.957 | 0.992 | 0.996 |
| SetConv-UNet, profiles only | 1.0662 | 1.914 | 0.975 | 0.973 |
| kriging OI, profiles only, same cap | 1.0376 | 1.849 | 0.949 | 0.947 |
| kriging OI, profiles only, every profile | 1.0015 | 1.781 | 0.916 | 0.918 |
| **SetConv-UNet + altimetry (SLA) only** | **0.9162** | **1.664** | **0.838** | **0.858** |
| **SetConv-UNet + satellite SST/SLA/SSS** | **0.8883** | **1.604** | **0.813** | **0.842** |

Same months, same floats, same size, two seeds: the **backbone swap alone
changes almost nothing** (0.992 → 0.975 J, within a seed spread of ~0.017),
while **the extra modality moves J from 0.916 to 0.813** — six times the seed
spread, and the first configuration in this project that clearly beats the
interpolator.

Two details that make it credible rather than a leak:

* The gain is **largest at depth** (−20 % at 700-1400 m against −15 % at
  0-100 m). That is the steric-height mechanism — sea level integrates the
  column — not SST copied down to the 5 m level.
* The **altimetry-only control** (SLA, no SST/SSS) delivers J 0.838 of the 0.813
  total. Altimetry assimilates no in-situ profile, so the result cannot be the
  L4 SST/SSS analyses handing back the floats they ingested.

**The main tables are profiles-only by design** ("adding them would change more
than the fusion rule"). That decision is defensible for isolating the fusion
rule, and it is also why the cross-modality method has never been evaluated in a
configuration where cross-modality can show. Fix that and the project has a
result.

### 1.5 Why the DFS-vs-uniform ordering flipped

It did not flip in any measurable sense. The evidence estimate is nearly flat at
this density (τ CV 0.08-0.17 — the estimator itself reports the observations as
almost independent), the arms differ by ~0.01 J, and the three-seed spread is
±0.005-0.07 J. Two weeks reporting opposite signs is what noise of that size
does. Either evaluate the mechanism where redundancy exists (multi-source
inputs, real-time/delayed-mode pairs, dense regions, or the synthetic OSSE where
density is a knob) or stop reporting the ordering as a result.

### 1.6 What the ablations say (17 arms x 2 seeds, `ablation_ladder.md`)

Gulf Stream, held-out 2023-24, baseline 1.1822 z (J 0.977), seed spread ±0.012:

| switch | Δ vs baseline (z) | J |
|---|---:|---:|
| SetConv backbone **+ every profile** | **−0.038** | 0.946 |
| refiner length scales set to 150 km / 100 m | **−0.033** | 0.950 |
| all audit fixes stacked | **−0.032** | 0.947 |
| refiner output gate 0.05 → 1.0 | **−0.031** | 0.951 |
| SetConv backbone (cap 128) | −0.022 | 0.959 |
| no target dropout | −0.022 | 0.959 |
| GAOT backbone | −0.020 | 0.961 |
| 8 months per optimiser step | −0.019 | 0.962 |
| no global latent / uniform mass / count mass | −0.013 … −0.016 | 0.964-0.967 |
| level tokens, input QC, exact-position climatology | −0.004 … −0.009 | 0.970-0.971 |
| **every profile, without the locality fixes** | **+0.010** | 0.985 |

Three things to take from it.

**The numbers are not insensitive — they were being moved in the wrong place.**
The fusion-rule switches (DFS / uniform / count) sit inside the seed spread,
exactly as they do in the published tables. The switches that make the model
*local* move four times as much.

**More data only helps a model that can use it locally.** Handing the baseline
every profile makes it **worse** (+0.010), while the same extra profiles are
worth −0.030 to the kriging OI and −0.038 to the SetConv backbone, which is
local by construction. The 128-cap is not the whole story: non-locality is.

**The gyre behaves differently and that is physical.** There only
`level_tokens` (−0.023) and `fixed_stack` (−0.014) beat the baseline; locality
fixes do nothing, because the gyre's anomalies stay correlated out to 300-400 km
(§4 of the audit) and there is little local structure to resolve.

With the locality fixes the model finally reaches the interpolator at matched
inputs (1.1490 vs kriging's 1.1560 at cap 128; 0.8448 vs 0.8451 in the gyre) but
still does not pass it when both are given every profile (1.1445 vs 1.1197).
Nothing here approaches the 0.10 J that the satellite channels buy.

---

## 2. Novelty risk against prior cross-modality work

| candidate contribution | prior art | status after the audit |
|---|---|---|
| Shared latent + coordinate query decoding | Perceiver IO; ADAF-Ocean (neural-process mapping over multi-source ocean obs, arXiv 2511.06041, Nov 2025); GraphDOP | **Blocked.** Adopted; cite, do not claim. |
| Sparse obs → dense ocean field with a CNN/U-Net | CLOINet, ReconMOST, TS-Cast (Ocean Science 2026), attention-enhanced 3-D U-Net++ (ESSD 2026) | **Blocked.** Crowded and active. |
| Satellite surface → subsurface T/S | OSnet, NeSPReSO, Buongiorno-Nardelli, TS-Cast | **Blocked as a task.** Our version is only novel in *how* (see below), not *that*. |
| Token-multiplicity / partition-invariant attention (MBCA) | No direct competitor found for observation fusion | **Novel as a property, unsupported as a benefit.** The proof and the unit test stand; at Argo density it changes nothing measurable. |
| DFS evidence weighting by ridge leverage | DFS is standard in variational DA (ECMWF observation-influence diagnostics); the novelty is using it as an attention prior | **Weak.** Inert in this regime for the same reason. |
| Evaluation protocol: WMO-disjoint held-out floats, float-clustered CIs, EN4/ECCO as references that have already seen the answer, a fitted-OI ceiling | Most of the field scores against gridded reanalyses that assimilated the floats, without an OI baseline | **The strongest asset**, and what makes §1.1 and §1.4 checkable. |

**Where a defensible claim now lives:** not "our fusion rule is better", but
*"how much of the subsurface anomaly is recoverable from each observing system,
measured against a fitted-OI ceiling on a protocol where the truth is a float
the model never saw — and the answer is that profiles alone are nearly
exhausted by linear interpolation, while altimetry adds a measurable
tenth of J at depth."* That is a real contribution and the audit supplies the
evidence for it.

Two risks to manage explicitly: **(i)** ADAF-Ocean is the closest ocean
precedent and is recent — position against it, not around it; **(ii)** J values
near 1 read as a weak result when reported bare, and as an honest measurement
when reported with the floor and the OI baseline. The difference is what is
measured alongside.

---

## 3. Venue recommendations

**1. JAMES (AGU) or AIES (AMS) — recommended, submit this quarter.** Both take
methods-plus-evaluation work, both gold open access, neither requires beating
the state of the art to publish a careful, well-instrumented result. The audit,
the ceiling, the protocol, the OI comparison and the satellite contrast are a
complete paper. AIES if the framing is "what ML can and cannot recover from the
Argo array"; JAMES if it is "a reconstruction system and its evaluation".

**2. NeurIPS 2027 Datasets & Benchmarks (deadline ~June 2027).** The regional
cohorts, the WMO-disjoint split, the reference rows (climatology, persistence,
kriging OI, EN4, ECCO) and the clustered-CI harness are a benchmark others could
use. Here the protocol *is* the contribution.

**3. ICML 2027 — abstract 16 Jan 2027, paper 22 Jan 2027.** Worth it only with a
positive methods result by mid-January: the satellite-informed model beating the
fitted OI across both regions and several seeds, with the mechanism isolated
(altimetry-only control, depth-band breakdown). On current evidence that is
reachable, but it needs the full-protocol version, not the 60-month satellite
split.

**4. Nature Communications / Nature Machine Intelligence — not yet.** Both want a
demonstrated advance with reach. Revisit if the satellite result holds globally
and full-depth.

**Note on ICLR:** ICLR 2027's abstract deadline is **18 September 2026**, papers
**25 September 2026** — days away, not missed. Not realistic for this work now,
but the meeting's premise was wrong.

**In parallel:** a 4-page workshop paper (Climate Change AI / ML for Earth System
Modeling) on the ceiling measurement and the OI comparison puts the result in
front of the right audience months before a journal decision.

---

## 4. What to do next, in order

1. **Replace `Causal OI` with the kriging OI** in every table, and report J
   against the nugget floor as well as against climatology.
2. **Run the satellite arm on the full protocol.** This needs satellite fields
   before 2016 (or accept the 2016-2023 era as a documented secondary protocol
   and say so).
3. **Remove the 128-profile cap** (kriging says J 0.956 → 0.926; the learned
   arms agree) and make the local paths local (refiner length scales, gate,
   coordinate rescaling).
4. **Fix the target**: input QC, climatology at the profile's own position,
   robust per-level scaling.
5. **Then** revisit the backbone, and only afterwards the evidence weighting, in
   a regime that contains redundancy to correct.
