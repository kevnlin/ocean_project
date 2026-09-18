# Replacing the Perceiver-IO backbone — options, and the first result

*Kevin's action from the 2026-09-17 meeting. The trial implementation lives in
`src/ocean_tokenizer/setconv.py`; the arms are in `reports/real_data/ablation_ladder.md`.*

## What the current trunk does not have

`D4RTFusion` is a Perceiver-IO: profiles become five depth-band tokens each, the
month is squeezed through 32 unaddressed latent slots, and a query goes looking
for what it needs by attention. Two structural facts of this problem are absent
from that design and have to be learned from 264 monthly samples:

* **Locality.** The answer at a point depends mostly on observations near it —
  measured here, the anomaly decorrelates in 50-150 km. The model's own locality
  priors start at 3 500 × 5 600 km (the refiner Gaussian) and 398 km (the finest
  coordinate wavelength), both far wider than the field.
* **Translation equivariance.** The map from "a float 100 km north-east" to "the
  answer here" is the same map everywhere in the box. Attention over free latent
  slots has to learn it separately for each part of the region.

GAOT (spatially addressed anchors) was the first attempt at fixing this and did
not separate from the Perceiver backbone beyond seed spread
(`reports/real_data/main_table2.md`).

## Candidates

| candidate | what it buys | cost / risk | reference |
|---|---|---|---|
| **SetConv → U-Net → bilinear query** (ConvCNP family) — *implemented* | locality and translation equivariance by construction; off-grid in and out; no token bottleneck; levels stay separate | regional grid fixes the finest resolution; needs a grid per region | Gordon et al., ICLR 2020; Vaughan et al., GMD 2022; Andersson et al. 2023 (ConvGNP / DeepSensor); Allen et al., *Nature* 2025 (Aardvark, raw observations) |
| **Translation-equivariant transformer neural process** (relative-position attention, pseudo-tokens) | keeps attention and set-valued input, adds the invariance the current trunk lacks | O(N²) unless pseudo-tokens; more moving parts | Ashman et al., ICML 2024 |
| **Graph encoder–processor–decoder on a mesh** | natural for scattered observations; multi-scale by mesh refinement | heavier; edge construction is a design surface of its own | GraphCast; GraphDOP (ECMWF, obs-to-obs) |
| **Learned / flow-dependent covariance on top of OI** (deep kernel, 4DVarNet-style) | attacks the *actual* gap: the classical baseline is already at the ceiling for a stationary covariance | changes the framing from "reconstruction network" to "learned assimilation" | Fablet et al. (4DVarNet); NeurOST |
| **kNN local attention** (the current refiner, done properly) | smallest change; fixes locality without a new trunk | still no translation equivariance | Point Transformer family |

## The first result

`SetConvUNet` is size-matched to the registered model (408 792 vs 405 063
parameters), trained on the same months, same cap, same seeds, same fixed
held-out set (2 seeds; `backbone_setconv`, and `backbone_setconv_all` with the
cap removed).

| arm | Gulf Stream test TEMP (z) | J | N. Pacific gyre | J |
|---|---:|---:|---:|---:|
| baseline (Perceiver-IO, cap 128) | 1.1822 | 0.977 | 0.8677 | 0.836 |
| GAOT anchors | 1.1622 | 0.961 | — | — |
| SetConv-UNet (cap 128) | 1.1601 | 0.959 | 0.8894 | 0.856 |
| **SetConv-UNet, every profile** | **1.1445** | **0.946** | 0.8815 | 0.849 |
| kriging OI, same cap | 1.1560 | 0.956 | 0.8451 | 0.814 |
| kriging OI, every profile | 1.1197 | 0.926 | 0.8261 | 0.796 |

Read this honestly: in the Gulf Stream the new backbone is the best learned arm
and is the only one that converts extra profiles into accuracy (the Perceiver
trunk gets **worse** when given every profile). In the gyre it is slightly worse
than the baseline, where anomalies are large-scale and locality buys nothing.
**Neither backbone passes the kriging OI when both see the same profiles.**

On the overfit ladder it behaves like the Perceiver trunk (0.040 z / 0.064 °C
memorising one month), so capacity is not what separates them and any held-out
difference is inductive bias, not fitting power.

## The thing the backbone cannot fix

The audit's kriging OI reaches **J 0.956 (cap) / 0.926 (all profiles)** in the
Gulf Stream with zero trained parameters. Every learned row sits at or behind
that. A different backbone can close the gap to the interpolator; it cannot, on
its own, get past it, because **profiles-only reconstruction of a monthly
anomaly is close to a linear-Gaussian problem** — the optimal estimator *is* a
covariance, and the covariance is fitted well.

Two ways past, in order of expected value:

1. **Give the model information the OI does not have.** The main tables are
   deliberately profiles-only, which is exactly the configuration where a
   cross-modality method cannot show cross-modality value. Satellite SST, SLA
   and SSS are already in the repo (`data/real_obs_1deg.zarr`) and are the
   standard inputs in this literature (OSnet, NeSPReSO, TS-Cast, ADAF-Ocean).
   **Caveat to plan around:** that store covers **2016-2023 only**, so it does
   not reach the `recent3` test years; this arm needs either a satellite-era
   split (train 2016-2020, validate 2021, test 2022-2023) or an extended
   download.
2. **Make the covariance non-stationary.** Kriging assumes one covariance
   everywhere in the box. The Gulf Stream's is anisotropic, front-following and
   state-dependent. A network that conditions the effective covariance on the
   surface state is doing something a stationary OI provably cannot — and it is
   a claim that can be demonstrated directly by comparing against the fitted-OI
   ceiling region by region.

**Result of (1), run on the satellite-covered years** (train 2016-2020,
validate 2021, test 2022-23; Gulf Stream; 2 seeds; same 409 K size):

| arm | test TEMP (z) | J |
|---|---:|---:|
| SetConv-UNet, profiles only | 1.0662 | 0.975 |
| kriging OI, profiles only, every profile | 1.0015 | 0.916 |
| **SetConv-UNet + altimetry (SLA) only** | **0.9162** | **0.838** |
| **SetConv-UNet + SST/SLA/SSS** | **0.8883** | **0.813** |

The backbone question turns out to be second order; the modality question is
first order. The gain is largest in the deepest band, and altimetry alone — a
source that assimilates no in-situ profile, so it cannot leak a held-out float
back — carries most of it.

Both routes are testable with what is already in the repo, and both give the
paper a claim that survives the OI baseline.
