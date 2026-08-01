# DFS-Attention — Evidence Probes (plan Section 11)

*Analytic probes of the evidence estimator: no trained weights are involved, because `tau_i` is a property of the observing geometry at a target scale. Prediction-change columns use randomly initialised fixed-seed models (architectural, not learned, behaviour).*

Protocol target scale: dx = 111 km, dz = 50 m, dt = 30 d. Every DFS below is the **exact** ridge-leverage trace (`k_neighbors=None`) for sets of <= 700 tokens — no localisation error enters them; the larger gridded sets use a 256-token neighbourhood, which is converged for that geometry.


## 1 · Exact duplication

Baseline: **96 tokens, DFS 95.78**. Re-ingesting a measurement must add no evidence and must not move the prediction. ΔDFS and Δŷ are measured against each row's own un-duplicated input (the real-time / delayed-mode row uses a second, independent 24-profile draw).

| case | factor | tokens | DFS | ΔDFS | Δŷ perceiver | Δŷ resampler | Δŷ MBCA | Δŷ DFS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| one level token | ×2 | 97 | 95.78 | +0.00e+00 | 0.0010 | 0.0021 | 0.0003 | 0.0000 |
| one level token | ×4 | 99 | 95.78 | +0.00e+00 | 0.0029 | 0.0062 | 0.0009 | 0.0000 |
| one level token | ×8 | 103 | 95.78 | +0.00e+00 | 0.0064 | 0.0143 | 0.0021 | 0.0000 |
| 6 whole profiles | ×2 | 120 | 95.78 | +7.63e-06 | 0.0022 | 0.0012 | 0.0023 | 0.0000 |
| 6 whole profiles | ×4 | 168 | 95.78 | +0.00e+00 | 0.0046 | 0.0026 | 0.0048 | 0.0000 |
| 6 whole profiles | ×8 | 264 | 95.78 | +1.53e-05 | 0.0066 | 0.0039 | 0.0069 | 0.0000 |
| real-time + delayed-mode | ×2 | 192 | 95.10 | +0.00e+00 | 0.0006 | 0.0003 | 0.0000 | 0.0000 |

## 2 · Same location, different depths

One column; more of its depth bands are supplied. Different depths describe different water, so evidence must grow with the number of bands — the correction at the centre of the plan.

| bands supplied | deepest level (m) | DFS | DFS / band |
|---:|---:|---:|---:|
| 1 | 45 | 0.998 | 0.998 |
| 2 | 186 | 1.997 | 0.998 |
| 3 | 408 | 2.994 | 0.998 |
| 4 | 985 | 3.991 | 0.998 |

## 3–4 · Dense vertical sampling: smooth column vs thermocline

Identical sampling change (10 dbar → 2 dbar, 5× the levels) applied to a vertically smooth anomaly column and to one whose anomaly is concentrated in a sharp thermocline dipole at 120 m, **matched in RMS amplitude** so only vertical structure differs. Tokens are 10 m bands, so vertical density is expressible in the token set at all.

Read the two target blocks together: at a 50 m target the extra levels cannot be used by *either* column — correctly, since the requested field has no 2 dbar structure — while at a 10 m target the thermocline column converts them into evidence and the smooth one still cannot. Redundancy is a property of the column *and* the question being asked.

| target | column | sampling | levels | tokens | DFS | gain vs 10 dbar |
|---|---|---:|---:|---:|---:|---:|
| protocol (Δz = 50 m) | smooth | 10 dbar | 30 | 30 | 7.14 | ×1.00 |
| protocol (Δz = 50 m) | smooth | 2 dbar | 149 | 30 | 8.20 | ×1.15 |
| protocol (Δz = 50 m) | thermocline | 10 dbar | 30 | 30 | 8.42 | ×1.00 |
| protocol (Δz = 50 m) | thermocline | 2 dbar | 149 | 30 | 9.62 | ×1.14 |
| fine (Δz = 10 m) | smooth | 10 dbar | 30 | 30 | 9.35 | ×1.00 |
| fine (Δz = 10 m) | smooth | 2 dbar | 149 | 30 | 10.80 | ×1.15 |
| fine (Δz = 10 m) | thermocline | 10 dbar | 30 | 30 | 17.87 | ×1.00 |
| fine (Δz = 10 m) | thermocline | 2 dbar | 149 | 30 | 19.74 | ×1.10 |

## 5 · Target vertical-resolution sweep (identical input)

The same 2 dbar profile, queried for coarser and finer reconstructions. Retained evidence must rise as the target gets finer.

| target Δz (m) | DFS, smooth column | DFS, thermocline |
|---:|---:|---:|
| 250 | 4.13 | 4.16 |
| 100 | 6.03 | 6.31 |
| 50 | 8.20 | 9.62 |
| 25 | 9.92 | 14.31 |
| 10 | 10.80 | 19.74 |
| 5 | 10.96 | 21.11 |

## 6 · Horizontal cases

| case | profiles | tokens | DFS |
|---|---:|---:|---:|
| uniform (global) | 24 | 96 | 95.78 |
| clustered (2 deg box) | 24 | 96 | 30.69 |
| half the profiles, each twice (new floats) | 24 | 96 | 47.95 |
| half the profiles, each twice (same float) | 24 | 96 | 47.89 |
| 16 profiles, one water mass | 16 | 64 | 23.54 |
| 16 profiles across a front | 16 | 64 | 23.79 |

### Two profiles 5 km apart

The plan's horizontal test case: redundant for a coarse target, genuinely two observations for a 5 km one.

| target | DFS (one profile) | DFS (the pair) | pair / single |
|---|---:|---:|---:|
| coarse (5 deg) | 3.941 | 4.068 | ×1.032 |
| protocol (1 deg) | 3.991 | 5.515 | ×1.382 |
| fine (0.25 deg) | 3.991 | 6.643 | ×1.665 |
| very fine (5 km) | 3.991 | 6.649 | ×1.666 |

### Gridded products (60°×60° field on the 1° analysis grid)

Refining the tokenization of a dense field exposes more of it — but only down to the target scale, where the evidence saturates. That is the principled form of "retokenizing must not multiply a field's influence": it is bounded by what the target can resolve, not by a normalisation convention.

| case | patch (deg) | tokens | DFS | DFS / token |
|---|---:|---:|---:|---:|
| 20x20 cells/patch | 20 | 9 | 8.02 | 0.891 |
| 10x10 cells/patch | 10 | 36 | 32.07 | 0.891 |
| 5x5 cells/patch | 5 | 144 | 128.27 | 0.891 |
| 2x2 cells/patch | 2 | 900 | 657.84 | 0.731 |
| 1x1 cells/patch | 1 | 3600 | 1011.02 | 0.281 |
| two overlapping products (5x5) | 5 | 288 | 224.10 | 0.778 |

One 5×5 product alone scores 128.27; adding a second, independently processed product of the same ocean does not double it.


---
*Generated by `experiments/23_dfs_evidence_probes.py` in 73.3 s.*
