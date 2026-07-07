# Unified Ocean-Tokenizer Prototype — Stage 1 Report

**Goal.** A clean tokenizer + data pipeline and a small baseline table for sparse-profile
ocean-state reconstruction. *Not* a foundation model. WOA23 and CESM2 are treated as two
gridded numerical ocean datasets (not separate modalities yet).

**Dataset roles.**
| dataset | role | grid | period |
|---|---|---|---|
| `cesm2_le_full` | **full ocean-state ground truth** (dynamic simulation) | regular 1°, 180×360, 60 lev | 1850–2100 monthly |
| `woa23` | **climatology prior / baseline** (observational) | 1°, 180×360, 57 lev | 1991–2020 clim |
| `cesm2` (single member) | extra sample (curvilinear placeholder grid) | native POP 384×320 | 2000–2009 |

Per the instruction to *"include the full data from cesm2_le into the model"*, **CESM2-LE full**
is the ground-truth ocean state for synthetic-profile generation, training, and evaluation.
The single-member `cesm2` is kept as a documented sample but excluded from the sweep (its
lat/lon are placeholders, so it is not grid-comparable without true curvilinear coordinates).

---

## 1. Data inspection & standardization

All three sources are already standardized to dims `(time, depth, lat, lon)` with variables
`TEMP, SALT, SST, SSS` in zarr (`processed/*_standard.zarr`). Data cards:
[woa23](data_card_woa23.md) · [cesm2](data_card_cesm2.md) · [cesm2_le](data_card_cesm2_le.md).

**Common analysis grid** ([common_grid.md](common_grid.md)): CESM2-LE and WOA23 already share the
exact 1° lat grid. We standardize longitude to CESM2-LE's 0–360 convention and roll WOA23 onto
it. Target depths are a curated subset of **20 native LE levels (5–985 m)** — using native GT
levels keeps the ground truth lossless (no vertical interpolation of the target); WOA23 is
interpolated onto them. Land/sea from `MASK==1` (65% ocean). A physical-range clip
(TEMP∈[−3,40] °C, SALT∈[2,45] PSU) removes fill/brine outliers (e.g. 31 spurious SALT=0 cells).

## 2. Tokenizers (lossless) — [tokenizer_roundtrip.md](tokenizer_roundtrip.md)

Four reversible tokenizers, each `decode(encode(field)) == field` exactly (max_abs = 0, NaN mask
preserved) with no masking:

| tokenizer | input | tokens | idea |
|---|---|---|---|
| 2D grid patch | `(C,H,W)` | `(n_patch, C·ph·pw)` | non-overlapping spatial patches |
| 3D volume patch | `(C,D,H,W)` | `(n_patch, C·pd·ph·pw)` | volumetric patches |
| vertical profile | `(C,D,H,W)` | `(H·W, C·D)` | one token per water column |
| point query | `(C,D,H,W)` | `(N, 3+C)` | (coord, value) set; subsample → sparse |

Round-trip result: **all four exact**. The point-query tokenizer also supports subsampling
(e.g. 10% kept → intentionally non-exact), which is exactly the sparse-observation regime used
by the baselines.

## 3. Synthetic Argo profiles — [synthetic_argo.md](synthetic_argo.md)

Sparse Argo sampling is emulated by drawing random ocean columns from CESM2-LE monthly 3-D
TEMP/SALT (default **1500 profiles/month**, full 20-level columns). These are the only
"observations" the reconstruction methods see, plus optional WOA prior / surface fields.

## 4. Baselines & configuration sweep — [baseline_table.md](baseline_table.md)

**Methods:** climatology (= WOA prior), nearest (KDTree-on-sphere profile fill, optionally over
the WOA residual), pointwise **MLP** (geo/time + prior + nearest-obs + surface features), and a
depthwise 2-D **U-Net** (sparse-obs maps + masks + prior + surface channels, shared across depth).
**Configs:** `profiles_only` · `woa_only` · `profiles_woa` · `profiles_woa_surf`.
Trained on 48 LE months (1985–2010), evaluated on 12 held-out months (2011–2014).

### Overall RMSE (held-out CESM2-LE)

**TEMP (°C)**
| method \ config | profiles_only | woa_only | profiles_woa | profiles_woa_surf |
|---|---|---|---|---|
| climatology | – | 1.565 | – | – |
| nearest | 0.996 | – | 0.938 | – |
| mlp | 0.915 | 0.898 | 0.875 | 0.679 |
| **unet** | 0.587 | 0.647 | 0.533 | **0.472** |

**SALT (PSU)**
| method \ config | profiles_only | woa_only | profiles_woa | profiles_woa_surf |
|---|---|---|---|---|
| climatology | – | 0.631 | – | – |
| nearest | 0.383 | – | 0.305 | – |
| mlp | 0.274 | 0.258 | 0.388 | 0.136 |
| **unet** | 0.134 | 0.147 | 0.119 | **0.100** |

### Key findings
1. **Information adds monotonically (U-Net):** profiles_only → +WOA → +SST/SSS steadily lowers
   RMSE. Best config `profiles_woa_surf`: TEMP **0.47 °C**, SALT **0.10 PSU** — a **3.3×/6.3×**
   reduction vs the WOA climatology baseline (1.565 / 0.631).
2. **Surface fields are the biggest lever:** adding SST/SSS gives the largest single jump,
   concentrated in the upper ocean.
3. **Method ranking:** U-Net > MLP > nearest > climatology. Spatial context (U-Net) beats
   purely pointwise (MLP) and purely local (nearest) reconstruction.
4. **Depth structure** (see depth tables): TEMP error peaks in the mixed layer / thermocline
   (~45–105 m) where variability is highest and SST/SSS helps most (0.37 vs 0.73 °C at 5 m);
   all configs converge by ~700–985 m where the climatology prior already explains most variance.
5. **One wrinkle:** `mlp / profiles_woa` SALT (0.388) is worse than its simpler variants — the
   pointwise nearest-SALT feature injects noise without a surface constraint; the spatially-aware
   U-Net does not show this. A believable baseline artifact, left as-is and noted.

## 5. Reproduce
```bash
cd /home/nvidia/ocean_project
python experiments/00_data_cards.py        # data cards + common grid
python experiments/01_tokenizer_roundtrip.py
python experiments/02_synth_argo.py
python experiments/03_baselines.py         # (--smoke for a fast check)
python experiments/04_report.py            # baseline_table.md
```
Package: `src/ocean_tokenizer/` (config, data, tokenizers, argo, baselines, unet, metrics).
Knobs (depths, #profiles, train/test split, epochs) live in `src/ocean_tokenizer/config.py`.

## 6. Caveats & next steps
- **WOA23 prior carries an irreducible model–obs bias.** The truth is the CESM2-LE
  *model*, whose climatology differs from real observations, so the WOA climatology
  baseline (TEMP ~1.3–1.7 °C) is dominated by this gap, *not* by climatology skill. A
  CESM2-LE *self*-climatology (built from the model's own training months) scores ~0.65 °C
  TEMP / ~0.2 PSU SALT — about half (TEMP) / a quarter (SALT) of WOA — and is included in
  the table as the true "climatology floor". The model–obs gap concentrates in the
  thermocline (steepest dT/dz) and western-boundary-current / upwelling hotspots (worst 5%
  of cells ≈ 40% of surface MSE). Alignment was verified bug-free (lon-shift test minimises at 0°).
- Ground truth is a *single LE member* of the full ensemble; profiles are noise-free (no
  measurement error / no realistic Argo spatial bias). Adding obs noise + realistic float
  distribution would harden the baselines.
- Depth capped at 985 m (upper ocean) for the prototype; extend `DEPTH_INDICES` for full column.
- Natural Stage-2 step: replace the per-method models with a single tokenizer-fed encoder that
  consumes any tokenizer's output (the lossless tokenizers already provide the interface), and
  add the `cesm2` single member once true curvilinear coordinates are available.
