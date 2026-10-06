# Ocean Tokenizer — Sparse-Profile Subsurface Ocean Reconstruction

Reconstruct the full 3-D subsurface **temperature & salinity** field from sparse,
Argo-like profiles. The problem is framed as an **observing-system simulation
experiment (OSSE)** on CESM2-LE, with WOA23 as an observational prior: the model
climate is the (fully known) ground truth, so reconstructions can be scored exactly.

## Quick start on a new machine (current pipeline)

The current work is the **global real-Argo reconstruction** and its pipeline
audit, plus a **synthetic audit** on CESM2 where the truth is known. The
sections further down describe the earlier CESM2 OSSE line and are kept for
reference.

**1. Environment.** Python 3.12, one CUDA GPU (CPU works for smoke runs).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # loose pins
# or the exact environment the results were produced with:
pip install -r requirements-lock.txt
.venv/bin/python -m pytest -q             # 498 tests, ~30 s
```

**2. Data.** Everything the scripts read (processed Argo cohorts, WOA23,
CESM2, satellite fields, synthetic cohort, run outputs, checkpoints) is on
Hugging Face, laid out where the code expects it; see
[`docs/DATA.md`](docs/DATA.md) for what each folder is, the splits, the target
and the normalization.

```bash
hf download klin2323/ocean_project-data --repo-type dataset --local-dir .
```

**3. Preprocessing** (only to rebuild from raw; the processed files are in the download).

```bash
.venv/bin/python experiments/data/42_download_argo.py --regions global --start-year 2002
.venv/bin/python experiments/data/44_build_argo_cohort.py --regions global --grid global \
    --levels protocol --suffix _global                              # -> data/argo_cohort/global_global.nc
.venv/bin/python experiments/data/41_download_real_obs.py --start 2016 --end 2023
.venv/bin/python experiments/synthetic/41_synth_argo_cohort.py      # synthetic cohort
```

**4. Train and evaluate one model.** `62_sanity_train.py` trains, selects the
best validation checkpoint and scores it on the test years in one run; it
writes `outputs/audit/<region>/<tag>/summary_seed<seed>.json`.

```bash
# registered model, real data
.venv/bin/python experiments/real_data/62_sanity_train.py --region global --seed 1234 --tag baseline
# with the audit's fixes: target at the profile's position, local refiner
.venv/bin/python experiments/real_data/62_sanity_train.py --region global --seed 1234 \
    --tag real_exact_r500_g1 --ablation anomaly_exact --refiner-km 500 --refiner-dz 100 --refiner-gate 1.0
# overfit sanity check: one fixed month, 20k steps
.venv/bin/python experiments/real_data/62_sanity_train.py --region global --mode memorise --steps 20000 --tag real_mem20k
```

Every switch is listed by `--help`; `--ablation` takes a comma-separated list
(`anomaly_exact`, `qc`, `no_refiner`, `no_latent`, `mass_uniform`, ...).

**5. Reproduce the audits.** The queue runner spreads a job matrix over an
explicit GPU list and skips runs that already have a summary.

```bash
R=experiments/real_data/run_audit_queue.py
# real data: 2026-09-17 audit, 20k contribution study, pre-shutdown A/B/C
.venv/bin/python $R --queue ablation --gpus 0,1
.venv/bin/python $R --queue contrib --gpus 0,1
.venv/bin/python $R --queue real_final --gpus 0,1 --seeds 1234,1235,1236
.venv/bin/python $R --queue real_overfit --gpus 0,1 --seeds 1234
.venv/bin/python experiments/real_data/61_pipeline_audit.py    # data audit
.venv/bin/python experiments/real_data/63_audit_report.py      # -> reports/real_data/
# synthetic audit
.venv/bin/python experiments/synthetic/42_synth_argo_audit.py
.venv/bin/python $R --queue syn_overfit --gpus 0,1 --seeds 1234
.venv/bin/python $R --queue syn_refiner --gpus 0,1 --seeds 1234,1235,1236
.venv/bin/python $R --queue syn_final --gpus 0,1 --seeds 1234,1235,1236
.venv/bin/python $R --queue syn_refiner_k15 --gpus 0,1 --seeds 1234,1235,1236      # refiner sweep at 15 k steps
.venv/bin/python $R --queue syn_refiner_k15_l64 --gpus 0,1 --seeds 1234,1235,1236  # the same with 64 latent slots
.venv/bin/python $R --queue syn_mass_k15_l64 --gpus 0,1 --seeds 1234,1235,1236     # uniform / count mass at 64 slots, 15 k steps
.venv/bin/python experiments/synthetic/44_synth_argo_oi.py     # fixed baselines: climatology, nearest profile, OI
.venv/bin/python experiments/synthetic/43_synth_argo_report.py # -> reports/synthetic/synth_argo_audit.md
```

Results so far: [`reports/final_summary.md`](reports/final_summary.md).

## What's here

- **DFS-Attention** — the method: scale-aware *effective-evidence* fusion of
  heterogeneous observations. Each observation's degrees of freedom for signal are
  estimated by ridge leverage under a three-dimensional, stratification- and
  target-resolution-aware support kernel, transported through a fixed-budget
  resampler without loss, and fused against a climatological background
  (`ocean_tokenizer.dfs`, `fusion.DFSAttention`, [`docs/dfs_attention.md`](docs/dfs_attention.md)).
  MBCA — hand-designed physical weights + log-weighted attention — is retained as
  the baseline it is compared against.
- **Unified data pipeline** — CESM2-LE (ground truth) + WOA23 (prior) standardized to a
  common 1° / 20-level grid as Zarr (`experiments/data/standardize.py`, `ocean_tokenizer.data`).
- **Four lossless tokenizers** — grid-patch, volume-patch, vertical-profile, point-query
  (`decode(encode(field)) == field` exactly; `ocean_tokenizer.tokenizers`).
- **Synthetic Argo sampling + baseline sweep** — climatology, nearest-profile, pointwise
  MLP, depthwise 2-D U-Net across input configs (`ocean_tokenizer.baselines`).
- **Three re-implemented literature reference models** — NeSPReSO (PCA+MLP), OSnet
  (15× bootstrap-ensemble MLP), Buongiorno-Nardelli stacked-LSTM (`src/baselines/`).
- **Depth-banded RMSE comparison** of every method on identical held-out data.

## Current headline (protocol_v1: anomaly target, unobserved-only RMSE)

Under the corrected evaluation (train-only monthly CESM2 anomaly target,
observed profile columns excluded from scoring), the strongest baseline is the
multi-modal **depthwise 2-D U-Net (profiles + WOA + SST/SSS)** at
**0.15 °C / 0.03 PSU** full-column vs a 0.54 °C / 0.12 PSU train-climatology
floor. The surface-focused reference models (OSnet et al.) lead in the upper
ocean; no single method is best everywhere. The re-implemented literature
models are **SSH-ablated adaptations** (no SSH/ADT input) — numbers here do not
support superiority claims over the published originals.
Protocol: [`reports/synthetic/protocol_v1.md`](reports/synthetic/protocol_v1.md) ·
Audit: [`reports/synthetic/week1_audit.md`](reports/synthetic/week1_audit.md) ·
Bands: [`reports/synthetic/depth_band_eval.md`](reports/synthetic/depth_band_eval.md).
Scope: contemporaneous reconstruction only — no forecasting, no
super-resolution claims.

## Layout

```
src/ocean_tokenizer/   core package: config, data, tokenizers, argo, unet, baselines, metrics
                       + token_api (token schema), dfs (evidence estimator),
                         fusion (perceiver / resampler / MBCA / DFS-Attention)
src/baselines/         reference models (nesperso_pcamlp / osnet_mlp / nardelli_lstm) + README
                       + build_comparison_table.py
experiments/           runnable scripts 00–05 + standardize.py
reports/               generated markdown / CSV reports and data cards
checkpoints/           trained model weights (small; tracked)
data/, processed/      raw NetCDF + standardized Zarr  (NOT tracked — see "Data" below)
```

## Setup

Python 3.12; a single GPU is used if available (CPU works for smoke runs).

## Reproduce (earlier CESM2 OSSE line)

```bash
# DFS-Attention
python experiments/synthetic/23_dfs_evidence_probes.py   # -> reports/synthetic/dfs_evidence_probes.md
python experiments/synthetic/18_full_train.py --variant dfs --seed 1234 --tag fullA_dfs_s1234
python experiments/synthetic/19_full_eval.py  --tag fullA_dfs_s1234
python experiments/synthetic/25_dfs_report.py            # -> reports/synthetic/dfs_success_criterion.md

# core pipeline
python experiments/synthetic/00_data_cards.py            # data cards + common grid
python experiments/synthetic/01_tokenizer_roundtrip.py   # lossless round-trip check
python experiments/synthetic/02_synth_argo.py            # synthetic Argo example
python experiments/synthetic/03_baselines.py             # baseline sweep (--smoke for a fast check)
python experiments/synthetic/05_band_table.py            # -> reports/synthetic/baseline_table.md

# reference baselines (each --smoke for a quick check)
python src/baselines/nesperso_pcamlp.py
python src/baselines/osnet_mlp.py
python src/baselines/nardelli_lstm.py
python src/baselines/build_comparison_table.py # -> reports/synthetic/baseline_comparison.{md,csv}
```

See [`src/baselines/README.md`](src/baselines/README.md) for the reference-model details
and their documented departures from the original papers (chiefly: no SSH/ADT input).

## Experimental setup

| | |
|---|---|
| Ground truth | CESM2-LE full simulation (regular 1°, 180×360, 20 levels 5–985 m) |
| Prior | WOA23 monthly climatology |
| Synthetic Argo | 1500 random ocean columns / month (~3.5% coverage) |
| Split | 48 train months (1985–2010) / 12 test months (2011–2014), fixed seed 1234 |
| Metric | depth-banded RMSE — valid-cell-weighted, NaN-aware, ocean only |

## Data

The raw NetCDF (`data/`, ~4.7 GB) and standardized Zarr stores (`processed/`, ~42 GB) are
**not tracked in git**. Place the source CESM2-LE / WOA23 files under `data/` and run
`python experiments/data/standardize.py` to regenerate `processed/`. The large per-cell
prediction arrays (`predictions/*.npz`) are also untracked (they exceed GitHub's file-size
limit); the RMSE tables/CSVs and the trained checkpoints that produce them are included.

> **Note on numbers:** an earlier task brief cited 500 floats / 96–24 months / 30 levels;
> the live pipeline uses **1500 profiles/month, 48/12 months, 20 levels** — all reports
> reflect the live configuration.
