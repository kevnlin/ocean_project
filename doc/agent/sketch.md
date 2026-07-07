# Experiment Sketch — Unified Ocean Tokenizer Prototype

## Current State
- Phase: baseline sweep running (experiments/03_baselines.py, bg PID in outputs/baseline.pid)
- Tokenizers + round-trip: DONE (all 4 exact, max_abs=0)
- Data cards + synthetic Argo: DONE
- Next: 04_report.py once sweep finishes -> reports/baseline_table.md, then final_report.md

## Goal
Clean tokenizer/data pipeline + small baseline table. NOT a foundation model.
Reconstruct full 3D ocean state (TEMP/SALT over depth) from sparse synthetic Argo
profiles + WOA23 climatology prior (+ SST/SSS). GT = CESM2-LE full simulation.

## Datasets / roles
- cesm2_le_full : full LE simulation, regular 1deg, 1850-2100 -> GROUND TRUTH ("full ocean state")
- woa23         : observational climatology -> PRIOR / baseline
- cesm2         : single member (curvilinear placeholder grid) -> extra sample, not in sweep

## Common grid
- lat 180 (-89.5..89.5), lon 360 (0-360; WOA rolled onto it), 20 target depths 5-985 m
- ocean mask MASK==1 (65%); physical-range clip drops fill/brine outliers

## Design decisions
- Tokenizers are lossless rearrangements -> exact round-trip by construction.
- GT depths are a subset of native LE levels (no GT vertical interpolation).
- Methods: climatology, nearest (KDTree on sphere), pointwise MLP, depthwise 2D U-Net.
- Configs: profiles_only / woa_only / profiles_woa / profiles_woa_surf.
- z-score normalisation per (var, depth) from training GT.

## Next Steps
1. [ ] Finish sweep -> baseline_results.json + depth tables
2. [ ] 04_report.py -> baseline_table.md
3. [ ] final_report.md tying everything together
