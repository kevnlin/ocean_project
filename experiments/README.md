# experiments/

Every script is run **from the repository root**, e.g.
`python experiments/real_data/57_main_tables.py`. Numbers record the order the work was done in,
not a pipeline order, and were kept unchanged when the folder was split so old notes and commits
still identify each script. Generated reports land in the matching `reports/<folder>/`.

## `data/` — Data ingest and standardization

Downloads, QC and regridding. Run these first; everything else reads what they write under `data/`.

| script | what it does |
|---|---|
| `13_download_godas.py` | GODAS regional subset download (mentor doc §4). |
| `41_download_real_obs.py` | Download real satellite/in-situ observations and regrid to the 1-deg grid. |
| `42_download_argo.py` | Real Argo profile download — the float cohort the cross-check plan assumes. |
| `43_download_reference_products.py` | EN4 and ECCO — the external gridded reference products for plan P0/P4. |
| `44_build_argo_cohort.py` | Turn raw Argo float files into the analysis cohort the plan's protocol needs. |
| `standardize.py` | Standardize raw ocean datasets -> common zarr format (matches processed/). |

## `real_data/` — Real observations — GODAS and Argo

The current work: registered rows on real Argo, held-out-float evaluation, the cross-check packages (P0–P7), the main tables, and the global reconstruction maps.

| script | what it does |
|---|---|
| `14_godas_dfs_d4rt.py` | GODAS registered-row driver (mentor doc §6). |
| `45_argo_real_data.py` | Track A driver — the registered rows on REAL Argo, scored on held-out floats. |
| `46_argo_redundancy.py` | P2 — redundancy stress on REAL Argo, with the positive control. |
| `47_argo_reports.py` | Render the plan's deliverables from the signed artifacts. |
| `48_senseiver.py` | P4 — the Senseiver, run as the Senseiver, then adapted to our evaluation. |
| `50_prospective_test.py` | P7 — the prospective real-observation test, frozen before it is opened. |
| `51_track_b.py` | Track B — an independent re-implementation of the evaluation path. |
| `52_argo_recon_map.py` | 2-D reconstruction error maps on REAL held-out Argo floats. |
| `53_argo_global_recon.py` | Global 2-D T/S reconstruction on REAL data — DFS-Attention + Perceiver-IO |
| `54_argo_uncertainty.py` | P6 — post-hoc predictive uncertainty on frozen real-Argo checkpoints. |
| `56_senseiver_score.py` | Score the adapted Senseiver at held-out Argo float positions (plan P4). |
| `57_main_tables.py` | The four main tables, assembled from the signed artifacts. |
| `58_argo_fusion_tables.py` | Main tables on the Perceiver-IO family: DFS-Attention + D4RT query decoder. |
| `59_tau_spread.py` | How much does the DFS evidence estimate actually vary on real Argo input? |
| `60_input_density.py` | What the `--n-profiles` cap actually delivers, per month. |
| `61_pipeline_audit.py` | The 2026-09-17 audit: data prep, normalisation, tokenisation, and the predictability floor (anomaly correlation vs distance, nugget per latitude band). |
| `62_sanity_train.py` | Overfit sanity ladder, one-switch-at-a-time ablations and backbone trials, with W&B/JSONL training curves. |
| `63_audit_report.py` | Render `pipeline_audit.md`, `overfit_sanity.md`, `ablation_ladder.md` and their figures from what 61 and 62 wrote. |
| `64_architecture_figure.py` | The architecture diagram, annotated with the audit's numbers. |
| `run_audit_queue.py` | Run 62's job matrix across an explicit GPU list. |
| `run_argo_queue.sh` | Track A real-Argo run queue. |

## `synthetic/` — CESM2-LE synthetic era

The earlier benchmark: CESM2-LE as ground truth, synthetic Argo sampled from it, protocol_v1. Kept because it holds the controlled tests where ground truth is known — the only place the evidence mechanism can be checked against truth (e.g. 15, 23, 39, 40).

| script | what it does |
|---|---|
| `00_data_cards.py` | Write data cards for the three standardized datasets + the common grid card. |
| `01_tokenizer_roundtrip.py` | Round-trip test for all four tokenizers: field -> tokens -> field, error ~ 0. |
| `02_synth_argo.py` | Generate synthetic Argo-like profiles from CESM2-LE and summarise them. |
| `03_baselines.py` | Baseline sweep: methods x configs -> RMSE by variable and by depth. |
| `04_report.py` | Assemble the baseline table + depth-resolved tables into markdown reports. |
| `05_band_table.py` | Regenerate baseline_table.md with a depth-BANDED layout. |
| `06_week1_audit.py` | Week-1 audit: the corrected baseline matrix. |
| `07_error_map.py` | Spatial reconstruction-error heatmap for the best Week-1 model. |
| `08_density_ablation.py` | Week-2: profile-density ablation with multiple random seeds. |
| `09_week2_report.py` | Week-2 report: aggregate the multi-seed density ablation into tables + figures. |
| `10_layered_depth_eval.py` | Phase-1: layered depth evaluation on an extended (to ~1400 m) grid. |
| `11_layered_heatmap.py` | Phase-1: per-layer spatial reconstruction-error heatmaps (extended depth). |
| `12_layered_report.py` | Phase-1: aggregate the layered depth evaluation into a report + figure. |
| `13_joint_audit.py` | Task 1 — joint-depth U-Net convergence audit under protocol_v1. |
| `14_joint_audit_report.py` | Task 1 + Task 8 reporting: joint-depth U-Net audit closure. |
| `15_poc_toy.py` | Task 7 Stage A — synthetic toy: does MBCA fix tokenization-multiplicity bias? |
| `16_poc_ocean.py` | Task 7 Stage B — small ocean-subset proof of concept for the fusion variants. |
| `17_invariance_summary.py` | Task 6 deliverable — one summary table of all invariance test results. |
| `18_full_train.py` | Task 8 (Week 4) — first full-scale training run of a shared-latent fusion variant. |
| `19_full_eval.py` | Task 8b (Week 4) — post-training evaluation of one full-run checkpoint. |
| `20_full_report.py` | Task 8c (Week 4) — aggregate the full-scale runs into the Week-4 report. |
| `21_baselines_protocol_v1.py` | Task 9a — protocol_v1 non-neural baseline rows for the headline table. |
| `22_withheld_profile_eval.py` | Task 9b — withheld-profile evaluation (leave-profiles-out generalisation). |
| `23_dfs_evidence_probes.py` | Section-11 required experiments for DFS-Attention — the evidence probes. |
| `25_dfs_report.py` | DFS-Attention — the success-criterion report (plan Section 12). |
| `26_oi_tuning.py` | Phase-1 / Task 1.3 — tune the optimal-interpolation baseline (L, gamma, k). |
| `27_oi_vs_unet.py` | Phase-1 / Task 1.4 [Milestone M1] — OI vs the learned reconstructors. |
| `28_make_ssh.py` | Phase-4 / Task 4.1 — build the pseudo-SSH ("satellite altimeter") modality. |
| `29_dfs_operating_point.py` | Phase 0 gate — measure and calibrate the DFS operating point. |
| `29_ssh_ablation.py` | Phase-4 / Task 4.2 — does the pseudo-SSH channel help? |
| `30_oi_report.py` | Phase-1 / Task 1.5 — turn the OI caches into the two M1 reports + figures. |
| `31_density_powerlaw.py` | Phase-2 / Task 2.2 — power-law fit of the profile-density curve. |
| `32_reconstruction_figure.py` | Reconstruction visualisation for the Monday deck — CURRENT U-Net results. |
| `33_variance_table.py` | RMSE + target variance per level/band — the numbers an external group needs |
| `34_d4rt_lead_train.py` | D4RT causal space-time query decoder — training (spec §2.3/§2.4 on protocol_v1). |
| `35_d4rt_lead_eval.py` | D4RT lead evaluation — unobserved-only RMSE at leads 0..3 (spec §2.1, §6). |
| `36_d4rt_recon_heatmap.py` | 2-D reconstruction of T/S with the current architecture + spatial error heatmap. |
| `37_obs_stress.py` | Observation stress test on ONE frozen checkpoint (the DFS-Attention claim). |
| `38_profile_reduction.py` | Mentor stress test — eval-only Argo-profile reduction on FROZEN models. |
| `39_redundancy_regimes.py` | Phase 2 — put the pathology into the evaluation. |
| `40_redundancy_accuracy.py` | Phase 2 — does redundancy handling change ACCURACY when redundancy exists? |
| `merge_density_json.py` | Phase-2 / Task 2.1 — merge an extended density sweep into the week-2 curve. |
| `run_full_queue.py` | Queue runner for the full-scale runs (variants x seeds; 12 by default, |
| `run_oi_queue.sh` | Phase-1 (M1) queue: tune OI on validation months, then score the headline |
| `run_seeds_queue.sh` | Remaining seeds for the two GPU experiments, so every headline number reaches |

## Conventions

- Scripts locate `src/` and the project root relative to their own file, so they run from any working directory as long as they stay one folder below `experiments/`.
- The `run_*.sh` / `run_*_queue.py` launchers `cd` to the root themselves.
- `50_prospective_test.py` is not a test — it is the P7 frozen prospective evaluation. Its name matches pytest's `*_test.py` pattern, which is why the test suite is scoped to `tests/`.
