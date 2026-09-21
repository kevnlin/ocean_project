# reports/

Most files here are **generated** — edit the script, not the report, and re-run it. Each folder
mirrors `experiments/<folder>/`, and each generator writes into its own folder.

## Start here

| report | what it is | regenerate with |
|---|---|---|
| [`real_data/main_tables.md`](real_data/main_tables.md) | Tables 1–4: DFS / Uniform / Count by lead, by depth band, against baselines, against EN4/ECCO | `experiments/real_data/57_main_tables.py` |
| [`real_data/pipeline_audit.md`](real_data/pipeline_audit.md) | **Read with the tables.** What the data path does to the numbers, and how predictable the anomaly is at a held-out float | `experiments/real_data/61_pipeline_audit.py` |
| [`crosscheck/final_crosscheck_report.md`](crosscheck/final_crosscheck_report.md) | Track A vs Track B across every cross-check package | `experiments/real_data/47_argo_reports.py` |
| [`real_data/tau_spread.md`](real_data/tau_spread.md) | How much the DFS evidence estimate actually varies on real input | `experiments/real_data/59_tau_spread.py` |
| [`real_data/input_density.md`](real_data/input_density.md) | What the `--n-profiles` cap really delivers per month | `experiments/real_data/60_input_density.py` |
| [`synthetic/protocol_v1.md`](synthetic/protocol_v1.md) | The frozen synthetic-era protocol | hand-written |

## `real_data/` — 17 files

Current real-observation results.

- `main_tables.md` — `57_main_tables.py`
- `tau_spread.md`, `input_density.md` — `59`, `60`
- `fig_argo_global_*`, `fig_en4_smoke*`, `fig_diag_*` — global 2-D reconstruction maps, `53_argo_global_recon.py` (`_en4` = scored against EN4, `_demean` = annual mean removed)
- `godas_rows.md` — first real GODAS run, `14_godas_dfs_d4rt.py`
- `main_equatorial_reconstruction.md` — hand-written note on the equatorial Pacific
- `pipeline_audit.md` — the 2026-09-17 audit of data prep / normalisation / tokenisation, with the predictability floor, `61_pipeline_audit.py` + `63_audit_report.py`
- `overfit_sanity.md` — can the model fit data it is allowed to memorise, `62_sanity_train.py --mode memorise|copy|small`
- `ablation_ladder.md` — one switch at a time on a fixed held-out set, `62_sanity_train.py --ablation ...`
- `fig_pair_correlation.png`, `fig_overfit.png`, `fig_loss_curves.png`, `fig_ablation.png`, `fig_architecture.{png,svg}` — the audit's figures, `63`/`64`

## `crosscheck/` — 59 files

Everything `47_argo_reports.py` writes, from the signed artifacts in `outputs/`. Names follow
`<track>_<package>_<region>.md`:

| prefix | meaning |
|---|---|
| `main_` | Track A — the main implementation |
| `intern_` | Track B — the independent re-implementation |
| `crosscheck_` | the two compared, at the level of residuals |

Packages: `control_ladder`, `layout_generalization`, `prospective_test`, `real_data_baseline`, `redundancy`, `sparsity`. Regions: `gulfstream`, `npac_gyre`, `eq_pacific`.
Also: `*_external_baselines.md`, `*_uncertainty.md`, `final_crosscheck_report.md`, and the hand-written `parallel_crosscheck_status.md`.

## `synthetic/` — 62 files

The CESM2-LE era. Generators write here through `config.REPORTS_SYNTHETIC`.

| theme | reports | script |
|---|---|---|
| data & tokenizer | `data_card_*`, `common_grid`, `tokenizer_roundtrip`, `synthetic_argo` | 00, 01, 02 |
| baselines | `baseline_table`, `baseline_comparison.*`, `week1_audit` | 03–06, `src/baselines/` |
| density | `week2_density_ablation`, `density_4000_6000`, `fig_week2_*`, `fig_density_powerlaw` | 08, 09, 31 |
| depth | `layered_depth_eval`, `depth_band_eval`, `joint_unet_audit`, `fig_layered_*`, `fig_audit_*` | 10–14 |
| full training | `full_training_*`, `fig_full_*`, `invariance_test_summary` | 17–20 |
| DFS evidence | `dfs_evidence_probes`, `dfs_success_criterion`, `phase0_operating_point` | 23, 25, 29 |
| OI & SSH | `oi_*`, `fig_oi_*`, `ssh_ablation`, `rmse_target_variance` | 26–30, 33 |
| D4RT | `d4rt_lead_probe`, `d4rt_recon_heatmap*`, `fig_d4rt_recon_*`, `fig_reconstruction_na_*` | 32, 34–36 |
| stress & redundancy | `obs_stress_*`, `profile_reduction`, `fig_redundancy_*`, `phase1_phase2_redundancy` | 37–40 |
| history | `final_report` (Stage 1), `intern_week1`, `protocol_v1` | hand-written |

## `notes/` — 7 files

Hand-written, not generated: `monday_2026-08-11_slides.md`, `prior_art_overlap.md`,
`reading_cbottle.md`, and, from the 2026-09-17 meeting:
`gap_novelty_venues.md` (performance gap, novelty risk, venue recommendations),
`backbone_options.md` (replacing the Perceiver-IO trunk, and the first result),
`update_2026-09-18.md` (what to show and what to cut in the next update),
`architecture_diagram_prompt.md`.
