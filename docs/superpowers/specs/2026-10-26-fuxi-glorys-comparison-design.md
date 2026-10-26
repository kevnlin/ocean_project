# KA-Ocean against a FuXi-Ocean-style model, on CESM2 and in transfer to GLORYS12

Date: 2026-10-26. Branch `fuxi-transfer` (from `size-study` ded2b90). Approved by Kevin. Written before any
run and before any new read of the GLORYS 2019-2020 cells.

## Question

The size study's chosen KA-Ocean (`y100_x1375`: method A, 17.7M parameters, 100 CESM2 years, 200,000 steps,
leads 0, 1, 3 and 6 months in one model) is compared twice:

1. **Against FuXi-Ocean's architecture**, trained on the same data, inputs, budget and objective. Only the
   model differs: does KA's U-Net state with a kriging read-out forecast better than FuXi-Ocean's encoder and
   attention core?
2. **In transfer to GLORYS12**: both models, trained on simulation only, forecast the GLORYS12 reanalysis
   ocean of 2019-2020, beside KA-Ocean trained on GLORYS itself (`g_ka`).

## Why a re-implementation

FuXi-Ocean (Huang et al., arXiv:2506.03210v2) cannot be run. Its repository (`huangqiusheng/FuXi-Ocean`,
checked 2026-10-26, last push 2026-06-27) holds an ONNX-runtime inference script and an RMSE evaluator: no
model definition, no training code, no weights ("ONNX weight files are not included yet"), no licence. The
paper trains on HYCOM 2006-2014 at 1/12 degree, 6-hourly, to 10 days, and gives RMSE only in figures. The
arm is therefore the paper's architecture (its section 3.2) re-implemented and trained under this study's
protocol, named **FuXi-Ocean-style**, as the ARMOR3D-style and Aardvark-style arms are named.

## Arms

| arm | model | trained on | runs |
|---|---|---|---|
| `y100_x1375` KA-Ocean | `101` method A (`--kriging ka_flow`, joint background), x1.375 widths | `cesm2_vlong_uniform`, 1904-2003, 200,000 steps | the size study's three seeds, reused; no new training |
| `y100_fuxi` FuXi-Ocean-style | `101 --backbone fuxi --kriging none` | the same | new: seeds 1234, 1235, 1236 |
| `g_ka` (reference) | `101` method A, x1 widths (9.4M parameters), 100,000 steps | `glorys12_uniform`, GLORYS 1997-2016 | the GLORYS OSSE's; read from its sealed results, never re-scored |

`g_ka` is the in-domain reference (what training on GLORYS buys), not a competitor: it is smaller and
trained for half the steps.

## What is equal between FuXi-Ocean-style and KA-Ocean

Everything `169`'s job sets except the model: the cohort and window (`cohort_options("cesm2_vlong_uniform",
100)`), 200,000 steps, 1,024 queries a step, 6,080 input profiles a month, leads 0/1/3/6 in one model with
`101`'s training draws, the objective (`101`'s masked loss: MSE plus 0.02 Gaussian NLL), the optimiser
(AdamW, lr 3e-4, weight decay 0.01, a 300-step warm-up then cosine to 0.1 lr, gradient clip 1), bf16
(`--amp`), validation every 2,000 steps on the fixed 8,000-cell draw of 2004, the checkpoint at the lowest
2004 score, 2005 held out and scored once at the end on every cell, the seeds.

The read-out after the state is the same code for both: the state read bilinearly at the query, the query's
58 features, the level and lead embeddings, the MLP (hidden 352) and the mean and spread heads. FuXi-style
has no kriging (`--kriging none`): the kriging read-out is KA's own contribution. Its total parameter count
is within 2 % of the KA runs' 17,730,974. Of those, 482,700 are the shared read-out and the decay
constants, which leaves the FuXi-style backbone about 17.25M (KA's U-Nets and kriging hold the same share).

## The FuXi-Ocean-style backbone (paper section 3.2)

Input: the tensor KA's encoder reads, on the 180 x 360 grid: 81 profile channels (T and S z-anomalies and
their seen flags at 20 levels, the profile count) and 12 context channels (SST, SSS, steric height and their
masks; sin and cos of latitude and longitude; the season). The lead `l` in months.

* **Context network M** (Eq. 1): `F_S = E + MLP(tau)`. `E` is a learnable embedding per token position
  (45 x 90 x C'), "learnable embeddings for spatial coordinates"; `tau` is the sinusoidal encoding of the input
  month's calendar month and of the lead. The paper's S also holds the bathymetry; it is not an input here,
  as KA has none (the masks carry the coast).
* **Encoder E** (Eq. 2): two convolutions with kernel = stride = 2 (180 x 360 -> 90 x 180 -> 45 x 90), each
  followed by LayerNorm. The second's output is scaled by `(1 + F_S)` at every token. That is Eq. 2's
  `W ⊙ F_S` acting on the output channels: scaling a linear map's output at a token equals scaling its
  weights' output rows there. The last layer of `MLP(tau)` starts at zero.
* **Fusion**: with one input month the concatenation of section 3.2.1 has one member; its 1 x 1 convolution
  and normalisation are kept.
* **Prediction module**: D Swin-v2 blocks (scaled cosine attention, log-spaced continuous position bias,
  residual post-norm, MLP ratio 4) on the 45 x 90 tokens, windows 9 x 9, shifted by 4 in every second block.
  Windows wrap across longitude (the sphere is periodic) and are masked across the poles. Both norms of
  every block are AdaLN: LayerNorm whose scale and shift are linear in `F_S` at that token, starting at zero
  (plain LayerNorm). The encoder's and decoder's norms are plain LayerNorm.
* **Decoder D**: transposed convolutions mirroring the encoder (45 x 90 -> 90 x 180 -> 180 x 360), each
  followed by normalisation, with skips from the encoder's 90 x 180 features and from the input at
  180 x 360 (concatenated, mixed by a 1 x 1 convolution). The output is the 176-channel state the shared
  read-out reads.
* **Leads**: direct. One pass per input month and lead; the lead enters through `tau` (the encoder's
  modulation and every AdaLN), as KA's lead evolution and the Poseidon arm take theirs.
* **Sizes**: C', D and the heads are fixed in the plan by counting parameters (target: the KA runs'
  count, within 2 %). Starting point: C' = 384 (192 at 90 x 180), D = 6, 12 heads, which counts to about
  17.4M with the read-out. FuXi-Ocean's own size is unpublished: the paper, its README and its code give no
  parameter count, width, depth, heads or patch size, and the weights are unreleased. The nearest published
  configuration, the FuXi weather model's (Chen et al. 2023, arXiv:2306.12873: width 1536, 48 Swin-v2
  blocks, 4 x 4 patches; about 1.8B with this AdaLN), would be FuXi's family scale, not FuXi-Ocean's. Kevin
  chose the matched size (2026-10-26).

Left out of the paper's recipe, each for the same reason (equal footing):

* **Mixture-of-Time** (section 3.2.3) chooses among predictions from several input windows; with KA's
  inputs (one month) there is one window.
* **Autoregression**: KA forecasts every lead directly; a rollout would differ in the lead method and in
  the compute per step.
* **The latitude-weighted Charbonnier loss and the two training stages**: both arms train on one objective;
  the queries are drawn in proportion to area, so the loss is already area-weighted.
* **The paper's optimiser** (AdamW with beta2 0.95, cosine 2.5e-4 to 1e-8, 60,000 iterations): both arms use
  `101`'s. If the smoke diverges at lr 3e-4, the FuXi-style runs use the paper's 2.5e-4 and the report
  says so.
* **Output in physical channels**: the decoder ends in the state the shared read-out reads.

Known asymmetry: KA's x1.375 is the best of three sizes on 2004 (0.7210, against 0.7225 for x1 and 0.7380
for x1.75); FuXi-style has one size, matched to it. The selection gain is 0.0015 on 2004 (0.2 %). The report
states it and does not correct for it.

## Test sets

### CESM2 2005 (in-domain, held out)

`101`'s final scoring: every cell of 2005, leads 0/1/3/6, the chosen checkpoint, with persisted OI and the
climatology on the same cells. KA's numbers are in the size study's summaries
(`forecast.best.by_lead[lead].by_year["2005"]`), FuXi-style's in its own runs' summaries.

### GLORYS12 2019-2020 (transfer; sealed cells, a declared extra pass)

The cohort `glorys12_test`, month index 72-95 (cohort calendar 2006-2007), every held-out cell (729,600 a
lead), leads 0/1/3/6: the cells and baselines `159` scored on 2026-10-22. No model of this study is trained
or selected on any GLORYS month; `g_ka`'s entries are read from `159`'s `results.json`.

* **Weights**: each CESM2 run's chosen checkpoint, unchanged; no GLORYS fine-tuning.
* **Statistics are GLORYS's own**, as they were for `g_ka`: the cohort's 1997-2016 climatology (the
  anomalies), the per-depth standardisation of the anomalies fitted on its training rows, `156`'s surface
  z-scores, and the OI file of `glorys12_test`. The OI enters only the persisted-OI row: the networks'
  background is the zero standardised anomaly (`synthetic_latent_experiment`'s first guess). The
  checkpoint's recorded CESM2 normalisation is written into the results beside the one used.
* **Same scenes as `159`**: built as `159` built them for `g_ka` (`train_years` 20), so the persisted OI and
  the climatology of every entry must equal `159`'s main pass to a relative 1e-9, or nothing is written
  (`168`'s extra-pass identity).
* **Same grid and features**: the checkpoint's grid must equal the cohort's surface grids, and the query
  feature count must be 58; otherwise `173` refuses. Checked by hand on 2026-10-26: `glorys12_test`'s
  surface grids are 180 x 360 at latitudes -89.5 to 89.5 and longitudes 0.5 to 359.5, the KA checkpoints'
  grid, and both cohorts' satellite caches have 56 feature columns with the surface at columns 44, 46, 48.

`173 --check PATH` runs first, on one seed of each arm: it scores the checkpoint on its own CESM2 2004-2005
(the caches its run recorded) and requires its run's recorded scores back (a relative 1e-3, `168`'s
check). It shows that the FuXi code left KA's model and the scoring path unchanged. `--register` records
every checkpoint and data file by SHA-256; `--score` runs once, when Kevin asks, after the FuXi-style runs
finish, and refuses a second time.

## Decision

Score: the mean over leads 0, 1, 3, 6 of the per-variable standardised RMSE (`macro_z`: the mean over T and
S of the RMSE in per-depth z units of the scored cohort), mean and standard deviation over the three seeds.
It is computed on CESM2 2005 and, separately, on GLORYS 2019-2020.

* "KA better" if KA's score is lower than FuXi-style's by more than the larger of the two arms' seed
  standard deviations; "FuXi-style better" the other way round; otherwise "no difference beyond the seed
  spread". Each test set gets its own verdict; the report states both.
* Tables by lead in physical units (T in degC, S in PSU), mean ± sd over seeds, beside persisted OI and the
  climatology, and on GLORYS beside `g_ka`.
* Transfer cost: each CESM2-trained arm's GLORYS RMSE as a ratio to `g_ka`'s, by lead and variable.
  `g_ka`'s sealed results hold physical RMSEs only, so it enters the tables and this ratio, not the score.

## Code

1. `src/ocean_tokenizer/fuxi_backbone.py`: the backbone above, `forward(x, lead, month)`.
   `grid_kriging_ocean.py`: backbone `"fuxi"` (its sizes as config fields with defaults, so earlier
   checkpoints load unchanged; it needs evolution off and the joint background, as Poseidon does; the
   lead-conditioned encode branch serves both). `101`: `--backbone fuxi`, and `fuxi_backbone.py` in its
   source hashes.
2. `172_fuxi_campaign.py`: `169`'s job with `--backbone fuxi --kriging none --state-channels 176 --hidden
   352` in place of the size options and `ka_flow`; no kept snapshots. A 600-step smoke
   first measures peak memory and seconds a step, which set the GPU slots, and shows whether lr 3e-4 is
   stable. Runs under `outputs/fuxi_20261026/` (ignored by git).
3. `159_sealed_glorys_test.py`: committed from the `ka-dev6` worktree unchanged. It produced `g_ka`'s sealed
   results, and `173` reads its data paths.
4. `173_glorys_transfer.py`: `--check`, `--register`, `--score` as above; writes
   `outputs/transfer_20261026/registration.json` and `results.json`. The main pass's files are only read.
5. `174_fuxi_transfer_report.py`: `reports/synthetic/fuxi_transfer_20261026.md` and `.json`: both tables,
   both verdicts computed (not hand-written), the transfer cost, parameters and hours per arm, the
   asymmetries above.
6. Tests: the backbone (output shape at leads 0 and 6, the lead changing the output once `MLP(tau)` is
   non-zero, the window mask wrapping in longitude and not in latitude, the parameter count within 2 %),
   the config's refusals, `172`'s job list (names, options, seed order), `173`'s refusals (a second
   registration or scoring, a grid mismatch, a baseline differing from the main pass), the verdict
   function.

## Runs

* The queue runs from a detached worktree at the commit that adds the backbone and `172`, as the size study
  ran from `/DATA2/linke/ocean_project_size`, so later commits on the branch do not change the code a
  running job reads. The CESM2 caches (`outputs/cohort_cesm2_vlong_uniform/` and its OI folder) are
  hard-linked into it from `ka-dev6` (`cp -al`, the size study's step; the same files the size study used).
  `outputs/fuxi_20261026` in this checkout links to its outputs.
* `173` runs from this checkout. The GLORYS caches, the `glorys12_test` OI file and `159`'s sealed folder
  are hard-linked into it from `ka-dev6` (no extra disk).
* GPUs: the queue's memory gate. All eight GPUs were busy on 2026-10-26 (95-99 % utilisation), so launches
  wait for free memory. Hours a run are set by the smoke.
* Disk: `/DATA2` had 544 GB free on 2026-10-26; three runs without snapshots need a few GB.

## Not in this study

FuXi-style at other sizes; FuXi-style trained on GLORYS; GLORYS fine-tuning of either model; CRPS and
coverage (KA's sealed CRPS used `116`'s calibration, which FuXi-style would need too); FuXi-Ocean's own
weights (unreleased); Mixture-of-Time and autoregression (above).

## Known consequences

* `101`'s source hash changes (a new file in its list), so no earlier checkpoint can `--resume-training`;
  finished runs still evaluate. The size study's runs are finished.
* The sealed GLORYS cells are read again, as a declared extra pass. Nothing is selected on them, and the
  main pass's files are not changed.
* `159` enters git as the file that produced the 2026-10-22 sealed results.
