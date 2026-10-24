# KA-Ocean model size on 100 CESM2 years

Date: 2026-10-23. Branch `size-study` (from `main` c7429ca). Approved by Kevin. Written before any run.

## Question

Does a larger KA-Ocean (Zihao's gridded model, method A: U-Net state, lead evolution, kriging
read-out) forecast better when it trains on 100 simulated years? If it does, the best size's forecast
at leads 0, 1, 3 and 6 months is the result to report.

## Runs

* Three sizes, one width factor on every channel count (`--state-channels`, `--encoder-channels`,
  `--evolution-channels`, `--hidden`; nothing else changes):

  | arm | factor | state | encoder | evolution | hidden | parameters |
  |---|---|---|---|---|---|---|
  | `y100_x1` | 1 | 128 | 64 128 192 256 | 64 128 192 | 256 | 9.4M |
  | `y100_x1375` | 1.375 | 176 | 88 176 264 352 | 88 176 264 | 352 | 17.7M |
  | `y100_x175` | 1.75 | 224 | 112 224 336 448 | 112 224 336 | 448 | 28.6M |

* Seeds 1234, 1235, 1236; seeds outermost, so every size has seed 1234 before any second seed runs.
  Nine runs.
* Every run: runner `101_grid_ka_forecast.py` with A's settings (`--kriging ka_flow`, joint background,
  `--surface`, `--amp`), cohort `cesm2_vlong_uniform` with `--train-years 100` (1904-2003), 200,000 steps,
  1,024 queries a step, lr 3e-4, `--context-profiles 6080`, leads 0 1 3 6 in one model,
  `--val-every 2000`, `--checkpoint-every 500`.
* The cohort's normalisation, climatology and OI fits are used exactly as `c_y154` used them; only the
  training window (100 instead of 154 years), the steps and the size differ from `c_y154`.
* A 500-step smoke of each size before the queue: peak memory and seconds a step, which set how many
  runs share a GPU. Expected about 6 / 10 / 15 hours a run, about 95 GPU-hours, on the GPUs free at
  launch (0, 2, 3, 4, 6 on 2026-10-23; other users hold 1, 5 and 7).

## Selection and held-out year

Each run's checkpoint is chosen on target year 2004 (`--val-years 2004`: mean over leads of the
per-variable standardised RMSE on the fixed 8,000-cell draw). Target year 2005 (`--test-years 2005`)
is held out of that choice and scored once at the end, for the chosen checkpoint and for the model of
the last step. KA's sealed tests (CESM2 2006-2007, 2008-2010) are not read.

## Decision

* "Bigger helps" if `y100_x175`'s 2005 score (mean standardised RMSE over leads, mean of three seeds)
  is lower than `y100_x1`'s by more than the larger of the two sizes' seed standard deviations.
* The best size is the lowest **2004** score; its 2005 table by lead (temperature and salinity RMSE in
  physical units, mean ± sd over seeds, beside persisted OI and climatology) is the forecast result.
  Choosing on 2005 would end its being held out.

## Convergence

The curves are the convergence check, drawn as a figure. Every run's `train.log` has a record every
2,000 steps: the training MSE averaged over the steps since the previous record, and the 2004
validation score (mean standardised RMSE over leads). `171_size_curves.py` reads the logs and draws
`reports/synthetic/size_20261023_curves.png` (matplotlib, light surface, 150 dpi):

```
              x1 · 9.4M            x1.375 · 17.7M        x1.75 · 28.6M
training   ╭───────────────╮   ╭───────────────╮   ╭───────────────╮
MSE (log)  │╲__             │   │╲__             │   │╲__             │   one line per seed
           │    ‾‾──────┊───│   │    ‾‾──────┊───│   │    ‾‾──────┊───│   ┊ = 180k (last tenth)
           ╰───────────────╯   ╰───────────────╯   ╰───────────────╯
2004 val.  ╭───────────────╮   ╭───────────────╮   ╭───────────────╮
score      │╲               │   │╲               │   │╲               │   ● = each seed's best step
           │ ‾──────────●┊─ │   │ ‾─────────●─┊─ │   │ ‾───────────┊●│
           ╰───────────────╯   ╰───────────────╯   ╰───────────────╯
                 steps (thousands, 0-200)
```

* Two rows, never a second y-axis: training MSE on a log scale (top), the 2004 validation score
  (bottom). Rows share their y-axis across sizes, so the panels compare directly.
* One column per size, its title the factor and parameter count; within a panel one 2 px line per
  seed, all in the size's colour.
* Colour encodes the size as an ordered tier: one blue ramp, light to dark, `#6da7ec` (x1),
  `#2a78d6` (x1.375), `#184f95` (x1.75); `dataviz/scripts/validate_palette.js --ordinal` passes on
  the light surface `#fcfcfb`. Text, axes and the recessive grid stay in neutral inks.
* Each seed's best validation step is a dot with a surface-coloured ring; a dotted vertical line marks
  180k, the start of the last tenth. A legend names the three sizes, the dot and the line.

Below the figure, a table per run: the change of the training MSE over the last tenth (180-200k)
against the tenth before (`97_forecast_curves.tail_change`), the best validation step and score. A
best step inside the last tenth means the run was still improving at 200k, and the table says so
rather than calling it converged. The script reads only logs, so it runs at any time, also while
the queue is going, to watch the curves form; unfinished runs are drawn as far as they have got.

## Code

1. `101_grid_ka_forecast.py` gains `--test-years`, as `94_forecast_reconstruction.py` did: the
   checkpoint is chosen on `--val-years` only; the final scoring covers `--val-years` and
   `--test-years`; a year in both is refused; `test_years` is recorded in the summary. Default empty, so
   every existing campaign's command means what it meant.
2. `169_size_campaign.py`: the queue, from `150_wang_campaign.py`'s machinery (lock, attempts, GPU
   slots, status file); runs under `outputs/size_20261023/`, ignored by git.
3. The cohort's caches (`outputs/cohort_cesm2_vlong_uniform/` and
   `outputs/audit/synthetic/fixed_baselines_cesm2_vlong_uniform/`) are copied from the `ka-dev6`
   worktree into this checkout, so the runs do not depend on that worktree.
4. `170_size_report.py`: by size and lead, 2005 and 2004, RMSE mean ± sd over seeds, persisted OI and
   climatology, parameters, hours, best step; the decision above computed, not hand-written; it links
   the convergence figure.
5. `171_size_curves.py`: the convergence figure and table above, reading the logs with
   `97_forecast_curves.parse` (101 writes the same records as 94).
6. Tests: `--test-years` (choice on validation years only, overlap refused) and the campaign's job list
   (names, options, seed order).

## Not in this study

The OI-anchored transformer, other learning rates per size, sizes above 30M parameters, depth changes,
and the sealed tests.

## Known consequences

Runner 101's source hash changes, so a checkpoint from before this change cannot be resumed with
`--resume-training`; it still evaluates. Larger models share lr 3e-4: if `y100_x175` underperforms,
an untuned learning rate is a possible cause the study cannot rule out.
