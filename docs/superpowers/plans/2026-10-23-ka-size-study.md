# KA-Ocean size study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train KA-Ocean at 9.4M / 17.7M / 28.6M parameters on 100 CESM2 years for 200k steps (3 seeds each), watch convergence in a curves figure, and report whether size helps and the best size's lead 0/1/3/6 forecast on held-out 2005.

**Architecture:** No model code changes. Runner `101` gains `--test-years`; a new queue `169` reuses `150`'s queue machinery with three width factors; `171` draws the convergence figure from the run logs with `97`'s log reader; `170` writes the report with `96`'s climatology. Spec: `docs/superpowers/specs/2026-10-23-ka-size-study-design.md`.

**Tech Stack:** Python 3.12 (`.venv`), PyTorch 2.13, NumPy, matplotlib (Agg), pytest; 8x RTX PRO 6000 (96 GB), shared with other users.

---

Facts the tasks rely on (checked 2026-10-23):

* Branch `size-study` from `main` c7429ca. Suite baseline: 1299 passed, 8 skipped, 4 failed — the 4 are `tests/test_poseidon_backbone.py`, which need the Poseidon runtime this checkout lacks; they are the only failures allowed throughout.
* Runner 101: `--val-years` default `[2004, 2005]`; the final scoring is `data.forecast_months(args.val_years, 0)` at two places (after training, and for `--evaluate-only`); `restore_training_inputs(args, training)` restores the years of a checkpoint; `--tag` is required; lr 3e-4; its log writes `validation {...}` records with `training_mse` and `validation_mean_standardized_rmse`, the format `97_forecast_curves.parse` reads.
* Summary shape: `forecast.best.by_lead[lead].by_year[year].scores` (`macro_z`, `TEMP.rmse_physical`, `SALT.rmse_physical`) and `.baseline_scores`; arrays `forecast_seed<seed>.npz` as `96_forecast_report.climatology` reads them.
* `150_wang_campaign`: `cohort_options(name, train_years)`, `queue(...)`, `completed`, `free_mb`, `write_json` depend only on job dicts; loading it by path has no side effects.
* Caches of `cesm2_vlong_uniform`: `/DATA2/linke/ocean_project_ka_dev6/outputs/cohort_cesm2_vlong_uniform/` (5.5 GB) and `.../outputs/audit/synthetic/fixed_baselines_cesm2_vlong_uniform/`, the same filesystem as this checkout. `/DATA2` had 65 GB free; nine runs need about 4 GB.
* A run of A (x1, 154 years, 100k steps) took 2.85 h and peaked at 30 GB.

`$SCRATCH` is the session's scratchpad directory (throwaway files, never committed). Every commit is authored by Kevin (repo git config) and ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. `SUITE` below means:
`python -m pytest -q -p no:cacheprovider 2>&1 | tail -1` — expected `4 failed, N passed, 8 skipped` with only Poseidon failures (`... | grep ^FAILED | grep -v poseidon_backbone` prints nothing).

### Task 1: Hard-link the cohort caches

**Files:** none tracked (`outputs/` is ignored).

- [ ] **Step 1: Link**

```bash
SRC=/DATA2/linke/ocean_project_ka_dev6/outputs
mkdir -p outputs/audit/synthetic
cp -al $SRC/cohort_cesm2_vlong_uniform outputs/cohort_cesm2_vlong_uniform
cp -al $SRC/audit/synthetic/fixed_baselines_cesm2_vlong_uniform outputs/audit/synthetic/fixed_baselines_cesm2_vlong_uniform
```

- [ ] **Step 2: Verify they are the same files, not copies**

Run: `for f in satellites lagged_surface surface_grids; do stat -c '%h %i %n' outputs/cohort_cesm2_vlong_uniform/$f.npz; done; df -h /DATA2 | tail -1`
Expected: link count `2` on each, and free space unchanged (about 65 GB).

### Task 2: `--test-years` in runner 101

**Files:**
- Modify: `experiments/synthetic/101_grid_ka_forecast.py` (docstring options, `arguments()`, a helper after `selection_score`, `restore_training_inputs`, `main()` checks and both final scorings)
- Test: `tests/test_grid_runner_resume.py`

- [ ] **Step 1: Failing tests** — append to `tests/test_grid_runner_resume.py`:

```python
def test_a_run_from_before_test_years_restores_none():
    args = argparse.Namespace(satellite_cache=None)
    runner.restore_training_inputs(args, training())
    assert args.test_years == []
    runner.restore_training_inputs(args, training(val_years=[2004], test_years=[2005]))
    assert args.val_years == [2004] and args.test_years == [2005]


def test_the_final_scoring_covers_validation_and_test_years_and_refuses_an_overlap():
    assert runner.scored_years([2004], [2005]) == [2004, 2005]
    assert runner.scored_years([2004, 2005], []) == [2004, 2005]
    with pytest.raises(ValueError):
        runner.scored_years([2004, 2005], [2005])


def test_test_years_default_to_none_and_snapshots_off(monkeypatch):
    monkeypatch.setattr("sys.argv", ["101", "--tag", "t"])
    args = runner.arguments()
    assert args.test_years == [] and args.val_years == [2004, 2005] and args.keep_every == 0
```

- [ ] **Step 2:** `python -m pytest -q -p no:cacheprovider tests/test_grid_runner_resume.py` — expected: the 3 new tests FAIL (`AttributeError` on `test_years` / `scored_years`).

- [ ] **Step 3: Implement**

(a) Docstring, after the line `  --state-channels, --encoder-channels, --evolution-channels, --hidden   model size`:
```
  --val-years Y..     target years that choose the checkpoint and are scored at the end
  --test-years Y..    target years held out of that choice, scored only at the end (none by default)
```
(b) `arguments()`, after the `--val-years` option:
```python
    p.add_argument("--test-years", nargs="+", type=int, default=[],
                   help="target years held out of the choice, scored once at the end")
```
(c) After `selection_score`:
```python
def scored_years(val_years, test_years):
    """The target years the final scoring covers: the validation years and the held-out test years."""
    if set(val_years) & set(test_years):
        raise ValueError("a test year cannot also choose the checkpoint")
    return sorted(set(val_years) | set(test_years))
```
(d) `restore_training_inputs`, after `args.leads, args.val_years = training["leads"], training["val_years"]`:
```python
    args.test_years = training.get("test_years", [])
```
(e) `main()`, after the `--development` refusal:
```python
    scored_years(args.val_years, args.test_years)   # refuse an overlap before any data is read
```
(f) Both `full = data.forecast_months(args.val_years, 0)` become
`full = data.forecast_months(scored_years(args.val_years, args.test_years), 0)`.

(g) `arguments()`, after `--checkpoint-every`:
```python
    p.add_argument("--keep-every", type=int, default=0,
                   help="also keep the model of every N-th step (step<NNNNNN>_seed<seed>.pt); 0: none")
```
and in the training loop, right after the `save_last(step)` block:
```python
            if args.keep_every and step % args.keep_every == 0:
                # kept, unlike last_seed: the model only, evaluable with --evaluate-only
                atomic_save(model_state(step, history[-1] if history else None), out/f"step{step:06d}_seed{args.seed}.pt")
```
and in the docstring option list:
```
  --keep-every N      also keep the model of every N-th step (default 0: only the best and the last)
```

`vars(args)` already goes into the summary's `training`, so `test_years` and `keep_every` are recorded there.

- [ ] **Step 4:** `python -m pytest -q -p no:cacheprovider tests/test_grid_runner_resume.py tests/test_grid_kriging_ocean.py tests/test_grid_softmax_readout.py` — expected: all pass. Then `SUITE`.
- [ ] **Step 5: Commit**

```bash
git add experiments/synthetic/101_grid_ka_forecast.py tests/test_grid_runner_resume.py
git commit -q -m "101_grid_ka_forecast: --test-years (held out of the checkpoint choice, scored at the end, as 94); --keep-every snapshots" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 3: The queue `169_size_campaign.py`

**Files:**
- Create: `experiments/synthetic/169_size_campaign.py`
- Modify: `.gitignore` (add `/outputs/size_20261023/` under the KA block)
- Test: `tests/test_size_campaign.py`

- [ ] **Step 1: Failing tests** — create `tests/test_size_campaign.py`:

```python
"""The size study: the queue's runs, the curves' reader and figure, the report's decision."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments/synthetic" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


campaign = script("169_size_campaign")


def after(command, flag, n=1):
    i = command.index(flag)
    return command[i + 1:i + 1 + n]


def test_three_sizes_scale_every_channel_count():
    c = {arm: campaign.job(arm, 1234)["command"] for arm in campaign.FACTORS}
    assert after(c["y100_x1"], "--encoder-channels", 4) == ["64", "128", "192", "256"]
    assert after(c["y100_x1375"], "--state-channels") == ["176"] and after(c["y100_x1375"], "--hidden") == ["352"]
    assert after(c["y100_x175"], "--encoder-channels", 4) == ["112", "224", "336", "448"]
    assert after(c["y100_x175"], "--evolution-channels", 3) == ["112", "224", "336"]


def test_seeds_are_outermost():
    tags = [j["tag"] for j in campaign.jobs(list(campaign.FACTORS), [1234, 1235])]
    assert tags == ["y100_x1_s1234", "y100_x1375_s1234", "y100_x175_s1234",
                    "y100_x1_s1235", "y100_x1375_s1235", "y100_x175_s1235"]


def test_runs_train_100_years_for_200k_steps_chosen_on_2004_with_2005_held_out():
    command = campaign.job("y100_x1", 1234)["command"]
    assert after(command, "--cohort") == ["cesm2_vlong_uniform"] and after(command, "--train-years") == ["100"]
    assert after(command, "--steps") == ["200000"] and after(command, "--leads", 4) == ["0", "1", "3", "6"]
    assert after(command, "--val-years", 2) == ["2004", "--test-years"] and after(command, "--test-years") == ["2005"]
    assert after(command, "--kriging") == ["ka_flow"] and "--surface" in command
    assert after(command, "--keep-every") == ["5000"] and after(command, "--checkpoint-every") == ["500"]


def test_a_smoke_run_is_short_and_kept_apart():
    item = campaign.job("y100_x175", 1234, smoke=True)
    assert item["steps"] == 600 and Path(item["output"]).parent.name == "smoke"
    assert after(item["command"], "--val-every") == ["300"]
```

- [ ] **Step 2:** `python -m pytest -q -p no:cacheprovider tests/test_size_campaign.py` — expected: FAIL (`FileNotFoundError`, no 169).

- [ ] **Step 3: Implement** — create `experiments/synthetic/169_size_campaign.py`:

```python
"""Queue of the size study (spec ``docs/superpowers/specs/2026-10-23-ka-size-study-design.md``):
KA-Ocean (``101``, method A) on ``cesm2_vlong_uniform`` with a 100-year training window (1904-2003),
200,000 steps, leads 0/1/3/6, the checkpoint chosen on 2004 and 2005 held out, at three sizes, one
width factor on every channel count of A:

  y100_x1      9.4M parameters (A's size)
  y100_x1375  17.7M
  y100_x175   28.6M

Seeds are outermost, so every size has seed 1234 before a second seed starts. Every run keeps a model
snapshot every 5,000 steps; its resume point is every 500. The queue (a GPU memory check before each
launch, retries from the last resume point, finished runs skipped) is ``150``'s.

  .venv/bin/python experiments/synthetic/169_size_campaign.py --smoke --seeds 1234 --gpus 0,2,3
  .venv/bin/python experiments/synthetic/169_size_campaign.py --gpus 0,2,3,4,6 --per-gpu 1
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/size_20261023"
RUNNER = ROOT / "experiments/synthetic/101_grid_ka_forecast.py"
STEPS, SMOKE_STEPS = 200000, 600
SEEDS = (1234, 1235, 1236)


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments/synthetic" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_queue = _script("150_wang_campaign")
VLONG, VLONG_NEEDS = _queue.cohort_options("cesm2_vlong_uniform", 100)
COMMON = ["--surface", "--queries", "1024", "--amp", "--leads", "0", "1", "3", "6",
          "--val-years", "2004", "--test-years", "2005", "--kriging", "ka_flow"]
#: arm -> width factor on A's channel counts
FACTORS = {"y100_x1": 1., "y100_x1375": 1.375, "y100_x175": 1.75}


def size_options(factor):
    width = lambda channels: [str(round(c * factor)) for c in channels]
    return ["--state-channels", *width([128]), "--encoder-channels", *width([64, 128, 192, 256]),
            "--evolution-channels", *width([64, 128, 192]), "--hidden", *width([256])]


def job(arm, seed, smoke=False):
    steps, every = (SMOKE_STEPS, 300) if smoke else (STEPS, 2000)
    folder = (OUTPUT / "smoke" if smoke else OUTPUT) / f"{arm}_s{seed}"
    command = [sys.executable, "-u", str(RUNNER), *COMMON, *VLONG, *size_options(FACTORS[arm]),
               "--steps", str(steps), "--val-every", str(every), "--checkpoint-every", str(min(every, 500)),
               "--keep-every", str(300 if smoke else 5000),
               "--seed", str(seed), "--tag", f"{arm}_s{seed}", "--output", str(folder)]
    return {"tag": f"{arm}_s{seed}", "arm": arm, "seed": seed, "steps": steps, "output": str(folder),
            "needs": [str(p) for p in VLONG_NEEDS], "command": command}


def jobs(arms, seeds, smoke=False):
    return [job(arm, seed, smoke) for seed in seeds for arm in arms]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", nargs="+", choices=list(FACTORS), default=list(FACTORS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gpus", default="0,2,3,4,6")
    parser.add_argument("--per-gpu", type=int, default=1, help="workers sharing one GPU")
    parser.add_argument("--min-free-mb", type=int, default=40000, help="free GPU memory a launch needs")
    parser.add_argument("--retry-after", type=float, default=600.)
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--stagger", type=float, default=120., help="seconds between two launches")
    args = parser.parse_args()
    name = "smoke" if args.smoke else "queue"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / f"{name}.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another queue is running")
    requested = jobs(args.arms, args.seeds, args.smoke)
    places = [(int(gpu), worker) for gpu in args.gpus.split(",") for worker in range(args.per_gpu)]
    pending = [j for j in requested if not _queue.completed(j)]
    started = time.time()

    def status(extra):
        _queue.write_json(OUTPUT / f"status_{name}.json", {"requested": len(requested),
            "completed": sum(_queue.completed(j) for j in requested), "started_unix": started,
            "updated_unix": time.time(), **extra})

    if not args.smoke:
        _queue.write_json(OUTPUT / "campaign.json", {"seeds": args.seeds, "steps": STEPS, "factors": FACTORS,
            "options": {arm: COMMON + VLONG + size_options(f) for arm, f in FACTORS.items()},
            "runner": str(RUNNER.relative_to(ROOT)), "jobs": [j["tag"] for j in requested]})
    print(json.dumps({"requested": len(requested), "to_run": len(pending),
                      "places": [f"{gpu}.{worker}" for gpu, worker in places]}), flush=True)
    try:
        _queue.queue(pending, places, args.stagger, status, min_free_mb=args.min_free_mb,
                     retry_after=args.retry_after, attempts_allowed=args.attempts)
        status({"running": {}, "pending": [], "finished": True, "runtime_s": time.time() - started})
    except Exception as error:
        status({"failure": repr(error)})
        raise


if __name__ == "__main__":
    main()
```

`.gitignore`: after `/external/poseidon_deps/` add `/outputs/size_20261023/`.

- [ ] **Step 4:** `python -m pytest -q -p no:cacheprovider tests/test_size_campaign.py` — expected: 4 passed. Then `SUITE`.
- [ ] **Step 5: Commit**

```bash
git add experiments/synthetic/169_size_campaign.py tests/test_size_campaign.py .gitignore
git commit -q -m "169_size_campaign: KA-Ocean at 9.4M/17.7M/28.6M on 100 CESM2 years, 200k steps, 2005 held out; 150's queue" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 4: Smoke runs (timing and memory)

- [ ] **Step 1:** Check the GPUs: `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader`. Pick three idle GPUs (used < 1 GB, 0 %).
- [ ] **Step 2:** `.venv/bin/python experiments/synthetic/169_size_campaign.py --smoke --seeds 1234 --gpus <three idle> --stagger 10 2>&1 | tail -5` (foreground; timeout 2 h).
Expected: three `"completed"` lines; `outputs/size_20261023/smoke/y100_x*_s1234/summary_seed1234.json` exist, and each smoke folder has `step000300_seed1234.pt` and `step000600_seed1234.pt` (the snapshots).
- [ ] **Step 3: Read the numbers**

```bash
python - <<'EOF'
import json
from pathlib import Path
base = None
for arm in ("y100_x1", "y100_x1375", "y100_x175"):
    folder = Path(f"outputs/size_20261023/smoke/{arm}_s1234")
    s = json.loads((folder / "summary_seed1234.json").read_text())
    t = [r["elapsed_s"] for r in s["history"]]
    block = t[1] - t[0]                                  # 300 steps and one validation
    base = base or block
    print(arm, f"params {s['params']/1e6:.1f}M  peak {s['peak_gpu_memory_mb']/1024:.1f} GB  "
               f"300-step block {block:.0f} s  est. {5.7 * block / base:.1f} h a run (x1 = 5.7 h)")
EOF
```
Expected: params 9.4 / 17.7 / 28.6M (the spec's), finite peaks. Decide `--per-gpu`: 2 if the largest peak is at most 40 GB, else 1; `--min-free-mb` = the largest peak + 8 GB (in MiB). Record both in the commit message of Task 5.

### Task 5: Launch the queue

- [ ] **Step 1:** Check the GPUs again (Task 4 Step 1); use every idle one.
- [ ] **Step 2: Start it detached, so it outlives this session**

```bash
nohup setsid .venv/bin/python experiments/synthetic/169_size_campaign.py --gpus <idle list> \
  --per-gpu <P> --min-free-mb <M> > outputs/size_20261023/queue.log 2>&1 < /dev/null &
```
- [ ] **Step 3:** After about 10 minutes: `tail -3 outputs/size_20261023/queue.log; cat outputs/size_20261023/status_queue.json | head -20; grep -h '^validation\|Traceback\|Error' outputs/size_20261023/y100_x*_s1234/train.log | tail -5`
Expected: `"started"` lines for the first jobs, `running` in the status, no `Traceback`. A first `validation` record appears after 2,000 steps. `df -h /DATA2 | tail -1`: note the free space; the snapshots need about 27 GB over the campaign.
- [ ] **Step 4:** Commit nothing (outputs are ignored); report to Kevin: which GPUs, the per-run estimate, when seed 1234 of every size should finish.

### Task 6: The convergence figure `171_size_curves.py`

**Files:**
- Create: `experiments/synthetic/171_size_curves.py`
- Test: `tests/test_size_campaign.py`

- [ ] **Step 1: Failing tests** — append to `tests/test_size_campaign.py`:

```python
curves = script("171_size_curves")


def log(folder, records):
    folder.mkdir(parents=True)
    (folder / "train.log").write_text("".join(
        'validation {"step": %d, "training_mse": %r, "validation_mean_standardized_rmse": %r}\n' % r for r in records))


def test_curve_runs_are_the_size_studys_by_name(tmp_path):
    for name in ("y100_x1_s1234", "y100_x175_s1235", "smoke", "c_y154_s1234"):
        log(tmp_path / name, [(2000, 1.0, .9)])
    assert [(r["arm"], r["seed"]) for r in curves.runs(tmp_path)] == [("y100_x1", 1234), ("y100_x175", 1235)]


def test_a_best_step_in_the_last_tenth_is_still_improving(tmp_path):
    steps = range(2000, 200001, 2000)
    log(tmp_path / "y100_x1_s1234", [(s, 1.0, 1 - s / 1e6) for s in steps])          # best at 200k
    log(tmp_path / "y100_x1375_s1234", [(s, 1.0, abs(s - 100000) / 1e6) for s in steps])   # best at 100k
    rows = {r["tag"]: r for r in curves.rows(curves.runs(tmp_path))}
    assert rows["y100_x1_s1234"]["best_step"] == 200000 and rows["y100_x1_s1234"]["still_improving"]
    assert rows["y100_x1375_s1234"]["best_step"] == 100000 and not rows["y100_x1375_s1234"]["still_improving"]
    assert rows["y100_x1_s1234"]["tail_change_pct"] == 0.0


def test_the_figure_is_drawn_from_unfinished_runs_too(tmp_path):
    log(tmp_path / "y100_x175_s1234", [(s, 2.0 / s ** .1, .8 + 1 / s ** .5) for s in range(2000, 40001, 2000)])
    path = tmp_path / "curves.png"
    curves.plot(curves.runs(tmp_path), path)
    assert path.stat().st_size > 10000
```

- [ ] **Step 2:** `python -m pytest -q -p no:cacheprovider tests/test_size_campaign.py` — expected: the 3 new tests FAIL (no 171).

- [ ] **Step 3: Implement** — create `experiments/synthetic/171_size_curves.py`:

```python
"""Training curves of the size study (``169_size_campaign.py``): is each size converged at 200k steps?

Reads ``train.log`` of every run under ``outputs/size_20261023`` (``y100_x<factor>_s<seed>``) with
``97_forecast_curves.parse`` and draws ``reports/synthetic/size_20261023_curves.png``: one column per
size; the training MSE (log scale) on top, the mean of every step since the previous record; the
2004 validation score below (mean standardised RMSE over leads, the checkpoint criterion); one line
per seed in the size's colour; each seed's best step a ringed dot; 180k, the start of the last
tenth, a dotted line. It prints, per run, the last step logged, the change of the training MSE over
the last tenth against the tenth before (``97_forecast_curves.tail_change``), and the best
validation step and score; a best step inside the last tenth means the run was still improving.
Run it at any time; unfinished runs are drawn as far as they have got.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/size_20261023"
FIGURE = ROOT / "reports/synthetic/size_20261023_curves.png"
STEPS = 200000
RUN = re.compile(r"^(y100_x\d+)_s(\d+)$")
#: size -> (title, colour): an ordered tier, so one blue ramp light -> dark (validate_palette --ordinal passes)
SIZES = {"y100_x1": ("x1 · 9.4M", "#6da7ec"), "y100_x1375": ("x1.375 · 17.7M", "#2a78d6"),
         "y100_x175": ("x1.75 · 28.6M", "#184f95")}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments/synthetic" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


curves = _script("97_forecast_curves")


def runs(output=OUTPUT):
    found = []
    for folder in sorted(output.iterdir()) if output.exists() else []:
        match = RUN.match(folder.name)
        if match and match[1] in SIZES and (folder / "train.log").exists():
            found.append({"tag": folder.name, "arm": match[1], "seed": int(match[2]), "log": folder / "train.log"})
    return found


def rows(found):
    out = []
    for run in found:
        train, validation = curves.parse(run["log"])
        best = min(validation, key=validation.get) if validation else None
        out.append({"tag": run["tag"], "arm": run["arm"], "seed": run["seed"], "last_step": max(validation, default=0),
                    "tail_change_pct": curves.tail_change(train, STEPS), "best_step": best,
                    "best_score": validation.get(best), "still_improving": best is not None and best > STEPS - STEPS // 10})
    return out


def plot(found, path=FIGURE):
    fig, axes = plt.subplots(2, len(SIZES), figsize=(12, 6.4), sharex=True, sharey="row", facecolor=SURFACE)
    for column, (arm, (title, colour)) in enumerate(SIZES.items()):
        top, bottom = axes[0, column], axes[1, column]
        for run in (r for r in found if r["arm"] == arm):
            train, validation = curves.parse(run["log"])
            for ax, values in ((top, train), (bottom, validation)):
                steps = sorted(values)
                ax.plot(np.array(steps) / 1000, [values[s] for s in steps], color=colour, lw=1.0)   # 2 px at 150 dpi
            if validation:
                best = min(validation, key=validation.get)
                bottom.plot(best / 1000, validation[best], "o", ms=5, color=colour, mec=SURFACE, mew=1.0, zorder=3)
        top.set_title(title, color=INK, fontsize=11)
        top.set_yscale("log")
        for ax in (top, bottom):
            ax.set_facecolor(SURFACE)
            ax.axvline((STEPS - STEPS // 10) / 1000, color=INK_2, lw=.8, ls=":")
            ax.grid(color=GRID, lw=.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.tick_params(colors=INK_2, labelsize=9)
        bottom.set_xlabel("step (thousands)", color=INK_2)
        bottom.set_xlim(0, STEPS / 1000)
    axes[0, 0].set_ylabel("training MSE (log)", color=INK_2)
    axes[1, 0].set_ylabel("2004 validation score\n(mean standardised RMSE over leads)", color=INK_2)
    handles = [Line2D([], [], color=c, lw=1.0, label=t) for t, c in SIZES.values()]
    handles += [Line2D([], [], ls="", marker="o", ms=5, color=INK_2, mec=SURFACE, label="best validation step"),
                Line2D([], [], color=INK_2, lw=.8, ls=":", label="180k: start of the last tenth")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, labelcolor=INK_2, fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    found = runs()
    for r in rows(found):
        tail = "running" if r["tail_change_pct"] is None else f"training MSE {r['tail_change_pct']:+.1f} % over the last tenth"
        best = "no validation yet" if r["best_step"] is None else f"best {r['best_score']:.4f} at {r['best_step']}"
        flag = " (still improving)" if r["still_improving"] and r["last_step"] >= STEPS else ""
        print(f"{r['tag']:20} step {r['last_step']:>6}  {tail:42} {best}{flag}", flush=True)
    plot(found)
    print(FIGURE.relative_to(ROOT))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4:** `python -m pytest -q -p no:cacheprovider tests/test_size_campaign.py` — expected: 7 passed.
- [ ] **Step 5: Look at it (dataviz step 7).** Render the test's synthetic figure to `$SCRATCH/curves_check.png` (scratchpad) with three fake sizes x three seeds and open it with the Read tool: no label collisions, legend below the panels, shared row axes, dots ringed, the dotted line at 180k. Fix and re-render until it reads cleanly.
- [ ] **Step 6:** `SUITE`, then commit:

```bash
git add experiments/synthetic/171_size_curves.py tests/test_size_campaign.py
git commit -q -m "171_size_curves: the size study's training-MSE and validation curves, one column per size" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 7: The report `170_size_report.py`

**Files:**
- Create: `experiments/synthetic/170_size_report.py`
- Test: `tests/test_size_campaign.py`

- [ ] **Step 1: Failing tests** — append:

```python
import json
import pytest

report = script("170_size_report")


def fake_summary(macro_by_year, seed=1234):
    year = lambda m: {"scores": {"macro_z": m, "TEMP": {"rmse_physical": m}, "SALT": {"rmse_physical": m / 10}},
                      "baseline_scores": {"TEMP": {"rmse_physical": 1.}, "SALT": {"rmse_physical": .1}}}
    by_lead = {lead: {"by_year": {y: year(m) for y, m in macro_by_year.items()}} for lead in ("0", "1", "3", "6")}
    return {"seed": seed, "completed_steps": 200000, "forecast": {"best": {"by_lead": by_lead}}}


def test_the_score_is_the_mean_over_leads_of_a_year():
    assert report.score(fake_summary({"2004": .8, "2005": .7}), "2005") == pytest.approx(.7)


def test_bigger_helps_only_beyond_the_seed_spread():
    scores = {"y100_x1": {"2004": [.80, .81, .79], "2005": [.80, .81, .79]},
              "y100_x1375": {"2004": [.78, .78, .78], "2005": [.78, .78, .78]},
              "y100_x175": {"2004": [.77, .78, .76], "2005": [.77, .78, .76]}}
    assert report.decide(scores)["bigger_helps"]                       # 0.03 lower, spread 0.01
    scores["y100_x175"]["2005"] = [.795, .80, .79]
    assert not report.decide(scores)["bigger_helps"]                   # 0.005 lower, within the spread


def test_the_best_size_is_chosen_on_2004_not_on_the_held_out_year():
    scores = {"y100_x1": {"2004": [.80], "2005": [.70]}, "y100_x1375": {"2004": [.75], "2005": [.79]},
              "y100_x175": {"2004": [.78], "2005": [.71]}}
    assert report.decide(scores)["best"] == "y100_x1375"


def test_report_reads_only_finished_runs(tmp_path):
    for name, steps in (("y100_x1_s1234", 200000), ("y100_x175_s1234", 150000)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "summary_seed1234.json").write_text(json.dumps({**fake_summary({"2004": .8}), "completed_steps": steps}))
    assert list(report.runs(tmp_path)) == ["y100_x1"]
```

- [ ] **Step 2:** `python -m pytest -q -p no:cacheprovider tests/test_size_campaign.py` — expected: the 4 new tests FAIL (no 170).

- [ ] **Step 3: Implement** — create `experiments/synthetic/170_size_report.py`:

```python
"""Report of the size study (spec ``docs/superpowers/specs/2026-10-23-ka-size-study-design.md``).

Reads every finished ``summary_seed*.json`` (200,000 steps) under ``outputs/size_20261023`` and writes
``reports/synthetic/size_20261023.md`` and ``.json``:

  sizes     parameters, hours a run, best steps, and the score -- the mean over leads of the
            per-variable standardised RMSE of the chosen checkpoint on every cell -- on 2004, which
            chose the checkpoints, and on 2005, held out of that choice; mean ± sd over seeds
  decision  "bigger helps" if x1.75's 2005 score is below x1's by more than the larger of their seed
            standard deviations; the best size is the lowest 2004 score (choosing on 2005 would end
            its being held out); computed once every size has all three seeds
  by lead   every size's 2005 RMSE at leads 0, 1, 3 and 6 months, temperature (°C) and salinity (PSU),
            beside persisted OI and climatology (a zero anomaly); the best size's rows are the forecast

The convergence curves are ``171_size_curves.py``'s figure, linked here. Runs still training are left
out. Run it at any time.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/size_20261023"
REPORT = ROOT / "reports/synthetic/size_20261023"
STEPS, SEEDS = 200000, 3
SELECT, HELD_OUT = "2004", "2005"
CH = ("TEMP", "SALT")
UNITS = {"TEMP": "°C", "SALT": "PSU"}


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments/synthetic" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


forecast_report = _script("96_forecast_report")       # climatology, mean_sd
size_curves = _script("171_size_curves")              # SIZES, FIGURE
SIZES = size_curves.SIZES


def runs(output=OUTPUT):
    found = {}
    for path in sorted(output.glob("y100_x*_s*/summary_seed*.json")):
        arm = path.parent.name.rsplit("_s", 1)[0]
        summary = json.loads(path.read_text())
        if arm in SIZES and summary.get("completed_steps") == STEPS and "forecast" in summary:
            summary["arrays"] = path.parent / f"forecast_seed{summary['seed']}.npz"
            found.setdefault(arm, []).append(summary)
    return {arm: found[arm] for arm in SIZES if arm in found}


def score(summary, year):
    """The chosen checkpoint's mean over leads of the per-variable standardised RMSE on every cell of ``year``."""
    by_lead = summary["forecast"]["best"]["by_lead"]
    return float(np.mean([v["by_year"][year]["scores"]["macro_z"] for v in by_lead.values()]))


def decide(scores):
    """``scores`` {arm: {year: [one per seed]}} -> the spec's decision."""
    m = {arm: {y: forecast_report.mean_sd(v) for y, v in s.items()} for arm, s in scores.items()}
    small, large = m["y100_x1"][HELD_OUT], m["y100_x175"][HELD_OUT]
    spread = max(small[1] or 0., large[1] or 0.)
    return {"bigger_helps": bool(large[0] < small[0] - spread), "gain": small[0] - large[0], "seed_spread": spread,
            "best": min(m, key=lambda arm: m[arm][SELECT][0])}


def by_lead(summaries, year=HELD_OUT):
    leads = sorted(summaries[0]["forecast"]["best"]["by_lead"], key=int)
    clim = forecast_report.climatology(summaries[0]["arrays"])      # every run is scored on the same cells
    first = summaries[0]["forecast"]["best"]["by_lead"]
    return {lead: {**{ch: forecast_report.mean_sd([s["forecast"]["best"]["by_lead"][lead]["by_year"][year]["scores"][ch]["rmse_physical"]
                                                   for s in summaries]) for ch in CH},
                   **{f"oi_{ch}": first[lead]["by_year"][year]["baseline_scores"][ch]["rmse_physical"] for ch in CH},
                   **{f"clim_{ch}": clim[int(lead)]["by_year"][year][ch] for ch in CH}} for lead in leads}


def text(value, digits=4):
    mean, sd = value
    return f"{mean:.{digits}f}" + ("" if sd is None else f" ± {sd:.{digits}f}")


def markdown(found, scores, decision, leads):
    out = ["# KA-Ocean model size on 100 CESM2 years", "",
           "KA-Ocean (method A: U-Net state, lead evolution, `ka_flow` kriging read-out) on `cesm2_vlong_uniform`, "
           "training window 1904-2003, 200,000 steps of 1,024 queries, leads 0, 1, 3 and 6 months in one model. "
           "Each run's checkpoint is chosen on target year 2004; 2005 is held out of that choice. "
           "Spec: `docs/superpowers/specs/2026-10-23-ka-size-study-design.md`.", "",
           "Seeds finished: " + ", ".join(f"{SIZES[a][0]} {len(found.get(a, []))}" for a in SIZES) + ".", "",
           "## Sizes", "", "| size | parameters | hours a run | best steps | 2004 score (chose) | 2005 score (held out) |",
           "|---|---|---|---|---|---|"]
    for arm, summaries in found.items():
        hours = forecast_report.mean_sd([s["runtime_s"] / 3600 for s in summaries])
        out.append(f"| {SIZES[arm][0]} | {summaries[0]['params'] / 1e6:.1f}M | {text(hours, 1)} | "
                   f"{', '.join(str(s['best_step']) for s in summaries)} | {text(forecast_report.mean_sd(scores[arm][SELECT]))} | "
                   f"{text(forecast_report.mean_sd(scores[arm][HELD_OUT]))} |")
    out += ["", "Score: mean over leads of the per-variable standardised RMSE on every cell, mean ± sd over seeds.", "", "## Decision", ""]
    if decision is None:
        out.append("Waits until every size has all three seeds.")
    else:
        out += [f"* Bigger helps: **{'yes' if decision['bigger_helps'] else 'no'}** -- x1.75's 2005 score is "
                f"{decision['gain']:+.4f} below x1's; the larger seed sd is {decision['seed_spread']:.4f}.",
                f"* Best size (lowest 2004 score): **{SIZES[decision['best']][0]}**."]
    for ch in (CH if leads else ()):                  # nothing finished yet: no lead tables
        out += ["", f"## 2005 by lead: {'temperature' if ch == 'TEMP' else 'salinity'} RMSE ({UNITS[ch]})", "",
                "| lead | persisted OI | climatology | " + " | ".join(SIZES[a][0] for a in leads) + " |",
                "|---|---|---|" + "---|" * len(leads)]
        first = next(iter(leads.values()))
        for lead in first:
            cells = [text(leads[a][lead][ch]) + (" **best**" if decision and a == decision["best"] else "") for a in leads]
            out.append(f"| {lead} | {first[lead][f'oi_{ch}']:.4f} | {first[lead][f'clim_{ch}']:.4f} | " + " | ".join(cells) + " |")
    out += ["", "## Convergence", "", f"![training curves]({size_curves.FIGURE.name})", "",
            "`171_size_curves.py`: training MSE and the 2004 validation score per run; a best step after 180k means still improving."]
    return "\n".join(out) + "\n"


def main():
    found = runs()
    scores = {arm: {y: [score(s, y) for s in summaries] for y in (SELECT, HELD_OUT)} for arm, summaries in found.items()}
    complete = len(found) == len(SIZES) and all(len(v) == SEEDS for v in found.values())
    decision = decide(scores) if complete else None
    leads = {arm: by_lead(summaries) for arm, summaries in found.items()}
    REPORT.with_suffix(".md").write_text(markdown(found, scores, decision, leads))
    REPORT.with_suffix(".json").write_text(json.dumps({"scores": scores, "decision": decision, "by_lead": leads,
                                                       "seeds": {a: [s["seed"] for s in v] for a, v in found.items()}}, indent=2) + "\n")
    print(REPORT.with_suffix(".md").relative_to(ROOT))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4:** `python -m pytest -q -p no:cacheprovider tests/test_size_campaign.py` — expected: 11 passed. Then `SUITE`.
- [ ] **Step 5: Commit**

```bash
git add experiments/synthetic/170_size_report.py tests/test_size_campaign.py
git commit -q -m "170_size_report: sizes, the decision (2005 beyond seed spread; best on 2004), 2005 by lead" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 8: While the runs train, and when they finish

- [ ] **Step 1 (any time):** `.venv/bin/python experiments/synthetic/171_size_curves.py` and open `reports/synthetic/size_20261023_curves.png`; show Kevin the figure and the per-run lines.
- [ ] **Step 2 (all nine finished: `status_queue.json` has `"finished": true`):** run `171`, then `170`; read `reports/synthetic/size_20261023.md` and check the numbers against one summary by hand (one size, one lead, one variable).
- [ ] **Step 3:** Commit the figure and the report:

```bash
git add reports/synthetic/size_20261023.md reports/synthetic/size_20261023.json reports/synthetic/size_20261023_curves.png
git commit -q -m "Reports: KA-Ocean size study on 100 CESM2 years (curves, decision, 2005 forecast by lead)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
- [ ] **Step 4:** Report to Kevin: the decision, the best size's 2005 forecast by lead, the convergence reading per size. Merge `size-study` into `main` only on his word.
