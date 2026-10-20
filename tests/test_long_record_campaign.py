"""The long-record grid, its completion rule, the earlier-month sampling and the speed metric."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments/synthetic" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


campaign = script("90_long_record_campaign")
cohort = script("87_cesm2_long_cohort")
report = script("92_long_record_report")


def test_grid_is_36_runs_seeds_outermost_and_the_window_ends_first():
    jobs = campaign.jobs()
    assert len(jobs) == 36 and len({j["tag"] for j in jobs}) == 36
    assert [j["seed"] for j in jobs] == [1234] * 12 + [1235] * 12 + [1236] * 12
    assert [j["years"] for j in jobs[:12]] == [4] * 3 + [40] * 3 + [10] * 3 + [20] * 3


def test_sizes_grow_slots_and_width_together_on_the_registered_model():
    for job in campaign.jobs(seeds=[1]):
        command = job["command"]
        value = lambda flag: command[command.index(flag) + 1]
        assert all(command.count(flag) == 1 for flag in ("--latents", "--width", "--train-years"))
        assert (int(value("--latents")), int(value("--width"))) == {
            "s": (64, 192), "m": (128, 256), "l": (256, 384)}[job["size"]]
        assert int(value("--train-years")) == job["years"]
        assert value("--recipe") == "profile_shared" and value("--scale") == "res_cell"
        assert value("--heads") == "8" and value("--blocks") == "6" and value("--steps") == "15000"
        assert "--surface" in command and "--development" in command


def test_a_run_counts_only_with_the_full_budget_and_a_2005_score(tmp_path):
    job = campaign.job(10, "m", 1235, output=tmp_path)
    assert not campaign.completed(job)
    folder = Path(job["output"])
    folder.mkdir()
    summary = {"completed_steps": 15000, "seed": 1235, "train_years": 10, "development": {}}
    (folder / "summary_seed1235.json").write_text(json.dumps(summary))
    assert campaign.completed(job)
    (folder / "summary_seed1235.json").write_text(json.dumps({**summary, "completed_steps": 40}))
    with pytest.raises(ValueError):
        campaign.completed(job)


def grid():
    lat_c, lon_c = np.arange(-89.5, 90.), np.arange(.5, 360.)
    rng = np.random.default_rng(1)
    wet0 = rng.uniform(size=(180, 360)) > .3
    raw = {v: rng.normal(size=(20, 180, 360)).astype("float32") for v in ("TEMP", "SALT")}
    raw["TEMP"][:, 40:60, 100:140] = 0.
    raw["SALT"][:, 40:60, 100:140] = 0.
    fill = (raw["TEMP"] == 0) & (raw["SALT"] == 0)
    clean = {v: np.where(fill, np.nan, a) for v, a in raw.items()}
    clim = {v: rng.normal(size=(12, 20, 180, 360)).astype("float32") for v in ("TEMP", "SALT")}
    return raw, clean, clim, wet0, lat_c, lon_c


def test_earlier_months_are_drawn_like_41_and_reproducibly():
    raw, clean, clim, wet0, lat_c, lon_c = grid()
    columns, arrays = cohort.draw_month(-100, raw, clean, clim, wet0, lat_c, lon_c)
    again, _ = cohort.draw_month(-100, raw, clean, clim, wet0, lat_c, lon_c)
    other, _ = cohort.draw_month(-99, raw, clean, clim, wet0, lat_c, lon_c)
    assert np.array_equal(columns["lat"], again["lat"]) and not np.array_equal(columns["lat"], other["lat"])
    assert columns["lat"].size == 7600 and (columns["float_split"] == "heldout_float").sum() == 1520
    idx, _ = cohort.stencil(columns["lat"], columns["lon"], lat_c, lon_c)
    assert all(wet0[i, j].all() for i, j in idx)                 # every stencil corner is ocean
    assert np.all(columns["month_index"] == -100) and columns["wmo"][0] == "L3320000"
    assert np.array_equal(np.isnan(arrays["CLIM_CELL_TEMP"]), np.isnan(arrays["TEMP"]))
    touched = np.isnan(arrays["TEMP"]) & np.isfinite(arrays["TEMP_RAW"])
    assert touched.any()                                         # the fill is removed before interpolation
    assert all(a.dtype == np.float32 and a.shape == (7600, 20) for a in arrays.values())


def test_steps_to_reach_a_score_are_interpolated_between_validations():
    history = [{"step": 1000, "validation_mean_standardized_rmse": .5},
               {"step": 2000, "validation_mean_standardized_rmse": .3},
               {"step": 3000, "validation_mean_standardized_rmse": .2}]
    assert report.reach(history, .4) == pytest.approx(1500.)
    assert report.reach(history, .6) == 1000.
    assert report.reach(history, .1) is None


def test_a_longer_budget_gets_its_own_names_and_steps(tmp_path):
    items = campaign.jobs(years=(4, 40), seeds=(1234,), output=tmp_path, steps=30000)
    assert [j["tag"] for j in items] == [f"k30_y{y}_{s}_s1234" for y in (4, 40) for s in "sml"]
    for item in items:
        command = item["command"]
        assert command.count("--steps") == 1 and command[command.index("--steps") + 1] == "30000"
    assert campaign.job(4, "s", 1234, output=tmp_path)["tag"] == "y4_s_s1234"
    folder = Path(items[0]["output"])
    folder.mkdir()
    summary = {"completed_steps": 15000, "seed": 1234, "train_years": 4, "development": {}}
    (folder / "summary_seed1234.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        campaign.completed(items[0])
    (folder / "summary_seed1234.json").write_text(json.dumps({**summary, "completed_steps": 30000}))
    assert campaign.completed(items[0])


def fake_summary(seed, steps, temp, salt, years):
    scores = {"TEMP": {"rmse_physical": temp}, "SALT": {"rmse_physical": salt}, "macro_z": temp * 4}
    history = [{"step": s, "validation_mean_standardized_rmse": temp * 4 + (steps - s) / 1e5}
               for s in range(1000, steps + 1, 1000)]
    return {"seed": seed, "completed_steps": steps, "train_years": years, "params": 5_000_000,
            "best_step": steps, "history": history, "runtime_s": 7200., "peak_gpu_memory_mb": 4096.,
            "validation": {"scores": scores},
            "development": {"scores": scores,
                            "baseline_scores": {"TEMP": {"rmse_physical": .086}, "SALT": {"rmse_physical": .0226}}}}


def test_the_report_puts_the_30k_runs_against_the_15k_cells(tmp_path):
    for tag, steps, temp in (("y4_s_s1", 15000, .066), ("y40_s_s1", 15000, .066),
                             ("k30_y4_s_s1", 30000, .064), ("k30_y40_s_s1", 30000, .062)):
        folder = tmp_path / tag
        folder.mkdir()
        years = int(tag.split("y")[1].split("_")[0])
        (folder / "summary_seed1.json").write_text(json.dumps(fake_summary(1, steps, temp, .016, years)))
    short, oi, target = report.collect(report.runs(tmp_path))
    long, _, _ = report.collect(report.runs(tmp_path, steps=30000), target, report.LONG_CURVE_STEPS)
    assert set(short) == {(4, "s"), (40, "s")} and set(long) == {(4, "s"), (40, "s")}
    assert long[(40, "s")]["test_TEMP"][0] == .062 and short[(40, "s")]["test_TEMP"][0] == .066
    text = report.markdown_long(long, short, oi, target)
    assert "-6.1 %" in text                      # 40 years at 30k against 40 years at 15k
    assert "| 40 y, s |" in text and "30000" in text
