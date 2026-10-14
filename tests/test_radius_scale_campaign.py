"""Job list and bookkeeping of the radius / local-scale campaign."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/synthetic/82_radius_scale_campaign.py"
SPEC = importlib.util.spec_from_file_location("radius_scale_campaign", SCRIPT)
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


def test_arm_names_spell_out_their_options():
    assert C.flags("control") == []
    assert C.flags("r500") == ["--local-radius-km", "500"]
    assert C.flags("qres_month") == ["--scale", "res_month", "--scale-by", "query"]
    assert C.flags("qanom_month_lossz") == ["--scale", "anom_month", "--scale-by", "query",
                                            "--loss-units", "z"]
    assert C.flags("r250+qres_cell") == ["--local-radius-km", "250", "--scale", "res_cell",
                                         "--scale-by", "query"]


def test_a_table_name_without_q_is_the_first_per_profile_design():
    assert C.flags("res_month") == ["--scale", "res_month", "--scale-by", "profile"]
    assert C.flags("anom_month_lossz") == ["--scale", "anom_month", "--scale-by", "profile",
                                           "--loss-units", "z"]


@pytest.mark.parametrize("arm", ["r0", "rabc", "res", "res_year", "r250+r500", "anom_cell+res_cell",
                                 "qres_cell+res_cell", "q", "qr250", "control+r250", ""])
def test_an_arm_name_that_is_not_understood_is_refused(arm):
    with pytest.raises(ValueError):
        C.flags(arm)


def test_a_job_is_the_reference_settings_plus_the_arm(tmp_path):
    job = C.job("r500+qres_month", "argo", 1235, tmp_path)
    assert job["tag"] == "r500_qres_month_argo_s1235"
    assert job["output"] == str(tmp_path / "r500_qres_month_argo_s1235")
    command = job["command"]
    assert command[2].endswith("experiments/synthetic/81_radius_scale_reconstruction.py")
    joined = " ".join(command)
    for setting in ("--variant local_transformer", "--width 192", "--latents 96", "--blocks 6",
                    "--query-blocks 2", "--heads 8", "--steps 15000", "--context-profiles 6080",
                    "--queries 1024", "--val-every 1000", "--checkpoint-every 500", "--amp",
                    "--seed 1235", "--local-radius-km 500", "--scale res_month --scale-by query"):
        assert setting in joined
    assert "--surface" not in command and "--development" not in command
    assert f"--scale-tables {tmp_path / 'scale_tables.npz'}" in joined


def test_the_surface_input_adds_the_matched_cache(tmp_path):
    command = C.job("control", "surface", 1234, tmp_path)["command"]
    assert command[command.index("--satellite-cache") + 1] == str(tmp_path / "satellites.npz")
    assert "--surface" in command
    assert "--scale-tables" not in command
    with pytest.raises(ValueError, match="inputs"):
        C.job("control", "sst", 1234, tmp_path)


def test_jobs_of_a_request_have_distinct_tags(tmp_path):
    jobs = C.jobs(["control", "r250", "qres_month"], ["argo", "surface"], [1234, 1235], tmp_path)
    assert len(jobs) == 12 and len({j["tag"] for j in jobs}) == 12
    assert [j["tag"] for j in jobs[:2]] == ["control_argo_s1234", "r250_argo_s1234"]


def test_only_a_full_length_run_counts_as_finished(tmp_path):
    job = C.job("r250", "argo", 1234, tmp_path)
    assert not C.completed(job)
    folder = Path(job["output"])
    folder.mkdir()
    (folder / "summary_seed1234.json").write_text(json.dumps({"seed": 1234, "completed_steps": 300}))
    with pytest.raises(ValueError, match="incomplete"):
        C.completed(job)
    (folder / "summary_seed1234.json").write_text(json.dumps({"seed": 1234, "completed_steps": 15000}))
    assert C.completed(job)


def test_an_interrupted_run_continues_from_its_last_state(tmp_path):
    job = C.job("r250", "argo", 1234, tmp_path)
    assert C.training_command(job) == job["command"]
    folder = Path(job["output"])
    folder.mkdir()
    (folder / "last_seed1234.pt").touch()
    assert C.training_command(job)[-3:] == ["--resume-training", "--load-checkpoint",
                                            str(folder / "last_seed1234.pt")]


def test_2005_is_scored_from_the_selected_checkpoint_only(tmp_path):
    job = C.job("r250", "argo", 1234, tmp_path)
    command = C.development_command(job)
    assert command[-4:] == ["--evaluate-only", "--development", "--load-checkpoint",
                            str(Path(job["output"]) / "best_seed1234.pt")]
    assert not C.scored_2005(job)
    folder = Path(job["output"])
    folder.mkdir()
    (folder / "summary_seed1234.json").write_text(json.dumps({"seed": 1234, "completed_steps": 15000}))
    assert not C.scored_2005(job)
    (folder / "summary_seed1234.json").write_text(json.dumps(
        {"seed": 1234, "completed_steps": 15000, "development": {"scores": {}}}))
    assert C.scored_2005(job)


def test_slots_put_several_workers_on_one_gpu():
    assert C.slots([6], 3) == [(6, 0), (6, 1), (6, 2)]
    assert C.slots([6, 7], 2) == [(6, 0), (6, 1), (7, 0), (7, 1)]
    with pytest.raises(ValueError):
        C.slots([6, 6], 1)


def test_a_named_cohort_gets_its_own_folder_baselines_and_tables(tmp_path):
    job = C.job("r500+qres_month", "argo", 1234, tmp_path, cohort="ihesp_hr_uniform")
    assert job["output"] == str(tmp_path / "ihesp_hr_uniform" / "r500_qres_month_argo_s1234")
    command = job["command"]
    assert command[command.index("--cohort") + 1] == "ihesp_hr_uniform"
    assert command[command.index("--oi-summary") + 1].endswith(
        "outputs/audit/synthetic/fixed_baselines_ihesp_hr_uniform/summary.json")
    assert command[command.index("--scale-tables") + 1] == str(tmp_path / "ihesp_hr_uniform" / "scale_tables.npz")
    plain = C.job("control", "argo", 1234, tmp_path)["command"]
    assert "--cohort" not in plain and "--oi-summary" not in plain


def test_the_surface_input_of_a_named_cohort_reads_that_cohorts_cache(tmp_path):
    command = C.job("control", "surface", 1234, tmp_path, cohort="ihesp_hr_uniform")["command"]
    assert "--surface" in command
    assert command[command.index("--satellite-cache") + 1] == str(tmp_path / "ihesp_hr_uniform" / "satellites.npz")


def test_the_number_of_latent_slots_is_an_arm_option(tmp_path):
    assert C.flags("l64") == ["--latents", "64"]
    assert C.flags("l64+qres_cell") == ["--latents", "64", "--scale", "res_cell", "--scale-by", "query"]
    assert C.flags("r500+l64+qres_cell")[:4] == ["--local-radius-km", "500", "--latents", "64"]
    job = C.job("l64+qres_cell", "surface", 1235, tmp_path)
    assert job["tag"] == "l64_qres_cell_surface_s1235"
    command = job["command"]
    # the reference's own slot count comes first, so the arm's value is the one the runner keeps
    assert [command[i + 1] for i, a in enumerate(command) if a == "--latents"] == ["96", "64"]
    for bad in ("l0", "l64+l32", "lx"):
        with pytest.raises(ValueError):
            C.flags(bad)
