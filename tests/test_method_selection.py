"""The method-selection queue (114) and the rule that chooses the final method (115)."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments/real_data" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


campaign = script("114_method_selection_campaign")
selection = script("115_method_selection")


def value(command, flag):
    assert command.count(flag) == 1
    return command[command.index(flag) + 1]


def test_each_arm_runs_its_candidate_at_the_setup_of_its_dataset():
    prior, poseidon, random = (campaign.job(arm, 1234) for arm in ("rb_prior", "rc_poseidon", "pr_random"))
    for item in (prior, poseidon):
        command = item["command"]
        assert value(command, "--dataset") == "real_global" and value(command, "--steps") == "30000"
        assert value(command, "--val-years") == "2021" and value(command, "--kriging") == "ka_flow"
    assert (value(prior["command"], "--background"), value(prior["command"], "--backbone")) == ("prior", "unet")
    assert (value(poseidon["command"], "--background"), value(poseidon["command"], "--backbone"),
            value(poseidon["command"], "--poseidon-init")) == ("joint", "poseidon", "pretrained")
    command = random["command"]
    assert "--dataset" not in command and value(command, "--steps") == "70000"
    assert value(command, "--cohort") == "cesm2_long_uniform" and value(command, "--poseidon-init") == "random"
    assert command[command.index("--val-years") + 1:command.index("--val-years") + 3] == ["2004", "2005"]


def test_a_failed_run_resumes_from_its_last_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "OUTPUT", tmp_path)
    item = campaign.job("rb_prior", 1235)
    assert "--resume-training" not in campaign.command_for(item)
    Path(item["output"]).mkdir(parents=True)
    (Path(item["output"]) / "last_seed1235.pt").write_bytes(b"")
    assert campaign.command_for(item)[-3:] == ["--resume-training", "--load-checkpoint",
                                               str(Path(item["output"]) / "last_seed1235.pt")]


def scores(a, b, c):
    """Per-seed scores whose dataset means are a, b, c on CESM2 and on real data."""
    return {name: {"cesm2": [m - .001, m, m + .001], "real": [m - .002, m, m + .002]}
            for name, m in (("A", a), ("B", b), ("C", c))}


def test_the_lowest_overall_score_wins_when_it_clears_the_tie_margin():
    outcome = selection.decide(scores(.80, .796, .788))
    assert outcome["chosen"] == "C" and outcome["best"] == "C"
    assert outcome["overall"]["C"] == pytest.approx(.788 / .80)


def test_within_half_a_percent_the_simplest_candidate_is_chosen():
    assert selection.decide(scores(.80, .794, .792))["chosen"] == "B"
    assert selection.decide(scores(.80, .797, .7975))["chosen"] == "A"


def test_the_two_datasets_weigh_equally_by_their_ratio_to_a():
    mixed = {"A": {"cesm2": [.8] * 3, "real": [.9] * 3}, "B": {"cesm2": [.8] * 3, "real": [.9] * 3},
             "C": {"cesm2": [.76] * 3, "real": [.9405] * 3}}
    outcome = selection.decide(mixed)
    # 5 % better on CESM2 but 4.5 % worse on real data: within the margin of A, so A stays
    assert outcome["overall"]["C"] == pytest.approx((.95 + 1.045) / 2) and outcome["chosen"] == "A"


def test_no_decision_while_a_run_is_missing():
    incomplete = scores(.80, .79, .78)
    incomplete["C"]["real"][2] = None
    with pytest.raises(ValueError):
        selection.decide(incomplete)


def summary(folder, seed, *, steps=30000, background="prior", backbone="unet", val_years=(2021,), macro=.85):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"summary_seed{seed}.json").write_text(json.dumps({
        "completed_steps": steps, "seed": seed, "val_years": list(val_years),
        "config": {"kriging": "ka_flow", "background": background, "backbone": backbone, "poseidon_init": "pretrained"},
        "validation": {"scores": {"macro_z": macro}}}))


def test_scores_are_read_only_from_the_candidates_finished_runs(tmp_path):
    folder = tmp_path / "rb_prior_s1234"
    assert selection.score(folder, 1234, 30000, "prior", "unet", "real") is None
    summary(folder, 1234)
    assert selection.score(folder, 1234, 30000, "prior", "unet", "real") == pytest.approx(.85)
    with pytest.raises(ValueError):
        selection.score(folder, 1234, 30000, "joint", "unet", "real")
    summary(tmp_path / "short_s1234", 1234, steps=20000)
    with pytest.raises(ValueError):
        selection.score(tmp_path / "short_s1234", 1234, 30000, "prior", "unet", "real")
