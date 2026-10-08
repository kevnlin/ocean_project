"""Historical learned-OI checkpoint and protocol eligibility guards."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "experiments/synthetic/55_replay_learned_oi.py"
SPEC = importlib.util.spec_from_file_location("prior_learned_oi_replay", PATH)
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


@pytest.fixture
def historical():
    folder = ROOT / "outputs/audit/synthetic/anc_kriging"
    if not folder.exists():
        pytest.skip("historical checkpoint unavailable")
    summary = json.loads((folder / "summary_seed1234.json").read_text())
    state = torch.load(folder / "model_seed1234.pt", map_location="cpu", weights_only=True)
    # The registered cohort depths are sufficient for checkpoint/band audits.
    depths = np.array([5., 15., 25., 35., 45., 55., 65., 85., 105., 125.,
                       145., 165.1, 186.3, 222.6, 267.7, 326.9, 408.8, 527.7, 707.6, 984.7])
    return summary, state, depths


def test_existing_checkpoint_is_per_depth_and_validation_selected(historical):
    summary, state, depths = historical
    audit = R.audit_historical(summary, state, depths)
    assert audit["selected_step"] == 1500
    assert audit["recorded_validation_mean_standardized_rmse"] == pytest.approx(.31586764953397495)
    assert sum(value.numel() for value in state.values()) == 120
    assert len(torch.unique(state["log_ell_x"][:20])) > 4
    assert "not saved" in audit["training_query_budget"]["actual_original_cli_queries"]


@pytest.mark.parametrize("key,value", [("anomaly", "cell"), ("input_qc_z", 10.),
                                       ("use_time", True), ("k", 16), ("steps", 6000)])
def test_mismatched_protocol_is_not_an_eligible_matched_baseline(historical, key, value):
    summary, state, depths = historical
    changed = copy.deepcopy(summary)
    changed[key] = value
    with pytest.raises(ValueError, match="setting differs"):
        R.audit_historical(changed, state, depths)


def test_wrong_checkpoint_and_wrong_selected_step_are_refused(historical):
    summary, state, depths = historical
    changed_state = {key: value.clone() for key, value in state.items()}
    changed_state["log_gamma"][0] += .2
    with pytest.raises(ValueError, match="scales"):
        R.audit_historical(summary, changed_state, depths)
    changed = copy.deepcopy(summary)
    changed["best_step"] = 500
    with pytest.raises(ValueError, match="validation minimum"):
        R.audit_historical(changed, state, depths)


def test_historical_replay_guard_requires_count_and_physical_error(historical):
    summary, _, _ = historical
    original = summary["scores"]["development"]
    R.compare_historical(original, original, "development")
    changed = copy.deepcopy(original)
    changed["TEMP"]["n"] -= 1
    with pytest.raises(ValueError, match="count"):
        R.compare_historical(changed, original, "development")
    changed = copy.deepcopy(original)
    changed["SALT"]["rmse_physical"] *= 1.01
    with pytest.raises(ValueError, match="does not replay"):
        R.compare_historical(changed, original, "development")
