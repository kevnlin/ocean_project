"""Evidence gates for automatic delivery of the final scientific report."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
module_spec = importlib.util.spec_from_file_location(
    "final_report_delivery", ROOT / "experiments/synthetic/59_final_report_delivery.py")
delivery = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(delivery)


def fixture(tmp_path):
    review = tmp_path / "literature.json"
    review.write_text('{"papers": []}')
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{}')
    selection = tmp_path / "selection.json"
    selection.write_text('{}')
    matched = tmp_path / "matched.md"
    matched.write_text('matched report')
    completion = {"phase": "complete", "training_runs": 39, "steps_per_run": 15000,
                  "metrics_sha256": delivery.sha256(metrics),
                  "selection_sha256": delivery.sha256(selection),
                  "report_sha256": delivery.sha256(matched)}
    delivery.atomic_json(tmp_path / "completion.json", completion)
    spec = {"format_version": 1, "output_root": str(tmp_path),
            "metrics_path": str(metrics), "selection_path": str(selection),
            "matched_report_path": str(matched),
            "report_path": str(tmp_path / "final.md"),
            "data_path": str(tmp_path / "final.json"), "command": ["unused-renderer"],
            "sealed_inputs": [{"path": str(review), "sha256": delivery.sha256(review)}]}
    spec_path = tmp_path / "research_report_spec.json"
    delivery.atomic_json(spec_path, spec)
    return spec, spec_path


def write_renderer_output(spec, fingerprints):
    Path(spec['report_path']).write_text('完整研究报告')
    delivery.atomic_json(spec['data_path'], {
        "completion": {"phase": "complete"},
        "test_n_per_variable": {"TEMP": 351895, "SALT": 351895},
        "sources": {key: {"sha256": fingerprints[path]} for key, path in (
            ("metrics", spec['metrics_path']), ("selection", spec['selection_path']),
            ("completion", str(Path(spec['output_root']) / 'completion.json')))}})


def test_missing_completion_never_starts_renderer(tmp_path, monkeypatch):
    spec, spec_path = fixture(tmp_path)
    (tmp_path / 'completion.json').unlink()
    monkeypatch.setattr(delivery.subprocess, 'run', lambda *a, **kw: pytest.fail('renderer started early'))
    assert delivery.main(['--output-root', str(tmp_path), '--spec', str(spec_path), '--once']) == 2
    assert not (tmp_path / 'research_report_completion.json').exists()


@pytest.mark.parametrize('changed', ['literature.json', 'metrics.json', 'selection.json', 'matched.md'])
def test_changed_review_or_completed_evidence_is_rejected(tmp_path, monkeypatch, changed):
    spec, spec_path = fixture(tmp_path)
    (tmp_path / changed).write_text('changed')
    monkeypatch.setattr(delivery.subprocess, 'run', lambda *a, **kw: pytest.fail('renderer started on changed input'))
    with pytest.raises(ValueError, match='changed'):
        delivery.deliver(spec_path, tmp_path)


def test_pending_renderer_cannot_be_delivered(tmp_path, monkeypatch):
    spec, spec_path = fixture(tmp_path)
    def render(*args, **kwargs):
        Path(spec['report_path']).write_text('pending')
        delivery.atomic_json(spec['data_path'], {'completion': {'phase': 'pending'}})
    monkeypatch.setattr(delivery.subprocess, 'run', render)
    with pytest.raises(RuntimeError, match='pending'):
        delivery.deliver(spec_path, tmp_path)
    assert not (tmp_path / 'research_report_completion.json').exists()


def test_success_exit_without_machine_report_is_a_renderer_error(tmp_path, monkeypatch):
    spec, spec_path = fixture(tmp_path)
    monkeypatch.setattr(delivery.subprocess, 'run', lambda *a, **kw: Path(spec['report_path']).write_text('partial'))
    with pytest.raises(RuntimeError, match='machine-readable report'):
        delivery.deliver(spec_path, tmp_path)
    assert not (tmp_path / 'research_report_completion.json').exists()


def test_delivery_verifies_provenance_and_is_idempotent(tmp_path, monkeypatch):
    spec, spec_path = fixture(tmp_path)
    fingerprints = delivery.validate_inputs(spec, tmp_path)
    calls = []
    def render(*args, **kwargs):
        calls.append(args)
        write_renderer_output(spec, fingerprints)
    monkeypatch.setattr(delivery.subprocess, 'run', render)
    first = delivery.deliver(spec_path, tmp_path)
    assert first['phase'] == 'complete'
    assert delivery.deliver(spec_path, tmp_path) == first
    assert len(calls) == 1
    Path(spec['report_path']).write_text('modified after delivery')
    with pytest.raises(RuntimeError, match='no longer matches'):
        delivery.deliver(spec_path, tmp_path)


def test_evidence_cannot_change_during_rendering(tmp_path, monkeypatch):
    spec, spec_path = fixture(tmp_path)
    fingerprints = delivery.validate_inputs(spec, tmp_path)
    def render(*args, **kwargs):
        write_renderer_output(spec, fingerprints)
        Path(spec['metrics_path']).write_text('changed while rendering')
    monkeypatch.setattr(delivery.subprocess, 'run', render)
    with pytest.raises(RuntimeError, match='during rendering'):
        delivery.deliver(spec_path, tmp_path)


def test_wrong_metric_support_and_provenance_fail(tmp_path):
    spec, _ = fixture(tmp_path)
    fingerprints = delivery.validate_inputs(spec, tmp_path)
    write_renderer_output(spec, fingerprints)
    data = delivery.load_json(spec['data_path'])
    data['test_n_per_variable']['SALT'] = 1
    delivery.atomic_json(spec['data_path'], data)
    with pytest.raises(RuntimeError, match='every scored value'):
        delivery.validate_report(spec, fingerprints)
    data['test_n_per_variable']['SALT'] = 351895
    data['sources']['selection']['sha256'] = 'incorrect'
    delivery.atomic_json(spec['data_path'], data)
    with pytest.raises(RuntimeError, match='selection provenance'):
        delivery.validate_report(spec, fingerprints)


def test_failed_experiment_blocks_delivery(tmp_path):
    _, spec_path = fixture(tmp_path)
    delivery.atomic_json(tmp_path / 'finalization_status.json', {'phase': 'failed'})
    with pytest.raises(RuntimeError, match='completion guard failed'):
        delivery.main(['--output-root', str(tmp_path), '--spec', str(spec_path), '--once'])
    assert delivery.load_json(tmp_path / 'research_report_status.json')['phase'] == 'failed'


def test_delivery_spec_cannot_change_during_render(tmp_path, monkeypatch):
    spec, spec_path = fixture(tmp_path)
    fingerprints = delivery.validate_inputs(spec, tmp_path)
    def render(*args, **kwargs):
        write_renderer_output(spec, fingerprints)
        spec['command'] = ['changed-renderer']
        delivery.atomic_json(spec_path, spec)
    monkeypatch.setattr(delivery.subprocess, 'run', render)
    with pytest.raises(RuntimeError, match='specification changed during rendering'):
        delivery.deliver(spec_path, tmp_path)
    assert not (tmp_path / 'research_report_completion.json').exists()
