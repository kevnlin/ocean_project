"""Deliver the comprehensive report after the verified matched experiment.

This independent watcher does not start training, select models or open model
prediction arrays. Its sealed specification fixes the report renderer and the
reviewed literature; the renderer checks the complete scientific evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/synthetic_matched_20261007"


class PendingDelivery(RuntimeError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_json(path):
    path = Path(path)
    if not path.is_file():
        raise PendingDelivery(f"Waiting for {path}")
    return json.loads(path.read_text())


def validate_inputs(spec, output):
    """Only verified completion permits comprehensive report generation."""
    if spec.get("format_version") != 1 or not spec.get("sealed_inputs"):
        raise ValueError("A sealed version-1 report specification is required")
    for item in spec["sealed_inputs"]:
        path = Path(item["path"])
        if not path.is_file():
            raise PendingDelivery(f"Waiting for reviewed input: {path}")
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Sealed report input changed: {path}")
    completion_path = Path(output) / "completion.json"
    completion = load_json(completion_path)
    if completion.get("phase") != "complete" or completion.get("training_runs") != 39:
        raise PendingDelivery("The 39-run campaign is not verified complete")
    if completion.get("steps_per_run") != 15000:
        raise ValueError("The verified training budget differs")
    fingerprints = {str(completion_path): sha256(completion_path)}
    for key, field in (("metrics_sha256", "metrics_path"),
                       ("selection_sha256", "selection_path"),
                       ("report_sha256", "matched_report_path")):
        path = Path(spec[field])
        if not path.is_file():
            raise PendingDelivery(f"Completed artifact is missing: {path}")
        fingerprint = sha256(path)
        if fingerprint != completion.get(key):
            raise ValueError(f"Completed evidence changed: {path}")
        fingerprints[str(path)] = fingerprint
    for item in spec["sealed_inputs"]:
        fingerprints[item["path"]] = item["sha256"]
    return fingerprints


def validate_report(spec, fingerprints):
    report, data_path = Path(spec["report_path"]), Path(spec["data_path"])
    if not report.is_file() or not report.read_text().strip():
        raise RuntimeError("Renderer did not produce the comprehensive report")
    if not data_path.is_file():
        raise RuntimeError("Renderer did not produce the machine-readable report")
    data = load_json(data_path)
    if data.get("completion", {}).get("phase") != "complete":
        raise RuntimeError("Renderer produced a pending or unverified report")
    if data.get("test_n_per_variable") != {"TEMP": 351895, "SALT": 351895}:
        raise RuntimeError("Comprehensive report does not cover every scored value")
    sources = data.get("sources", {})
    for key, path in (("metrics", spec["metrics_path"]),
                      ("selection", spec["selection_path"]),
                      ("completion", str(Path(spec["output_root"]) / "completion.json"))):
        if sources.get(key, {}).get("sha256") != fingerprints[path]:
            raise RuntimeError(f"Comprehensive report lost verified {key} provenance")
    for item in spec["sealed_inputs"]:
        if "source_key" in item and sources.get(item["source_key"], {}).get("sha256") != item["sha256"]:
            raise RuntimeError(f"Comprehensive report lost sealed {item['source_key']} provenance")
    for path, expected in fingerprints.items():
        if sha256(path) != expected:
            raise RuntimeError(f"Report evidence changed during rendering: {path}")
    return {"report_path": str(report), "report_sha256": sha256(report),
            "data_path": str(data_path), "data_sha256": sha256(data_path)}


def deliver(spec_path, output):
    output = Path(output)
    spec_fingerprint = sha256(spec_path) if Path(spec_path).is_file() else None
    spec = load_json(spec_path)
    if Path(spec["output_root"]).resolve() != output.resolve():
        raise ValueError("Delivery specification belongs to a different campaign")
    fingerprints = validate_inputs(spec, output)
    delivered_path = output / "research_report_completion.json"
    if delivered_path.exists():
        delivered = load_json(delivered_path)
        artifacts = validate_report(spec, fingerprints)
        if delivered.get("spec_sha256") != spec_fingerprint or delivered.get("evidence_sha256") != fingerprints or any(
                delivered.get(key) != value for key, value in artifacts.items()):
            raise RuntimeError("An existing delivery no longer matches the verified evidence")
        return delivered
    command = spec.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(s, str) for s in command):
        raise ValueError("Report renderer must be an explicit argument list")
    subprocess.run(command, cwd=ROOT, check=True)
    artifacts = validate_report(spec, fingerprints)
    if sha256(spec_path) != spec_fingerprint:
        raise RuntimeError("The sealed delivery specification changed during rendering")
    delivered = {"phase": "complete", "training_runs": 39, "steps_per_run": 15000,
                 **artifacts, "spec_sha256": spec_fingerprint,
                 "evidence_sha256": fingerprints, "verified_unix": time.time()}
    atomic_json(delivered_path, delivered)
    return delivered


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--spec", type=Path, default=OUTPUT / "research_report_spec.json")
    parser.add_argument("--poll-seconds", type=float, default=20.)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if args.poll_seconds < 1:
        parser.error("Polling interval must be at least one second")
    status_path = args.output_root / "research_report_status.json"
    try:
        while True:
            finalization = args.output_root / "finalization_status.json"
            if finalization.exists() and load_json(finalization).get("phase") == "failed":
                raise RuntimeError("The experiment completion guard failed; report delivery is blocked")
            try:
                delivered = deliver(args.spec, args.output_root)
            except PendingDelivery as error:
                atomic_json(status_path, {"phase": "waiting_for_verified_experiment",
                            "reason": str(error), "updated_unix": time.time()})
                if args.once:
                    print(str(error), flush=True)
                    return 2
                time.sleep(args.poll_seconds)
                continue
            atomic_json(status_path, {"phase": "complete", "report_path": delivered["report_path"],
                                    "verified_unix": time.time()})
            print(f"Comprehensive research report verified: {delivered['report_path']}", flush=True)
            return 0
    except Exception as error:
        atomic_json(status_path, {"phase": "failed", "failure": repr(error), "updated_unix": time.time()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
