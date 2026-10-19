"""Sealing of the CESM2 2006-2007 test: register once, verify every file, score once."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sealed", ROOT / "experiments/synthetic/109_sealed_cesm2_test.py")
sealed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sealed)


def files(tmp_path):
    a, b, data = tmp_path / "a.pt", tmp_path / "b.pt", tmp_path / "cohort.nc"
    a.write_bytes(b"model a")
    b.write_bytes(b"model b")
    data.write_bytes(b"cohort")
    return a, b, data


def test_a_registration_records_every_file_and_cannot_be_written_twice(tmp_path):
    a, b, data = files(tmp_path)
    path = tmp_path / "registration.json"
    reg = sealed.register(path, [("grid", a), ("reference", b)], {"cohort": data})
    assert [e["family"] for e in reg["entries"]] == ["grid", "reference"]
    assert reg["entries"][0]["sha256"] == sealed.sha256(a) and reg["data"]["cohort"]["sha256"] == sealed.sha256(data)
    assert json.loads(path.read_text()) == reg
    with pytest.raises(SystemExit):
        sealed.register(path, [("grid", a)], {"cohort": data})


def test_scoring_refuses_a_changed_file(tmp_path):
    a, b, data = files(tmp_path)
    reg = sealed.register(tmp_path / "registration.json", [("grid", a), ("reference", b)], {"cohort": data})
    sealed.verify(reg)
    b.write_bytes(b"model b, retrained")
    with pytest.raises(SystemExit):
        sealed.verify(reg)


def test_an_unknown_family_is_refused(tmp_path):
    a, _, data = files(tmp_path)
    with pytest.raises(SystemExit):
        sealed.register(tmp_path / "registration.json", [("transformer", a)], {"cohort": data})
