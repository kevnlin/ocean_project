"""The P7 freeze guarantee: the model scored is the model registered.

These tests exist because that guarantee failed silently in a reported run.
`outputs/argo_P7_<region>/` held models an abandoned earlier open had trained
under exactly the filenames a later freeze pinned in a different directory.
Checkpoint lookup took the first directory containing the name, so three rows of
the holdout table were scored from models the freeze record did not pin -- and
the freeze's own before/after comparison could not see it, because that
comparison re-derives the hashes over one directory on both sides.

The fix is to resolve by bytes rather than by name, so the tests are about
exactly that: a decoy under the right name must never win.
"""
from __future__ import annotations

import os

import pytest

from ocean_tokenizer.protocol import resolve_pinned_checkpoint, sha256_file


def _write(path: str, payload: bytes) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(payload)
    return sha256_file(path)


def test_pin_beats_directory_order(tmp_path):
    """The decoy comes first on the search path and must still lose."""
    first, second = str(tmp_path / "p7"), str(tmp_path / "p0")
    _write(os.path.join(first, "row_s1234.pt"), b"abandoned first attempt")
    pin = _write(os.path.join(second, "row_s1234.pt"), b"the frozen model")

    path, decoys = resolve_pinned_checkpoint("row_s1234.pt", [first, second], pin)

    assert path == os.path.join(second, "row_s1234.pt")
    assert sha256_file(path) == pin
    assert decoys == [os.path.join(first, "row_s1234.pt")]


def test_no_copy_matches_the_pin_resolves_to_nothing(tmp_path):
    """A pin nothing satisfies must fail loudly, never fall back to a decoy.

    Returning the decoy "because it is the only candidate" is precisely the
    behaviour that produced a holdout number from an unregistered model.
    """
    d = str(tmp_path / "ckpt")
    _write(os.path.join(d, "row_s1234.pt"), b"not the frozen bytes")

    path, decoys = resolve_pinned_checkpoint("row_s1234.pt", [d], "0" * 64)

    assert path is None
    assert decoys == [os.path.join(d, "row_s1234.pt")]


def test_unpinned_lookup_keeps_first_match_wins(tmp_path):
    """Without a freeze record the rule is unchanged: P0 and P1 still work."""
    first, second = str(tmp_path / "a"), str(tmp_path / "b")
    _write(os.path.join(first, "row_s1234.pt"), b"one")
    _write(os.path.join(second, "row_s1234.pt"), b"two")

    path, decoys = resolve_pinned_checkpoint("row_s1234.pt", [first, second], None)

    assert path == os.path.join(first, "row_s1234.pt")
    assert decoys == []


def test_absent_checkpoint_reports_nothing_found(tmp_path):
    path, decoys = resolve_pinned_checkpoint("row_s1234.pt", [str(tmp_path)], "x")
    assert path is None and decoys == []


def test_missing_directories_are_skipped_not_raised(tmp_path):
    """A search path may legitimately name a directory that does not exist."""
    real = str(tmp_path / "real")
    pin = _write(os.path.join(real, "row_s1234.pt"), b"frozen")
    path, _ = resolve_pinned_checkpoint(
        "row_s1234.pt", [str(tmp_path / "nope"), None, real], pin)
    assert path == os.path.join(real, "row_s1234.pt")


@pytest.mark.parametrize("pin_is_for", ["first", "second"])
def test_whichever_directory_holds_the_frozen_bytes_wins(tmp_path, pin_is_for):
    """Resolution follows the pin, not a hard-coded directory preference."""
    first, second = str(tmp_path / "one"), str(tmp_path / "two")
    h1 = _write(os.path.join(first, "row_s1234.pt"), b"alpha")
    h2 = _write(os.path.join(second, "row_s1234.pt"), b"beta")
    pin = h1 if pin_is_for == "first" else h2
    want = first if pin_is_for == "first" else second

    path, decoys = resolve_pinned_checkpoint("row_s1234.pt", [first, second], pin)

    assert path == os.path.join(want, "row_s1234.pt")
    assert len(decoys) == 1
