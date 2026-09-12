"""P7 — the prospective real-observation test, frozen before it is opened.

The plan's P7 is the only package whose validity depends on the ORDER things
happened in, not just on what was computed:

    Before opening: freeze git SHA, checkpoints, manifests, preprocessing,
    baselines, primary endpoint, cluster units.  Once opened, no retuning.

An assertion that this was done is worth nothing after the fact.  So the script
has two stages and the second refuses to run unless the first left a record that
still matches:

    --stage freeze   writes freeze_record.json: git SHA, dirty flag, SHA-256 of
                     every checkpoint, SHA-256 of every data manifest, the
                     primary endpoint and cluster unit, and a timestamp.  Reads
                     NO holdout number.
    --stage open     re-derives all of it, refuses if anything moved, then
                     evaluates the holdout ONCE and writes the artifact.

If a checkpoint hash differs, the run stops.  That is the point: a checkpoint
that changed between freezing and opening means the thing being tested is not
the thing that was registered, and the honest response is to fail loudly rather
than to report a number with a caveat.

The primary endpoint, registered here rather than chosen later
--------------------------------------------------------------
    quantity        pooled RMSE on held-out-float measurements, z units
    channel         TEMP is primary; SALT reported alongside, never pooled
    lead            0
    cluster unit    WMO/platform (primary), source_month (also reported)
    split           the holdout year, used exactly once

  .venv/bin/python experiments/real_data/50_prospective_test.py --stage freeze
  .venv/bin/python experiments/real_data/50_prospective_test.py --stage open
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")

import numpy as np

from ocean_tokenizer import protocol as P

#: rows the open will score. Only learned rows need checkpoints; the
#: non-learned references (climatology, persistence, OI, EN4) are deterministic.
SCORED_ROWS = tuple(r for r in P.REGISTERED_ROWS if r != "objective_interpolation")

PRIMARY_ENDPOINT = {
    "quantity": "pooled RMSE on held-out-float measurements (train-only z units)",
    "primary_channel": "TEMP",
    "secondary_channel": "SALT",
    "lead": 0,
    "primary_cluster_unit": "wmo",
    "secondary_cluster_unit": "source_month",
    "split": "holdout",
    "uses": "exactly once, after freezing",
}

ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True, choices=["freeze", "open", "correct"])
ap.add_argument("--region", default="gulfstream", choices=list(P.REGIONS))
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--checkpoint-dir", default=None)
ap.add_argument("--output", default=None)
ap.add_argument("--force-open", action="store_true",
                help="open despite a freeze mismatch. Records the violation in "
                     "the artifact; never use this for a reported number.")
args = ap.parse_args()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CKDIR = args.checkpoint_dir or os.path.join(ROOT, "outputs",
                                            f"argo_P0_{args.region}")
OUT = args.output or os.path.join(ROOT, "outputs", f"argo_P7_{args.region}")
os.makedirs(OUT, exist_ok=True)
FREEZE = os.path.join(OUT, f"freeze_record_seed{args.seed}.json")


def checkpoint_hashes() -> dict:
    out = {}
    for p in sorted(glob.glob(os.path.join(CKDIR, f"*_s{args.seed}.pt"))):
        out[os.path.basename(p)] = P.sha256_file(p)
    return out


def current_state() -> dict:
    return {
        "git_commit": P.git_commit(),
        "git_dirty": P.git_dirty(),
        "protocol_hash": P.protocol_hash(),
        "protocol_version": P.PROTOCOL_VERSION,
        "data_manifests": P.data_manifest_hashes(ROOT),
        "checkpoints": checkpoint_hashes(),
        "primary_endpoint": PRIMARY_ENDPOINT,
        "registered_rows": list(P.REGISTERED_ROWS),
        "reference_rows": list(P.REFERENCE_ROWS),
        "region": args.region,
        "seed": args.seed,
        # Part of the registration, and worth comparing: silently changing WHICH
        # rows get scored between freeze and open would let the reported set be
        # chosen after the fact. It belongs in the state, not only in the record
        # -- leaving it out of current_state() made the diff report
        # "scored_rows: only before" and refuse every open.
        "scored_rows": list(SCORED_ROWS),
    }


def diff(a: dict, b: dict, prefix: str = "") -> list[str]:
    out = []
    for k in sorted(set(a) | set(b)):
        key = f"{prefix}.{k}" if prefix else k
        if k not in a or k not in b:
            out.append(f"{key}: {'only after' if k not in a else 'only before'}")
        elif isinstance(a[k], dict) and isinstance(b[k], dict):
            out += diff(a[k], b[k], key)
        elif a[k] != b[k]:
            out.append(f"{key}: {a[k]!r} -> {b[k]!r}")
    return out


if args.stage == "freeze":
    st = current_state()
    if not st["checkpoints"]:
        raise SystemExit(
            f"no checkpoints in {CKDIR} for seed {args.seed}. Freeze after the "
            f"P0 run has written them — freezing an empty set would let any "
            f"checkpoint be substituted later without tripping the check.")
    # Every row that will be SCORED must already be frozen. The first P7 run
    # froze the 5 rows that had checkpoints, then trained the 3 `oi_expert`
    # rows from scratch during the open -- 4,000 steps each, inside the
    # one-shot holdout. Checking only that SOME checkpoint exists is not
    # enough; the freeze has to cover the whole scored set.
    missing = [r for r in SCORED_ROWS
               if f"{r}_s{args.seed}.pt" not in st["checkpoints"]]
    if missing:
        raise SystemExit(
            "refusing to freeze: no checkpoint for " + ", ".join(missing) +
            f"\n  in {CKDIR} (seed {args.seed}).\n"
            "  Those rows would be TRAINED during the open, which is exactly "
            "the retuning P7 forbids.\n"
            "  Either train them first, or restrict --rows to the frozen set.")

    st["frozen_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    st["holdout_read"] = False
    with open(FREEZE + ".tmp", "w") as f:
        json.dump(st, f, indent=1)
    os.replace(FREEZE + ".tmp", FREEZE)
    print(f"frozen at {st['frozen_utc']}")
    print(f"  git {st['git_commit'][:10]} dirty={st['git_dirty']}")
    print(f"  {len(st['checkpoints'])} checkpoints")
    for k, v in st["checkpoints"].items():
        print(f"    {k}  {v[:16]}")
    print(f"  primary endpoint: {PRIMARY_ENDPOINT['quantity']}, "
          f"clustered by {PRIMARY_ENDPOINT['primary_cluster_unit']}")
    if st["git_dirty"]:
        print("\n  WARNING: working tree is dirty, so the recorded commit does "
              "not fully identify the frozen code.")
    print(f"\nrecord: {FREEZE}\n  sha256 {P.sha256_file(FREEZE)}")
    print("\nHoldout has NOT been read. Run --stage open to open it once.")
    raise SystemExit(0)

# ---------------------------------------------------------------- correct
if args.stage == "correct":
    """Re-score an ALREADY-OPENED holdout from the checkpoints the freeze pinned.

    This is not a second open, and the distinction matters. An open is a
    registration event: it converts a frozen claim into a number, and the plan
    allows one because a second look lets the result be selected. A correction
    changes nothing about what was registered -- same freeze record, same
    checkpoints, same cohort, same endpoint -- it repairs a mechanical fault in
    which bytes got loaded.

    The fault being repaired: checkpoint lookup resolved by FILENAME, and this
    script's own output directory (which still holds the models an abandoned
    earlier open trained under those names) precedes the checkpoint directory on
    that search path. Three rows of the reported table were produced by models
    the freeze record does not pin.

    This refuses to run if the previous open in fact loaded what was frozen --
    there is then nothing to correct, and re-scoring would be a second look
    wearing a correction's clothes.
    """
    if not os.path.exists(FREEZE):
        raise SystemExit(f"no freeze record at {FREEZE}.")
    frozen = json.load(open(FREEZE))
    if not frozen.get("holdout_read"):
        raise SystemExit(
            "this freeze record has not been opened. Use --stage open; a "
            "correction repairs an open that happened, it does not stand in "
            "for one.")
    art_path = os.path.join(OUT, f"artifact_P0_seed{args.seed}.json")
    if not os.path.exists(art_path):
        raise SystemExit(f"no artifact at {art_path} to correct.")
    training = json.load(open(art_path)).get("results", {}).get("training", {})
    pins = frozen.get("checkpoints", {})
    wrong = {r: (m.get("checkpoint_sha256"), m.get("loaded_from"))
             for r, m in training.items()
             if pins.get(f"{r}_s{args.seed}.pt")
             and m.get("checkpoint_sha256") != pins[f"{r}_s{args.seed}.pt"]}
    if not wrong:
        raise SystemExit(
            "the recorded open loaded exactly the checkpoints the freeze "
            "pinned. There is nothing to correct, and re-scoring the holdout "
            "would be a second look at a spent cohort.")
    print(f"correcting {len(wrong)} of {len(training)} rows that were scored "
          f"from unfrozen checkpoints:")
    for r, (got, src) in sorted(wrong.items()):
        print(f"  {r}: scored {str(got)[:16]} from {src}, "
              f"frozen {pins[f'{r}_s{args.seed}.pt'][:16]}")
    cmd = [os.path.join(ROOT, ".venv", "bin", "python"),
           os.path.join(ROOT, "experiments", "45_argo_real_data.py"),
           "--package", "P0", "--region", args.region, "--seed", str(args.seed),
           "--eval-split", "holdout", "--leads", "0",
           "--rows", ",".join(frozen.get("scored_rows", SCORED_ROWS)),
           "--require-checkpoints",
           "--checkpoint-dir", CKDIR,
           "--freeze-record", FREEZE,
           "--output", OUT]
    print("\n  " + " ".join(cmd), flush=True)
    import subprocess
    r = subprocess.run(cmd, text=True)
    if r.returncode != 0:
        raise SystemExit(f"scoring failed (rc={r.returncode}); freeze record "
                         f"left untouched.")
    frozen.setdefault("corrections", []).append({
        "corrected_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reason": "the recorded open resolved checkpoints by filename and "
                  "loaded models the freeze record does not pin",
        "rows_corrected": sorted(wrong),
        "scored_sha256_before": {k: v[0] for k, v in sorted(wrong.items())},
        "frozen_sha256": {k: pins[f"{k}_s{args.seed}.pt"]
                          for k in sorted(wrong)},
        "note": "not a second open: same freeze record, checkpoints, cohort and "
                "endpoint. Nothing was retrained or retuned between the open "
                "and this correction.",
    })
    with open(FREEZE + ".tmp", "w") as f:
        json.dump(frozen, f, indent=1)
    os.replace(FREEZE + ".tmp", FREEZE)
    print(f"\ncorrection recorded in {FREEZE}")
    raise SystemExit(0)

# ------------------------------------------------------------------- open
if not os.path.exists(FREEZE):
    raise SystemExit(
        f"no freeze record at {FREEZE}.\nP7 is only meaningful if the freeze "
        f"happened BEFORE the holdout was read. Run --stage freeze first.")
frozen = json.load(open(FREEZE))
if frozen.get("holdout_read"):
    raise SystemExit(
        "this freeze record has already been opened. The plan allows the "
        "holdout exactly once; opening it again would make the reported number "
        "a selected one. Freeze a new record against new checkpoints instead.")

now = current_state()
mismatches = diff({k: v for k, v in frozen.items()
                   if k not in ("frozen_utc", "holdout_read")}, now)
if mismatches:
    print("FREEZE MISMATCH — the state has moved since freezing:")
    for m in mismatches:
        print(f"  {m}")
    if not args.force_open:
        raise SystemExit(
            "\nRefusing to open. What is about to be tested is not what was "
            "registered. Re-freeze deliberately, or pass --force-open to record "
            "the violation in the artifact (never for a reported number).")
    print("\n--force-open given: proceeding and recording the violation.\n")

print(f"opening the holdout for {args.region} seed {args.seed} — once.",
      flush=True)
# `--checkpoint-dir` and `--freeze-record` are both load-bearing.
#
# Without them the open resolved checkpoints by directory order, and its output
# directory -- which holds whatever an abandoned earlier open trained under the
# same filenames -- comes FIRST on that search path.  Three rows of the first
# reported holdout table were scored from models the freeze had never seen, and
# the freeze's own mismatch check could not see it, because that check
# re-derives `checkpoint_hashes()` over CKDIR on both sides and CKDIR never
# changed.  Naming the directory removes the ambiguity; passing the record makes
# the open verify every load against the pin, so a future ambiguity aborts
# instead of reporting a number.
cmd = [os.path.join(ROOT, ".venv", "bin", "python"),
       os.path.join(ROOT, "experiments", "45_argo_real_data.py"),
       "--package", "P0", "--region", args.region, "--seed", str(args.seed),
       "--eval-split", "holdout", "--leads", "0",
       "--rows", ",".join(frozen.get("scored_rows", SCORED_ROWS)),
       "--require-checkpoints",
       "--checkpoint-dir", CKDIR,
       "--freeze-record", FREEZE,
       "--output", OUT]
print("  " + " ".join(cmd), flush=True)
import subprocess
r = subprocess.run(cmd, text=True)

frozen["holdout_read"] = True
frozen["opened_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
frozen["freeze_mismatches"] = mismatches
frozen["forced"] = bool(args.force_open and mismatches)
with open(FREEZE + ".tmp", "w") as f:
    json.dump(frozen, f, indent=1)
os.replace(FREEZE + ".tmp", FREEZE)
print(f"\nfreeze record marked opened. rc={r.returncode}")
