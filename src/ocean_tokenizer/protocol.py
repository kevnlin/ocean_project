"""The frozen cross-check specification, and the artifact both tracks sign.

`parallel_replication_crosscheck_plan.md` S2 lists what the two tracks MUST
share, and S6 lists the quantities that must match **exactly** — a mismatch
there "is a protocol problem", not a numerical disagreement.  Both lists are
prose in the plan.  This module makes them a data structure, because a shared
specification that lives only in a document is one that two tracks can silently
diverge from and discover months later.

The design in one line: **a track never asserts what it ran; it emits a
`ResultArtifact` whose hashes are computed from the files it actually read.**

Why hashes of manifests rather than of the data
-----------------------------------------------
Re-hashing 9.7 GB of Argo per run is wasteful, and hashing a Zarr directory is
ill-defined (chunk layout is an implementation detail).  Every ingest here
already writes a manifest containing a SHA-256 per file, so hashing the
*manifest* is a hash of the whole dataset that costs microseconds and changes
if and only if some file changed.  `data/argo/manifest.json` additionally
carries the GDAC index hash, which pins the float *selection* as well as the
float *contents*.

What is deliberately NOT in the protocol hash
---------------------------------------------
Seeds, device, worker counts, and output paths.  Two tracks are meant to differ
in those; forcing them into the hash would make every legitimate comparison
report a protocol violation.  Seeds are recorded in the artifact and compared
separately (S5 lists "seed list" among the things to compare, not among the
exact-match quantities).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field, asdict
from typing import Any

PROTOCOL_VERSION = "crosscheck_v1"

# --------------------------------------------------------------------------
# S2 — the frozen specification the two tracks share
# --------------------------------------------------------------------------
#: Year splits.  Identical to the GODAS driver's SPLITS and to the Argo cohort
#: builder's YEAR_SPLITS; asserted equal by test rather than kept in step by
#: hand, because three copies of a split definition is three chances to drift.
YEAR_SPLITS = {"train": (2000, 2018), "validation": (2019, 2021),
               "development": (2022, 2024), "holdout": (2025, 2025)}

#: A SECOND, clearly-labelled split table whose evaluation era sits inside
#: ECCO V4r4's coverage (which ends 2018-01-01).
#:
#: This exists because no later ECCO Central Estimate is obtainable.  The
#: official ECCO products page lists V4r4 (1992-2017) as the latest Central
#: Estimate -- V4r3 reaches 2015, V4r2/r1 reach 2011 -- and NASA CMR carries no
#: V4r5 at all.  So "download more ECCO" does not fix the gap; the only ECCO
#: product reaching 2023 is ECCO2 Cube92, a different product family on a
#: different grid, which must not be reported as continuous with V4r4.
#:
#: What makes shifting the splits legitimate rather than a convenience: the
#: held-out float cohort is **WMO-disjoint and year-independent**.  A float is
#: held out for its whole life, so moving the evaluation era does not move which
#: floats are held out.  Under this table the model still never sees the test
#: months and never sees the held-out floats, so the comparison stays
#: out-of-sample in both, and ECCO can be scored on the identical queries.
#:
#: It is a SECONDARY protocol and is never mixed into the main headline table --
#: the same treatment this project already gives the extended 23-level grid.
ECCO_OVERLAP_SPLITS = {"train": (2000, 2012), "validation": (2013, 2014),
                       "development": (2015, 2017), "holdout": (2017, 2017)}

#: "Most recent three years" protocol. About 80 % of the observations train the
#: model (2000-2021 holds 84 % of Gulf Stream and 88 % of N. Pacific gyre
#: profiles in 2000-2024), and the three most recent open years validate and
#: test it: 2022 selects checkpoints, 2023-2024 is scored. 2025 stays the sealed
#: P7 prospective holdout and is never opened here.
#:
#: The ranges do NOT overlap. `ArgoCohort.apply_splits` is last-wins, so an
#: overlapping table silently moves the shared year into the later label.
RECENT3_SPLITS = {"train": (2000, 2021), "validation": (2022, 2022),
                  "development": (2023, 2024), "holdout": (2025, 2025)}

#: Simulation pretraining. The CESM2-LE store holds 72 months (2000-2005), so the
#: simulation cohort is sampled at the real Argo positions of a dense 72-month
#: window (2016-2021) mapped month-for-month onto those simulation months. These
#: years label the SIMULATED ocean, not the real one; nothing here is a real
#: observation, so it cannot leak into the recent-three-years evaluation.
SIM_PRETRAIN_SPLITS = {"train": (2016, 2019), "validation": (2020, 2020),
                       "development": (2021, 2021)}

SPLIT_PROTOCOLS = {"main": YEAR_SPLITS, "ecco_overlap": ECCO_OVERLAP_SPLITS,
                   "recent3": RECENT3_SPLITS, "sim_pretrain": SIM_PRETRAIN_SPLITS}

#: P1 regions.  Both boxes span 25 deg latitude and 51 deg longitude so the
#: GODAS subsets land on an identical 38 x 26 grid — the model, token budget and
#: patch tiling are then literally the same object across regions, and a region
#: effect cannot be a grid-shape effect in disguise.
REGIONS = {
    "gulfstream": dict(lat=(25.0, 50.0), lon=(280.0, 331.0),
                       character="western boundary current, high EKE"),
    "npac_gyre": dict(lat=(20.0, 45.0), lon=(180.0, 231.0),
                      character="North Pacific subtropical gyre, quiet interior"),
    "eq_pacific": dict(lat=(-12.5, 12.5), lon=(180.0, 231.0),
                       character="central equatorial Pacific, cold tongue and "
                                 "shallow thermocline — the hardest water in "
                                 "the global reconstruction"),
}

CHANNELS = ("TEMP", "SALT")
GRID = (38, 26)
CONTEXT_MONTHS = 2
MAX_LEAD = 3

#: S1 "same cluster units".  WMO is primary per P7; source-month is the
#: secondary that must also be reported.  Naming them here stops one track
#: reporting a month-clustered CI against the other's float-clustered one and
#: calling the difference a result.
CLUSTER_UNITS = ("wmo", "source_month")
PRIMARY_CLUSTER_UNIT = "wmo"

#: S1 "same baseline definitions" — the registered rows, in ranking order for
#: the P0 table.  `objective_interpolation` is the operational reference;
#: thin/superob are the count-independent preprocessing ladder of P3.
REGISTERED_ROWS = (
    "objective_interpolation",
    "count_expertlocal_cbottle",
    "uniform_expertlocal_cbottle",
    "dfs_expertlocal_cbottle",
    "thin_expertlocal_cbottle",
    "superob_expertlocal_cbottle",
    "count_oi_expert_cbottle",
    "uniform_oi_expert_cbottle",
    "dfs_oi_expert_cbottle",
)

#: Non-learned reference rows that need no training, only an adapter.
REFERENCE_ROWS = ("train_climatology", "source_persistence", "en4", "ecco")

HEADLINE_SEEDS = (1234, 1235, 1236)

#: S6 numerical stopping conditions, named so a report cannot quietly omit one.
STOPPING_CONDITIONS = (
    "sign_of_dfs_minus_uniform",
    "ci_excludes_zero",
    "method_ranking",
    "layout_did_sign",
    "redundancy_positive_control",
)


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def git_commit() -> str:
    return _git("rev-parse", "HEAD")


def git_dirty() -> bool:
    """True if the working tree differs from HEAD.

    A dirty tree means the commit hash does not identify the code that ran, so
    the artifact records it rather than letting a git SHA imply more than it can.
    """
    return _git("status", "--porcelain") not in ("", "unknown")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(obj: Any) -> str:
    """Order-independent hash of a JSON-able object.

    ``sort_keys`` matters: two tracks building the same dict in a different
    insertion order must hash the same, or every comparison fails on nothing.
    """
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"),
                   default=str).encode()).hexdigest()


def protocol_hash() -> str:
    """One hash over everything S2 says the tracks must share."""
    return canonical_hash({
        "version": PROTOCOL_VERSION, "year_splits": YEAR_SPLITS,
        "regions": REGIONS, "channels": CHANNELS, "grid": GRID,
        "context_months": CONTEXT_MONTHS, "max_lead": MAX_LEAD,
        "cluster_units": CLUSTER_UNITS,
        "primary_cluster_unit": PRIMARY_CLUSTER_UNIT,
        "registered_rows": REGISTERED_ROWS, "reference_rows": REFERENCE_ROWS,
        "headline_seeds": HEADLINE_SEEDS,
    })


def collection_manifest_hashes(root: str) -> dict[str, str]:
    """SHA-256 of each ingest manifest present, keyed by dataset.

    These describe whole COLLECTIONS and grow whenever any dataset is extended.
    They are recorded for provenance and are deliberately **not** exact-match
    quantities -- see `data_manifest_hashes` for why.
    """
    out = {}
    for name, rel in (("godas_gulfstream", "data/godas_gulfstream/manifest.json"),
                      ("godas_npac_gyre", "data/godas_npac_gyre/manifest.json"),
                      ("argo", "data/argo/manifest.json"),
                      ("argo_cohort", "data/argo_cohort/manifest.json"),
                      ("reference_products", "data/reference/manifest.json")):
        p = os.path.join(root, rel)
        out[name] = sha256_file(p) if os.path.exists(p) else "absent"
    return out


def data_manifest_hashes(root: str, region: str | None = None) -> dict[str, str]:
    """SHA-256 of the data a run actually CONSUMED.

    Hashing the whole-collection Argo manifest was a design error, and it bit:
    downloading equatorial and then global floats rewrote that manifest three
    times, so Gulf Stream artifacts written weeks apart recorded three different
    hashes for data none of them had touched. Since `data_manifests` is an
    exact-match quantity under the plan's S6, a cross-check between two
    perfectly valid Gulf Stream runs would have reported a protocol violation
    caused by an unrelated download.

    A run consumes ONE region's cohort file and (optionally) the reference
    products. Those are what is hashed here. Collection-level manifests are
    still recorded, in `collection_manifests`, as provenance.

    Absent inputs are recorded as ``"absent"`` rather than omitted: a track that
    never downloaded EN4 and a track that did must produce visibly different
    artifacts, not artifacts that differ only by a missing key.
    """
    out = {}
    if region:
        for suffix in ("", "_global"):
            p = os.path.join(root, "data", "argo_cohort", f"{region}{suffix}.nc")
            if os.path.exists(p):
                out[f"argo_cohort:{region}{suffix}"] = sha256_file(p)
                break
        else:
            out[f"argo_cohort:{region}"] = "absent"
    rp = os.path.join(root, "data", "reference", "manifest.json")
    out["reference_products"] = sha256_file(rp) if os.path.exists(rp) else "absent"
    return out


def resolve_pinned_checkpoint(name: str, dirs: list[str],
                              pin: str | None) -> tuple[str | None, list[str]]:
    """Find the checkpoint ``name`` that a freeze record PINNED, by its bytes.

    Returns ``(path, decoys)``: the file whose SHA-256 equals ``pin``, and the
    same-named files that do not.  With ``pin=None`` this degrades to plain
    first-match-wins search and reports no decoys.

    Why this is not "just open the file"
    ------------------------------------
    P7's whole guarantee is that the model scored is the model registered, and
    filename search defeats it silently.  `outputs/argo_P7_<region>/` can hold
    models an ABANDONED earlier open trained under exactly the names the freeze
    later pinned in a different directory; whichever directory comes first on
    the search path wins, and nothing warns.  It happened: three rows of a
    reported holdout table were produced by checkpoints the freeze record did
    not pin, four hours older than the frozen ones.

    The freeze's own before/after check cannot catch this.  It re-derives the
    checkpoint hashes over ONE directory on both sides, so a freeze and an open
    that resolve the same *filename* to different *files* both pass it.  The
    ambiguity has to be resolved where the bytes are read, and a pin makes it
    decidable: the pin says which file is meant.
    """
    cands = [os.path.join(d, name) for d in dirs
             if d and os.path.exists(os.path.join(d, name))]
    if not cands:
        return None, []
    if pin is None:
        return cands[0], []
    hits = [c for c in cands if sha256_file(c) == pin]
    decoys = [c for c in cands if c not in hits]
    return (hits[0] if hits else None), decoys


# --------------------------------------------------------------------------
# S8 / S9 — the signed result artifact
# --------------------------------------------------------------------------
@dataclass
class ResultArtifact:
    """What a track hands over for a cross-check package.

    Fields map one-to-one onto plan S9's list of what to report, so producing
    the report is a rendering of this object rather than a separate act of
    bookkeeping that can disagree with what ran.
    """
    package: str                       # "P0" ... "P7"
    track: str                         # "A" (main) | "B" (intern)
    region: str
    results: dict                      # metric tables, CI tables, per-seed
    seeds: list = field(default_factory=list)
    command: str = ""
    checkpoint_hashes: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)   # targets, WMOs, months
    warnings: list = field(default_factory=list)
    notes: str = ""
    protocol_version: str = PROTOCOL_VERSION
    protocol_hash_: str = ""
    git_commit_: str = ""
    git_dirty_: bool = False
    data_manifests: dict = field(default_factory=dict)
    #: whole-collection manifest hashes: provenance only, never exact-matched
    collection_manifests: dict = field(default_factory=dict)
    created_utc: str = ""
    host: str = ""

    def finalize(self, root: str) -> "ResultArtifact":
        self.protocol_hash_ = protocol_hash()
        self.git_commit_ = git_commit()
        self.git_dirty_ = git_dirty()
        self.data_manifests = data_manifest_hashes(root, self.region)
        self.collection_manifests = collection_manifest_hashes(root)
        self.created_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.host = platform.node()
        if self.git_dirty_:
            self.warnings.append(
                "git working tree was dirty: the recorded commit does not "
                "fully identify the code that produced these numbers")
        return self

    def write(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path + ".tmp", "w") as f:
            json.dump(asdict(self), f, indent=1, default=str)
        os.replace(path + ".tmp", path)
        return sha256_file(path)

    @staticmethod
    def read(path: str) -> "ResultArtifact":
        return ResultArtifact(**json.load(open(path)))


#: S6 "exact-match quantities" — any mismatch is a protocol problem, not a
#: numerical disagreement.  Read off a ResultArtifact by `crosscheck.py`.
EXACT_MATCH_FIELDS = (
    "protocol_hash_",
    "protocol_version",
    "data_manifests",
    "counts",
)
