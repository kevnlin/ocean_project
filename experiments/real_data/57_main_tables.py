"""The four main tables, assembled from the signed artifacts.

Nothing here recomputes a number: every value is read from a `ResultArtifact`,
so the tables cannot drift from the runs they describe.

  Table 1  DFS vs Uniform vs Count, by forecast lead (0/1/3/6 months)
  Table 2  the same three rows, by depth band (0-100 / 100-300 / 300-700 /
           700-1400 m)
  Table 3  against the traditional baselines: climatology, causal OI, Count,
           Uniform, DFS
  Table 4  external references: ECCO V4r4 and EN4 alongside DFS/Uniform/OI

`J = RMSE_model / RMSE_climatology`, computed on exactly the same slice
(same rows scored, same lead, same band) as the RMSE beside it, so the
normalisation cannot be taken from a different cohort than the numerator.

  .venv/bin/python experiments/57_main_tables.py --suffix _ext
"""
from __future__ import annotations

import argparse, glob, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np

from ocean_tokenizer import protocol as P

ap = argparse.ArgumentParser()
ap.add_argument("--outputs", default=None)
ap.add_argument("--reports", default=None)
ap.add_argument("--suffix", default="_ext",
                help="artifact-dir suffix for the extended-grid runs")
ap.add_argument("--regions", nargs="+", default=["gulfstream", "npac_gyre"])
ap.add_argument("--leads", nargs="+", default=["0", "1", "3", "6"])
args = ap.parse_args()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUTS = args.outputs or os.path.join(ROOT, "outputs")
REPORTS = args.reports or os.path.join(ROOT, "reports", "real_data")
VARS = P.CHANNELS
NA = "—"

CORE = [("dfs_expertlocal_cbottle", "**DFS (ours)**"),
        ("uniform_expertlocal_cbottle", "Uniform"),
        ("count_expertlocal_cbottle", "Count / Perceiver")]
TRAD = [("train_climatology", "Climatology"),
        ("objective_interpolation", "Causal OI"),
        ("count_expertlocal_cbottle", "Count"),
        ("uniform_expertlocal_cbottle", "Uniform (neural)"),
        ("dfs_expertlocal_cbottle", "**DFS (ours)**")]
EXTERNAL = [("dfs_expertlocal_cbottle", "**DFS (ours)**", False),
            ("uniform_expertlocal_cbottle", "Uniform", False),
            ("objective_interpolation", "Causal OI", False),
            ("ecco", "ECCO V4r4", True),
            ("en4", "EN4", True)]
BANDS = ["0-100m", "100-300m", "300-700m", "700-1400m"]


def load(region: str) -> list:
    pat = os.path.join(OUTPUTS, f"argo_P0_{region}{args.suffix}",
                       "artifact_P0_seed*.json")
    out = []
    for p in sorted(glob.glob(pat)):
        d = json.load(open(p))
        known = set(P.ResultArtifact.__dataclass_fields__)
        out.append(P.ResultArtifact(**{k: v for k, v in d.items() if k in known}))
    return out


def cell(arts, row, lead, ch, band=None):
    """mean ± sd over seeds of RMSE, and of J against the same slice."""
    rs, js = [], []
    for a in arts:
        try:
            e = a.results["rows"][row][f"lead{lead}"]["channels"][ch]
            c0 = a.results["rows"]["train_climatology"][f"lead{lead}"]["channels"][ch]
            v = e["by_band"][band] if band else e.get("rmse")
            f = c0["by_band"][band] if band else c0.get("rmse")
        except (KeyError, TypeError):
            continue
        if v is None or not np.isfinite(v):
            continue
        rs.append(float(v))
        if f and np.isfinite(f) and f > 0:
            js.append(float(v) / float(f))
    if not rs:
        return NA, NA
    r = (f"{np.mean(rs):.4f}" if len(rs) == 1
         else f"{np.mean(rs):.4f} ± {np.std(rs):.4f}")
    j = NA if not js else f"{np.mean(js):.3f}"
    return r, j


def table1(region, arts) -> list[str]:
    L = [f"### {region}", "",
         "| method | channel | " + " | ".join(
             f"lead {l} mo — RMSE / J" for l in args.leads) + " |",
         "|---|---|" + "---|" * len(args.leads)]
    for row, label in CORE:
        for ch in VARS:
            cells = []
            for l in args.leads:
                r, j = cell(arts, row, l, ch)
                cells.append(f"{r} / {j}")
            L.append(f"| {label} | {ch} | " + " | ".join(cells) + " |")
    return L + [""]


def table2(region, arts) -> list[str]:
    L = [f"### {region} (lead 0)", "",
         "| method | channel | " + " | ".join(BANDS) + " |",
         "|---|---|" + "---:|" * len(BANDS)]
    for row, label in CORE:
        for ch in VARS:
            cells = [cell(arts, row, "0", ch, b)[0] for b in BANDS]
            L.append(f"| {label} | {ch} | " + " | ".join(cells) + " |")
    L += ["", "J against climatology, same slices:", "",
          "| method | channel | " + " | ".join(BANDS) + " |",
          "|---|---|" + "---:|" * len(BANDS)]
    for row, label in CORE:
        for ch in VARS:
            cells = [cell(arts, row, "0", ch, b)[1] for b in BANDS]
            L.append(f"| {label} | {ch} | " + " | ".join(cells) + " |")
    return L + [""]


def table3(region, arts) -> list[str]:
    L = [f"### {region} (lead 0)", "",
         "| method | TEMP RMSE | TEMP J | SALT RMSE | SALT J |",
         "|---|---:|---:|---:|---:|"]
    for row, label in TRAD:
        rt, jt = cell(arts, row, "0", "TEMP")
        rs, jsv = cell(arts, row, "0", "SALT")
        L.append(f"| {label} | {rt} | {jt} | {rs} | {jsv} |")
    return L + [""]


def load_suffix(region: str, suffix: str) -> list:
    pat = os.path.join(OUTPUTS, f"argo_P0_{region}{suffix}",
                       "artifact_P0_seed*.json")
    out = []
    for p_ in sorted(glob.glob(pat)):
        d = json.load(open(p_))
        known = set(P.ResultArtifact.__dataclass_fields__)
        out.append(P.ResultArtifact(**{k: v for k, v in d.items() if k in known}))
    return out


def _block(arts, rows, note) -> list[str]:
    L = ["| method | external reference? | TEMP RMSE | TEMP J | SALT RMSE | SALT J |",
         "|---|---|---:|---:|---:|---:|"]
    for row, label, ext in rows:
        rt, jt = cell(arts, row, "0", "TEMP")
        rs, jsv = cell(arts, row, "0", "SALT")
        L.append(f"| {label} | {'**yes**' if ext else 'no'} | {rt} | {jt} | "
                 f"{rs} | {jsv} |")
    return L + ["", note, ""]


def table4(region, arts) -> list[str]:
    """Two eras, kept apart.

    ECCO V4r4 ends 2017 and the main protocol evaluates on 2022-2024, so the two
    have no month in common. Putting them in one table would compare ECCO on one
    era against the model on another. Each block is therefore scored end to end
    within its own era, with its own DFS / Uniform / OI rows from the same run.
    """
    L = [f"### {region} — main protocol (evaluation era 2022-2024)", ""]
    L += _block(arts, [r for r in EXTERNAL if r[0] != "ecco"],
                "ECCO V4r4 ends 2017 and cannot be scored on this era at all. "
                "It appears in the block below.")
    eco = load_suffix(region, args.suffix + "_ecco")
    L += [f"### {region} — ECCO-overlap protocol (evaluation era 2015-2017)", ""]
    if not eco:
        L += ["_Not run._", ""]
        return L
    L += _block(eco, EXTERNAL,
                "A **secondary protocol**: the eras are shifted inside ECCO "
                "V4r4's coverage (train 2000-2012 / val 2013-2014 / eval "
                "2015-2017). Legitimate because the held-out float cohort is "
                "WMO-disjoint and year-independent, so shifting the era does "
                "not change which floats are held out. The learned rows here "
                "were **trained on 2000-2012**, so 2015-2017 is genuinely out "
                "of sample; an earlier version reused main-protocol "
                "checkpoints trained through 2018 and was in-sample in time. "
                "Never merged into the main headline table.")
    return L


HEAD = """<!-- generated by experiments/57_main_tables.py — do not edit by hand -->

> **Ground truth is held-out real Argo.** Every number is scored against
> measurements from **WMO-disjoint held-out floats** that appear in no training
> month. ECCO and EN4 are **external references, not ground truth** — both
> assimilate the very floats being scored, so they are upper references that
> have already seen the answer.
>
> `J = RMSE_model / RMSE_climatology`, computed on the identical slice as the
> RMSE beside it. J < 1 beats climatology.
>
> Architecture and training are matched across DFS / Uniform / Count: same
> encoder, same resampler budget, same decoder, same steps, same seeds. Only
> the observation-mass rule differs.
"""

out = ["# Main tables — real-data results", "", HEAD, ""]
any_data = False
for tno, (title, fn) in enumerate([
        ("Table 1 — Main real-data results, by forecast lead", table1),
        ("Table 2 — Performance by depth band", table2),
        ("Table 3 — Against traditional baselines", table3),
        ("Table 4 — External-reference comparison", table4)], 1):
    out += [f"## {title}", ""]
    for region in args.regions:
        arts = load(region)
        if not arts:
            out += [f"### {region}", "", "_No extended-grid run found._", ""]
            continue
        any_data = True
        out += fn(region, arts)
    if tno == 2:
        out += ["**The question this table asks:** does DFS help more in the "
                "deeper, more sparsely observed bands? Compare the J columns "
                "across bands rather than the RMSE columns — RMSE falls with "
                "depth simply because deep variability is smaller, so only the "
                "climatology-normalised J is comparable between bands.", ""]
    if tno == 4:
        out += ["**Both ECCO and EN4 assimilate the very floats being scored.** "
                "They are upper references that have already seen the answer, "
                "not competitors, and their beating the model is expected "
                "rather than a result.", ""]

path = os.path.join(REPORTS, "main_tables.md")
with open(path, "w") as f:
    f.write("\n".join(out))
print(f"wrote {os.path.relpath(path, ROOT)}"
      + ("" if any_data else "  (no extended-grid artifacts yet)"))
