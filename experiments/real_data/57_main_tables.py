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

  .venv/bin/python experiments/real_data/57_main_tables.py --suffix _ext
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
ap.add_argument("--size-arms", nargs="*",
                default=["_anom", "_anom_910k", "_anom_1m7"],
                help="artifact suffixes to compare as a model-size ladder")
ap.add_argument("--pretrain-arms", nargs="*", default=["_recent3_obs", "_recent3_ft"],
                help="artifact suffixes compared as from-scratch vs simulation-pretrained")
ap.add_argument("--density-arms", nargs="*", default=["_anom_p24", "_anom"],
                help="artifact suffixes to compare as a profile-count stress")
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
#: References shown beside the model in Table 1, so the lead trend can be
#: judged against things whose behaviour with lead is known in advance.
LEAD_REFS = [("train_climatology", "Climatology"),
             ("source_persistence", "Persistence"),
             ("objective_interpolation", "Causal OI"),
             ("en4", "EN4 (external)")]
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


def meta(arts) -> dict:
    """What an arm actually is, read off its artifacts rather than its name."""
    g = lambda k: {(a.counts or {}).get(k) for a in arts} - {None}
    pars = {r.get("params") for a in arts
            for r in (((a.results or {}).get("model") or {}).get("rows") or {}).values()}
    return {"target": g("anomaly_ref") or {"none"}, "n_profiles": g("n_profiles"),
            "split": g("split_protocol"), "init": g("init_checkpoint_dir"),
            "params": pars - {None}, "seeds": len(arts)}


def provenance(region: str, arts: list) -> list[str]:
    """Name the target and the model, so two arms can never be confused.

    A model fitted on the raw field and one fitted on the WOA23 anomaly field
    produce tables of identical shape and wholly different meaning.
    """
    m = meta(arts)
    tgt = "/".join(sorted(m["target"]))
    label = {"woa23_monthly": "WOA23 monthly anomaly",
             "none": "raw field (per-level train mean removed, no seasonal cycle)",
             "unknown": "unrecorded — the cohort did not say"}.get(tgt, tgt)
    bits = [f"**target:** {label}"]
    if len(m["params"]) == 1:
        bits.append(f"{m['params'].pop():,} parameters, identical across the three rows")
    elif m["params"]:
        bits.append(f"**parameter counts differ: {sorted(m['params'])}**")
    if m["n_profiles"]:
        bits.append(f"cap {'/'.join(str(x) for x in sorted(m['n_profiles']))} profiles/month")
    bits.append(f"{m['seeds']} seeds")
    return [f"_{region} — " + "; ".join(bits) + "._", ""]


def arm_rows(region: str, suffixes: list) -> list:
    out = []
    for sfx in suffixes:
        a = load_suffix(region, sfx)
        if a:
            out.append((sfx, meta(a), a))
    return out


def resolving_note(region: str, suffixes: list) -> list[str]:
    """State the noise floor, so a reader cannot mistake spread for an effect.

    A contrast smaller than the seed spread beside it is not resolved by three
    seeds. The two regions differ by nearly an order of magnitude here, so the
    same nominal gap means different things in each.
    """
    sds = []
    for sfx in suffixes:
        a = load_suffix(region, sfx)
        if len(a) > 1:
            v = [x.results["rows"]["dfs_expertlocal_cbottle"]["lead0"]
                 ["channels"]["TEMP"].get("rmse") for x in a]
            v = [y for y in v if y]
            if len(v) > 1:
                sds.append(float(np.std(v)))
    if not sds:
        return []
    worst = max(sds)
    return ["", f"_Seed spread in this region reaches ±{worst:.4f} RMSE across "
            f"three seeds. Any difference in the column above smaller than that "
            f"is not resolved by this many seeds, whichever way it points._", ""]


def table_pretrain(region: str) -> list[str]:
    """Does pretraining on the simulation help? Same split, budget, seeds and
    evaluation; only the starting weights differ."""
    rows = arm_rows(region, args.pretrain_arms)
    if len(rows) < 2:
        return [f"### {region}", "", "_Both arms not yet available._", ""]
    L = [f"### {region}", "",
         "| start | arm | seeds | row | TEMP J lead 0 | TEMP J lead 6 | lead 6 / lead 0 | SALT J lead 0 |",
         "|---|---|---:|---|---:|---:|---:|---:|"]
    for sfx, m, a in rows:
        start = "simulation-pretrained" if m["init"] else "from scratch"
        for key, label in CORE:
            r0, j0 = cell(a, key, "0", "TEMP")
            _, j6 = cell(a, key, args.leads[-1], "TEMP")
            _, js = cell(a, key, "0", "SALT")
            try:
                g = f"{float(j6) / float(j0):.3f}×"
            except ValueError:
                g = NA
            L.append(f"| {start} | `{sfx}` | {m['seeds']} | {label} | {j0} | {j6} | {g} | {js} |")
    return L + [""]


def table_size(region: str) -> list[str]:
    """Does more capacity help? Same target, same data, only width/depth move."""
    rows = arm_rows(region, args.size_arms)
    if len(rows) < 2:
        return [f"### {region}", "",
                f"_Fewer than two model sizes available ({len(rows)} of "
                f"{len(args.size_arms)})._", ""]
    L = [f"### {region} (lead 0, DFS row)", "",
         "| parameters | arm | seeds | TEMP RMSE | TEMP J | SALT RMSE | SALT J |",
         "|---:|---|---:|---:|---:|---:|---:|"]
    for sfx, m, a in sorted(rows, key=lambda r: min(r[1]["params"] or {0})):
        rt, jt = cell(a, "dfs_expertlocal_cbottle", "0", "TEMP")
        rs, js = cell(a, "dfs_expertlocal_cbottle", "0", "SALT")
        pz = f"{min(m['params']):,}" if m["params"] else NA
        L.append(f"| {pz} | `{sfx}` | {m['seeds']} | {rt} | {jt} | {rs} | {js} |")
    return L + resolving_note(region, args.size_arms)


def table_density(region: str) -> list[str]:
    """Does more real Argo help? Same model, only the profile cap moves."""
    rows = arm_rows(region, args.density_arms)
    if len(rows) < 2:
        return [f"### {region}", "", "_Both density arms not yet available._", ""]
    L = [f"### {region} (lead 0, DFS row)", "",
         "| profiles/month (cap) | arm | seeds | TEMP RMSE | TEMP J | SALT RMSE | SALT J |",
         "|---:|---|---:|---:|---:|---:|---:|"]
    for sfx, m, a in sorted(rows, key=lambda r: min(r[1]["n_profiles"] or {0})):
        rt, jt = cell(a, "dfs_expertlocal_cbottle", "0", "TEMP")
        rs, js = cell(a, "dfs_expertlocal_cbottle", "0", "SALT")
        nz = "/".join(str(x) for x in sorted(m["n_profiles"])) if m["n_profiles"] else NA
        L.append(f"| {nz} | `{sfx}` | {m['seeds']} | {rt} | {jt} | {rs} | {js} |")
    return L + resolving_note(region, args.density_arms) + ["The cap is not the delivered count: a month supplies fewer "
                "profiles than the cap whenever it has fewer, and training "
                "withholds ~30 % of each month's floats as targets.", ""]


def table1(region, arts) -> list[str]:
    L = [f"### {region}", "",
         "| method | channel | " + " | ".join(
             f"lead {l} mo — RMSE / J" for l in args.leads)
         + f" | lead {args.leads[-1]} / lead {args.leads[0]} |",
         "|---|---|" + "---|" * len(args.leads) + "---:|"]
    for row, label in CORE + LEAD_REFS:
        for ch in VARS:
            cells = []
            for l in args.leads:
                r, j = cell(arts, row, l, ch)
                cells.append(f"{r} / {j}")
            # How much does error actually grow over the horizon? On the raw
            # field this sat near 1.00x, which is what made the forecast look
            # suspicious: the query carries the target's calendar month and the
            # raw field is ~90 % climatology, so most of the answer does not
            # depend on lead at all.
            try:
                g = (float(cell(arts, row, args.leads[-1], ch)[0].split(" ")[0])
                     / float(cell(arts, row, args.leads[0], ch)[0].split(" ")[0]))
                grow = f"{g:.2f}×"
            except (ValueError, ZeroDivisionError):
                grow = NA
            L.append(f"| {label} | {ch} | " + " | ".join(cells) + f" | {grow} |")
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


def eval_era(arts) -> str:
    """The years actually scored, read from the artifacts' split protocol.

    Resolved the way `ArgoCohort.apply_splits` resolves it -- last range wins --
    so an overlapping table reports the years that were really evaluated rather
    than the range it declares.
    """
    splits = meta(arts)["split"] if arts else set()
    if len(splits) != 1:
        return "mixed or unrecorded split"
    name = next(iter(splits))
    table = P.SPLIT_PROTOCOLS.get(name)
    if not table:
        return f"{name} split"
    label = {}
    for nm, (lo, hi) in table.items():
        for y in range(lo, hi + 1):
            label[y] = nm
    dev = sorted(y for y, nm in label.items() if nm == "development")
    era = f"{dev[0]}-{dev[-1]}" if dev else "none"
    return f"{name} protocol, evaluation years {era}"


def table4(region, arts) -> list[str]:
    """Two eras, kept apart.

    ECCO V4r4 ends 2017 and the main protocol evaluates on 2022-2024, so the two
    have no month in common. Putting them in one table would compare ECCO on one
    era against the model on another. Each block is therefore scored end to end
    within its own era, with its own DFS / Uniform / OI rows from the same run.
    """
    L = [f"### {region} — {eval_era(arts)}", ""]
    L += _block(arts, [r for r in EXTERNAL if r[0] != "ecco"],
                "ECCO V4r4 ends 2017 and cannot be scored on this era at all. "
                "It appears in the block below.")
    eco = load_suffix(region, args.suffix + "_ecco")
    L += [f"### {region} — ECCO-overlap protocol"
          + (f" ({eval_era(eco)})" if eco else ""), ""]
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


HEAD = """<!-- generated by experiments/real_data/57_main_tables.py — do not edit by hand -->

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
>
> **Target.** Where the provenance line reads *WOA23 monthly anomaly*, the model
> is trained and scored on the anomaly field: the WOA23 monthly climatology is
> subtracted at each profile's own cell, level and calendar month. Zero anomaly
> is then exactly that climatology, so `J` is measured against a seasonally and
> spatially varying baseline instead of a single mean vertical profile. J values
> are therefore much closer to 1 than on the raw field, and **are not comparable
> with earlier raw-field tables**. EN4 and ECCO are moved onto the same anomaly
> before scoring, with the same climatology.
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
        if tno == 1:
            out += provenance(region, arts)
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
    if tno == 1:
        out += ["**The last column is the forecast-growth check.** It is how "
                "much error grows from the first lead to the last. Near 1.00x "
                "means the horizon costs nothing, which is the behaviour that "
                "looked suspicious on the raw field — there the query carries "
                "the target's calendar month and the field is ~90 % "
                "climatology, so most of the answer never depended on lead. On "
                "the anomaly field that crutch is gone, so this column is the "
                "direct test of whether the forecast is real.", "",
                "**How to read the trend.** Every lead is scored on the same "
                "target months and the same held-out floats; only the source "
                "month moves. Two rows therefore work as checks. *Climatology* "
                "predicts zero anomaly and never sees the source month, so it "
                "must read exactly 1.00x — if it does not, the target set is "
                "not fixed. *Persistence* carries the nearest source-month "
                "float forward, so its error must grow with lead as the ocean "
                "decorrelates. Persistence is not automatically better than "
                "climatology: when floats are sparser than the anomaly "
                "correlation scale, the nearest float's anomaly is mostly noise "
                "at the target, so persistence can score worse than "
                "climatology even at lead 0, and in fast-decorrelating regions "
                "it stops degrading within a few months. A learned row behaving "
                "physically beats both at short lead, and its error grows "
                "toward the climatology floor (J -> 1) as lead increases.", ""]

out += ["## Table 5 — Model size", ""]
for region in args.regions:
    out += table_size(region)
out += ["**The question this table asks:** does capacity help? Every arm shares "
        "the target, the data, the seeds and the training budget; only width and "
        "depth change. A flat column says the ceiling is not capacity.", ""]

out += ["## Table 7 — Simulation pretraining", ""]
for region in args.regions:
    out += table_pretrain(region)
out += ["**The question this table asks:** does pretraining on the CESM2-LE anomaly "
        "field, sampled at the real float positions, then fine-tuning on observations "
        "beat training on observations alone? The simulation store holds only 72 "
        "months, and pretraining validation stopped improving within 2000-3000 steps "
        "and never beat the simulation's own climatology in most runs, so a null or "
        "negative result here speaks to this simulation's size, not to pretraining "
        "in general.", ""]

out += ["**Split note for Tables 5 and 6.** The size ladder and the density arms "
        "were run before the most-recent-three-years split, on the main protocol "
        "(train 2000-2018, validation 2019-2021, test 2022-2024), at lead 0 only. "
        "Lead 0 was unaffected by the lead-target leak, so they remain valid, but "
        "their test years differ from Tables 1-4 and 7.", ""]

out += ["## Table 6 — Input density", ""]
for region in args.regions:
    out += table_density(region)
out += ["**The question this table asks:** does more real Argo help? Causal OI "
        "converts added profiles into accuracy automatically, so if the learned "
        "rows do not, the limit is the model rather than the observations.", ""]

path = os.path.join(REPORTS, "main_tables.md")
with open(path, "w") as f:
    f.write("\n".join(out))
print(f"wrote {os.path.relpath(path, ROOT)}"
      + ("" if any_data else "  (no extended-grid artifacts yet)"))
