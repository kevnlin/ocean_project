"""Tables of the radius / local-scale runs on the OI-anchored model.

Reads the runs of ``82_radius_scale_campaign.py`` (the ``control`` arm is the
reference model trained here) and the evaluation-time radius test of
``84_radius_at_evaluation.py``; writes one markdown report and a JSON file with
every number in it. The matched campaign's own published score for the reference
model is quoted from its committed table, for comparison with our control.

All scores are computed from the saved prediction arrays, in the matched
campaign's units, so a control and an arm are pooled by the same arithmetic:

  rmse                  pooled over cells, physical units (the campaign's score);
  rmse_local            the error divided by the local anomaly scale of its
                        cell (table ``anom_month``), per-depth z units: every
                        region counts alike;
  rmse_quiet_half       physical RMSE over the half of the cells with the
                        smaller local scale;
  rmse_energetic_tenth  the same over the tenth with the largest;
  coverage_68           share of errors inside the predicted one-sigma.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

OUTPUT = ROOT / "outputs/radius_scale_20261008"
CH = ("TEMP", "SALT")
PUBLISHED = ROOT / "reports/synthetic/concise_english_comparison_20261007.csv"
ORDER = ("oi", "control", "r250", "r500", "r750", "qanom_cell", "qanom_month", "qres_cell",
         "qres_month", "qanom_month_lossz", "qres_month_lossz", "anom_month", "res_month")
WORDS = {"oi": "frozen OI (the anchor)", "control": "no option (the reference model)",
         "anom_cell": "anomaly scale, cell", "anom_month": "anomaly scale, cell × month",
         "res_cell": "OI-error scale, cell", "res_month": "OI-error scale, cell × month"}


def describe(arm):
    parts = []
    for atom in arm.split("+"):
        name = atom.removesuffix("_lossz")
        loss = ", loss in z units" if atom != name else ""
        if atom in ("oi", "control"):
            parts.append(WORDS[atom])
        elif name[:1] == "q" and name[1:] in WORDS:
            parts.append(WORDS[name[1:]] + loss)
        elif name in WORDS:
            parts.append(WORDS[name] + ", each profile by its own entry (first design)" + loss)
        elif atom[:1] == "r" and atom[1:].isdigit():
            parts.append(f"radius {atom[1:]} km")
        elif atom[:1] == "l" and atom[1:].isdigit():
            parts.append(f"{atom[1:]} latent slots")
        else:
            parts.append(atom)
    return "; ".join(parts)


def published_reference(path=PUBLISHED):
    """The matched campaign's committed 2004 score of the Argo-only reference model."""
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if (row["family"], row["inputs"], row["split"]) == ("local_transformer192", "Argo", "validation"):
                return {"seeds": int(row["completed_seeds"]),
                        "TEMP": (float(row["temperature_rmse_degC_mean"]), float(row["temperature_rmse_degC_sample_sd"])),
                        "SALT": (float(row["salinity_rmse_PSU_mean"]), float(row["salinity_rmse_PSU_sample_sd"]))}
    return None


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else None


def run_metrics(arrays, scale, level_std):
    """Scores of one run from its arrays (``mean``, ``std``, ``target`` in z units).

    ``scale`` is the reference local scale of each scored cell, ``(N, 2)``;
    ``level_std`` the per-depth standard deviations that turn z into physical
    units. ``std`` may be None for a method without a predicted spread.
    """
    out = {}
    for j, ch in enumerate(CH):
        ok = np.isfinite(arrays["target"][:, j])
        error = (arrays["mean"][:, j] - arrays["target"][:, j])[ok].astype(np.float64)
        physical = error * np.asarray(level_std[ch])[arrays["level"][ok]]
        s = scale[ok, j]
        out[ch] = {"n": int(ok.sum()), "rmse": rms(physical), "rmse_local": rms(error / s),
                   "rmse_quiet_half": rms(physical[s <= np.median(s)]) if ok.any() else None,
                   "rmse_energetic_tenth": rms(physical[s >= np.percentile(s, 90)]) if ok.any() else None,
                   "coverage_68": None if arrays.get("std") is None or not ok.any()
                   else float(np.mean(np.abs(error) <= arrays["std"][ok, j]))}
    return out


def over_seeds(runs, ch, key):
    """``(mean, sample sd or None, number of seeds)`` of one score."""
    values = [r[ch][key] for r in runs if r[ch][key] is not None]
    if not values:
        return None, None, 0
    return float(np.mean(values)), (float(np.std(values, ddof=1)) if len(values) > 1 else None), len(values)


def cell(stat, digits):
    mean, sd, _ = stat
    if mean is None:
        return "—"
    return f"{mean:.{digits}f}" + (f" ± {sd:.{digits}f}" if sd is not None else "")


def load(folder, seed, split):
    """Arrays and stored scores of one run and split, or None if absent."""
    folder = Path(folder)
    summary_path, arrays_path = folder / f"summary_seed{seed}.json", folder / f"{split}_seed{seed}.npz"
    if not (summary_path.exists() and arrays_path.exists()):
        return None
    summary = json.loads(summary_path.read_text())
    if split not in summary:
        return None
    with np.load(arrays_path, allow_pickle=False) as z:
        arrays = {name: z[name] for name in ("mean", "std", "target", "level", "profile", "baseline",
                                             "normalization_scale")}
    return {"arrays": arrays, "scores": summary[split]["scores"],
            "baseline_scores": summary[split]["baseline_scores"], "best_step": summary["best_step"]}


def parse_folder(name):
    """``(arm, inputs, seed)`` of a campaign run folder, or None for anything else."""
    match = re.fullmatch(r"(.+)_(argo|surface)_s(\d+)", name)
    if match is None:
        return None
    # leading radius and slot-count options are joined to the rest with "+"
    arm = re.sub(r"^((?:[rl]\d+_)+)", lambda m: m.group(1).replace("_", "+"), match[1])
    return arm, match[2], int(match[3])


def discover(output, inputs):
    """``{arm: {seed: folder}}`` of this campaign's finished runs for one input set."""
    found = {}
    for path in sorted(Path(output).glob("*/summary_seed*.json")):
        parsed = parse_folder(path.parent.name)
        if parsed is None or parsed[1] != inputs:
            continue
        if json.loads(path.read_text()).get("completed_steps") == 15000:
            found.setdefault(parsed[0], {})[parsed[2]] = path.parent
    return found


def sort_key(arm):
    return (ORDER.index(arm) if arm in ORDER else len(ORDER), arm)


def table(rows, split_scores):
    """Markdown table; ``rows`` is ``[(label, seeds, [per-seed metrics], [selection scores])]``."""
    lines = ["| Arm | Seeds | TEMP °C | SALT PSU | Selection score | TEMP local | SALT local "
             "| TEMP quiet half | TEMP energetic tenth | SALT quiet half | SALT energetic tenth "
             "| 1σ coverage T / S |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, seeds, metrics, selection in rows:
        stat = lambda ch, key: over_seeds(metrics, ch, key)
        sel = (float(np.mean(selection)), float(np.std(selection, ddof=1)) if len(selection) > 1 else None,
               len(selection))
        cov = [stat(ch, "coverage_68")[0] for ch in CH]
        lines.append("| " + " | ".join([
            label, ", ".join(str(s) for s in seeds),
            cell(stat("TEMP", "rmse"), 4), cell(stat("SALT", "rmse"), 4), cell(sel, 4),
            cell(stat("TEMP", "rmse_local"), 3), cell(stat("SALT", "rmse_local"), 3),
            cell(stat("TEMP", "rmse_quiet_half"), 4), cell(stat("TEMP", "rmse_energetic_tenth"), 4),
            cell(stat("SALT", "rmse_quiet_half"), 4), cell(stat("SALT", "rmse_energetic_tenth"), 4),
            "—" if cov[0] is None else f"{cov[0]:.2f} / {cov[1]:.2f}"]) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--cohort", default=None, help="another synthetic cohort, by file name without .nc")
    args = p.parse_args()
    if args.output is None:
        args.output = OUTPUT / args.cohort if args.cohort else OUTPUT
    if args.report is None:
        args.report = ROOT / "reports/synthetic" / ("radius_scale_feedback_20261008"
                                                    + (f"_{args.cohort}" if args.cohort else "") + ".md")
    from ocean_tokenizer import local_scale, synth_argo_eval as S
    c, norm, _ = S.load(str(ROOT), cohort=args.cohort)
    level_std = {ch: np.asarray(norm.std[ch]) for ch in CH}
    with np.load(args.output / "scale_tables.npz", allow_pickle=False) as z:
        yardstick = z["anom_month"]
        table_metadata = json.loads(str(z["metadata"].item()))
    month_of_year = np.asarray(c.month_index) % 12

    def scored(run):
        a = run["arrays"]
        rows = a["profile"]
        scale = local_scale.lookup(yardstick, c.lat[rows], c.lon[rows], month_of_year[rows])
        scale = scale[np.arange(len(rows)), a["level"]]
        metrics = run_metrics(a, scale, level_std)
        for ch in CH:      # the arrays must reproduce the score the run stored
            if abs(metrics[ch]["rmse"] / run["scores"][ch]["rmse_physical"] - 1.) > 1e-4:
                raise ValueError("saved arrays and stored score disagree")
        return metrics, scale

    numbers, sections = {}, []
    for inputs in ("argo", "surface"):
        runs = discover(args.output, inputs)
        for split, title in (("validation", "2004 validation"), ("development", "2005")):
            per_arm, oi = {}, None
            for arm in sorted(runs, key=sort_key):
                for seed, folder in sorted(runs[arm].items()):
                    run = load(folder, seed, split)
                    if run is None:
                        continue
                    metrics, scale = scored(run)
                    per_arm.setdefault(arm, []).append((seed, metrics, run["scores"]["macro_z"], run["best_step"]))
                    if oi is None:       # the anchor is the same field in every run
                        anchor = {**run["arrays"], "mean": run["arrays"]["baseline"], "std": None}
                        oi = run_metrics(anchor, scale, level_std), run["baseline_scores"]["macro_z"]
            if not per_arm:
                continue
            numbers[f"{inputs}/{split}"] = {"oi": {"metrics": oi[0], "selection_score": oi[1]}, **{
                arm: [{"seed": s, "metrics": m, "selection_score": z, "best_step": b} for s, m, z, b in items]
                for arm, items in per_arm.items()}}
            for heading, keep in (("seed 1234", lambda s: s == 1234), ("all finished seeds", lambda s: True)):
                rows = [(describe("oi"), [], [oi[0]], [oi[1]])]
                for arm, items in per_arm.items():
                    chosen = [item for item in items if keep(item[0])]
                    if chosen:
                        rows.append((describe(arm), [i[0] for i in chosen], [i[1] for i in chosen],
                                     [i[2] for i in chosen]))
                if heading == "all finished seeds" and all(len(r[1]) <= 1 for r in rows):
                    continue
                name = "Argo only" if inputs == "argo" else "Argo + surface"
                sections.append(f"### {name}, {title}, {heading}\n\n" + table(rows, split))

    radius_sections = []
    for inputs, name in (("argo", "Argo only"), ("surface", "Argo + surface")):
        path = args.output / f"radius_at_evaluation_{inputs}.json"
        if not path.exists():
            continue
        r0 = json.loads(path.read_text())
        numbers[f"{inputs}/radius_at_evaluation"] = r0
        lines = ["| Radius km | Neighbours kept | Queries with none | TEMP °C | SALT PSU |", "|---|---|---|---|---|"]
        for i, radius in enumerate(r0["radii_km"]):
            rows = [run["by_radius"][i] for run in r0["runs"]]
            stat = lambda key: (float(np.mean([r[key] for r in rows])),
                                float(np.std([r[key] for r in rows], ddof=1)) if len(rows) > 1 else None, len(rows))
            lines.append(f"| {'none' if radius is None else int(radius)} | {stat('neighbours_kept')[0]:.1f} "
                         f"| {100 * stat('share_of_queries_with_none')[0]:.1f} % | {cell(stat('TEMP'), 4)} "
                         f"| {cell(stat('SALT'), 4)} |")
        radius_sections.append(f"### {name}: {len(r0['runs'])} trained control runs, radius at evaluation only\n\n"
                               + "\n".join(lines))

    scale_lines = ["| Table | TEMP 5th / 50th / 95th percentile | SALT 5th / 50th / 95th | At the 0.2 floor, T / S |",
                   "|---|---|---|---|"]
    for name, d in table_metadata["scale_over_training_profiles"].items():
        pick = lambda ch: " / ".join(f"{d[ch]['percentiles_5_25_50_75_95'][i]:.2f}" for i in (0, 2, 4))
        scale_lines.append(f"| `{name}` | {pick('TEMP')} | {pick('SALT')} | "
                           f"{100 * d['TEMP']['share_at_floor']:.1f} % / {100 * d['SALT']['share_at_floor']:.1f} % |")
    published = published_reference() if args.cohort is None else None
    reference_line = ("For comparison, the matched campaign's committed table "
        f"(`reports/synthetic/{PUBLISHED.name}`) gives the reference model, Argo only, on 2004 as "
        f"{published['TEMP'][0]:.4f} ± {published['TEMP'][1]:.4f} °C and {published['SALT'][0]:.4f} ± "
        f"{published['SALT'][1]:.4f} PSU over {published['seeds']} seeds; the `control` rows below are the "
        "same settings trained here.") if published else ""
    findings = args.report.with_suffix(".findings.md")
    text = "\n\n".join([
        "# Neighbour radius and local normalisation on the OI-anchored model"
        + (f" — cohort `{args.cohort}`" if args.cohort else ""),
        findings.read_text().strip() if findings.exists() else "",
        "## Tables\n\nGenerated by `experiments/synthetic/83_radius_scale_report.py`; every number is in "
        f"`{args.report.with_suffix('.json').name}`. Columns are defined in that script's header. "
        "A ± is the sample standard deviation over seeds.",
        *(["## Radius applied to trained models, no retraining", *radius_sections] if radius_sections else []),
        "## The scale tables\n\nThe scale over the training-year profiles (1.0 is the level's own RMS). "
        f"Cross-fitted OI error on the training years: {table_metadata['residual']['train_year_rmse_physical']['TEMP']:.4f} °C, "
        f"{table_metadata['residual']['train_year_rmse_physical']['SALT']:.4f} PSU.\n\n" + "\n".join(scale_lines),
        "## Trained arms\n\n" + reference_line, *sections]) + "\n"
    args.report.write_text(text.replace("\n\n\n\n", "\n\n"))
    args.report.with_suffix(".json").write_text(json.dumps(numbers, indent=1) + "\n")
    print("wrote " + str(args.report))


if __name__ == "__main__":
    main()
