"""Plot measured, registered ocean latent results as a shareable PNG and SVG.

The default refuses an unfinished campaign. ``--allow-pending`` creates an
explicitly labelled progress figure and never fills missing scores or curves.
2022--2023 remain previously used development years, not an independent test.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/latent_ocean"
CHANNELS = ("TEMP", "SALT")
SEEDS = (1234, 1235, 1236)
LABELS = {"dense": "Dense latent", "soft_moe": "Soft MoE",
          "local_transformer": "Local Transformer", "analysis_only": "OI-only"}
COLORS = {"dense": "#3D6B89", "soft_moe": "#8A667F",
          "local_transformer": "#557C69", "analysis_only": "#62676D"}
CHANNEL_COLORS = {"TEMP": "#3D6B89", "SALT": "#B56E47"}

_spec = importlib.util.spec_from_file_location("ocean_latent_report_figures", Path(__file__).with_name("71_latent_report.py"))
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def prefix(tag: str) -> str:
    return tag.rsplit("_s", 1)[0]


def finite(value, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"nonfinite {name}")
    return value


def configuration_label(job: dict, capacity: bool = True) -> str:
    variant = "analysis_only" if job.get("analysis_only") else job["variant"]
    label = LABELS[variant]
    if capacity and variant != "analysis_only":
        label += " · " + ("large" if "_large_" in job["tag"] else "base")
    return label


def measured_curve(summary: dict, job: dict) -> dict:
    """Use actual validation evaluations, including the zero-initialized anchor."""
    R.match_job(job, summary)
    initial = summary.get("initial")
    if not initial or "scores" not in initial:
        raise ValueError(f"completed run lacks the measured step-0 score: {job['tag']}")
    steps = [0]
    values = [finite(initial["scores"]["macro_z"], "initial macro_z")]
    for record in summary["history"]:
        step = record["step"]
        if not isinstance(step, int) or step <= steps[-1] or step > job["steps"]:
            raise ValueError(f"invalid measured validation steps: {job['tag']}")
        steps.append(step)
        values.append(finite(record["validation_macro_z"], "validation macro_z"))
    if steps[-1] != job["steps"]:
        raise ValueError(f"completed run does not reach its registered budget: {job['tag']}")
    best_step = summary["best_step"]
    if best_step not in steps:
        raise ValueError(f"best checkpoint is not a measured evaluation: {job['tag']}")
    best_value = finite(summary["validation"]["scores"]["macro_z"], "best macro_z")
    if not np.isclose(values[steps.index(best_step)], best_value, rtol=1e-7, atol=1e-9):
        raise ValueError(f"best checkpoint and learning curve disagree: {job['tag']}")
    return {"tag": job["tag"], "label": configuration_label(job),
            "variant": "analysis_only" if job.get("analysis_only") else job["variant"],
            "large": "_large_" in job["tag"], "steps": steps, "macro_z": values,
            "best_step": best_step, "best_macro_z": best_value}


def aggregate_details(details: dict, expected_seeds=SEEDS) -> dict | None:
    """Three-seed spread of channel-wise ratios to the matched anchor."""
    keys = [f"seed{seed}" for seed in expected_seeds]
    if not all(key in details for key in keys):
        return None
    if set(details) != set(keys):
        raise ValueError("development seed identities differ from the registered seeds")
    result = {"seeds": list(expected_seeds), "channels": {}}
    for channel in CHANNELS:
        model = np.array([finite(details[key]["scores"][channel]["J"], channel + " J") for key in keys])
        baseline = np.array([finite(details[key]["baseline_scores"][channel]["J"], "anchor J") for key in keys])
        counts = [details[key]["baseline_scores"][channel]["n"] for key in keys]
        if np.any(baseline <= 0) or not np.allclose(baseline, baseline[0], rtol=1e-10, atol=1e-12) or len(set(counts)) != 1:
            raise ValueError("three seeds do not share the matched anchor scoring evidence")
        ratios = model / baseline
        result["channels"][channel] = {"J_values": model.tolist(), "J_mean": float(model.mean()),
            "J_std": float(model.std(ddof=1)), "anchor_J": float(baseline[0]), "n": int(counts[0]),
            "ratio_values": ratios.tolist(), "ratio_mean": float(ratios.mean()),
            "ratio_std": float(ratios.std(ddof=1))}
    return result


def build_figure_data(report: dict, manifest: dict, selection: dict | None,
                      ablations: dict | None, levels: np.ndarray, expected_seeds=SEEDS) -> dict:
    """Resolve the plotted data without selecting anything on development."""
    if report.get("schema_version") != 1:
        raise ValueError("unsupported comparison schema")
    levels = np.asarray(levels, dtype=float)
    if levels.shape != (20,) or np.any(~np.isfinite(levels)) or np.any(np.diff(levels) <= 0):
        raise ValueError("this figure requires the actual 20 increasing fixed depth levels")
    jobs = manifest["jobs"]
    registered = {job["tag"]: job for job in jobs}
    if len(registered) != len(jobs) or report["registered_count"] != len(jobs):
        raise ValueError("search registration differs from the comparison report")
    completed = {item["job"]["tag"]: item for item in report["search"]}
    pending = []
    if len(completed) != len(report["search"]) or report["completed_count"] != len(completed):
        raise ValueError("duplicate/inconsistent completed search records")
    if set(completed) - set(registered):
        raise ValueError("unregistered run in comparison; integration runs cannot be plotted")
    curves = []
    for job in jobs:
        if job["tag"] not in completed:
            pending.append(f"search: {job['tag']}")
            continue
        item = completed[job["tag"]]
        if item["job"] != job:
            raise ValueError(f"comparison job differs from manifest: {job['tag']}")
        curves.append(measured_curve(item["summary"], job))

    rows, controls, depth = [], [], None
    chosen_tag = None
    if selection is None:
        pending.append("validation-only configuration selection")
    else:
        selected = selection["selected"]
        selected_tags = {job["tag"] for job in selected}
        if selected_tags != set(report["selected_tags"]) or len(selected) != len(selected_tags):
            raise ValueError("frozen selection and comparison disagree")
        if pending or not selected_tags <= set(completed):
            raise ValueError("selection was frozen before all search runs completed")
        groups = {group["configuration"]: group for group in report["groups"]}
        if len(groups) != len(report["groups"]) or set(groups) - {prefix(tag) for tag in selected_tags}:
            raise ValueError("unselected or duplicate development configuration")
        for job in selected:
            R.match_job(job, completed[job["tag"]]["summary"])
            group = groups.get(prefix(job["tag"]))
            data = aggregate_details(group.get("development_details", {}), expected_seeds) if group else None
            if data is None:
                pending.append(f"three-seed development: {prefix(job['tag'])}")
            rows.append({"configuration": prefix(job["tag"]), "label": configuration_label(job), "data": data})
        neural = [job for job in selected if not job.get("analysis_only")]
        if not neural:
            raise ValueError("frozen selection has no shared-state architecture")
        chosen = min(neural, key=lambda job: completed[job["tag"]]["summary"]["validation"]["scores"]["macro_z"])
        chosen_tag = chosen["tag"]
        group = groups.get(prefix(chosen_tag))
        ensemble = group.get("ensemble") if group else None
        if ensemble is None or ensemble.get("n_seeds") != len(expected_seeds):
            pending.append("selected-neural three-seed depth ensemble")
        else:
            records = ensemble["by_level"]
            by_level = {entry["level_index"]: entry for entry in records}
            if len(by_level) != len(records) or set(by_level) != set(range(len(levels))):
                raise ValueError("depth ensemble does not cover the matched 20 observed levels")
            depth = {"configuration": prefix(chosen_tag), "label": configuration_label(chosen),
                     "depth_m": levels.tolist(), "n_seeds": ensemble["n_seeds"], "channels": {}}
            for channel in CHANNELS:
                depth["channels"][channel] = {
                    "delta_J": [finite(by_level[level]["scores"][channel]["J"], "depth J")
                                - finite(by_level[level]["baseline_scores"][channel]["J"], "depth anchor J")
                                for level in range(len(levels))],
                    "n": [int(by_level[level]["scores"][channel]["n"]) for level in range(len(levels))]}

    if ablations is None:
        pending.append("registered fixed-OI controls")
    elif chosen_tag is None or ablations["selected_neural"] != chosen_tag:
        raise ValueError("fixed-OI controls were not registered for the validation-selected neural winner")
    else:
        kinds = {"full": "Full shared state", "latent_off": "No shared latent",
                 "local_off": "No added local path"}
        items = {item["job"]["tag"]: item for item in report["ablations"]}
        registered_controls = {job["tag"]: job for job in ablations["jobs"]}
        if len(items) != len(report["ablations"]) or set(items) - set(registered_controls):
            raise ValueError("unregistered or duplicate mechanism control")
        buckets = {kind: {} for kind in kinds}
        registered_seeds = {kind: set() for kind in kinds}
        for job in ablations["jobs"]:
            if not job.get("freeze_analysis") or job.get("latent_off") and job.get("local_off"):
                raise ValueError("mechanism panel requires the registered fixed-OI component controls")
            kind = "latent_off" if job.get("latent_off") else "local_off" if job.get("local_off") else "full"
            if job["seed"] in registered_seeds[kind]:
                raise ValueError("duplicate registered mechanism seed")
            registered_seeds[kind].add(job["seed"])
            item = items.get(job["tag"])
            if item is None:
                continue
            R.match_job(job, item["summary"])
            if "development_details" in item:
                buckets[kind][f"seed{job['seed']}"] = item["development_details"]
        for kind, label in kinds.items():
            if registered_seeds[kind] != set(expected_seeds):
                raise ValueError("mechanism registration differs from the three expected seeds")
            data = aggregate_details(buckets[kind], expected_seeds)
            if data is None:
                pending.append(f"three-seed fixed-OI development: {kind}")
            controls.append({"configuration": kind, "label": label, "data": data})

    # Every displayed channel ratio must use exactly the same fixed anchor.
    for channel in CHANNELS:
        bases = [row["data"]["channels"][channel] for row in rows + controls if row["data"]]
        if bases and any(base["n"] != bases[0]["n"] or not np.isclose(base["anchor_J"], bases[0]["anchor_J"], rtol=1e-10, atol=1e-12) for base in bases[1:]):
            raise ValueError("configuration/control comparisons have different matched anchors")
    return {"schema_version": 1, "curves": curves, "configurations": rows, "controls": controls,
            "depth": depth, "selected_neural": chosen_tag, "pending": pending,
            "search_completed": len(curves), "search_registered": len(jobs),
            "evidence_status": "2022-2023 previously used development; not independent test",
            "task": "retrospective same-month reconstruction at 20 observed depth levels",
            "spread": "sample standard deviation across seeds, ddof=1; not a confidence interval",
            "depth_statistic": "J(three-seed ensemble mean) minus J(matched anchor) at each observed depth; no interpolation experiment"}


def source_record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load_inputs(comparison: Path, manifest_path: Path, selection_path: Path,
                ablation_path: Path, cache_manifest: Path, output_root: Path) -> dict:
    report, manifest = read_json(comparison), read_json(manifest_path)
    selection = read_json(selection_path) if selection_path.exists() else None
    ablations = read_json(ablation_path) if ablation_path.exists() else None
    cache = read_json(cache_manifest)
    artifact = cache["arrays"]["cohort.levels"]
    levels_path = cache_manifest.parent / artifact["path"]
    level_source = source_record(levels_path)
    if level_source["sha256"] != artifact["sha256"]:
        raise ValueError("cached depth coordinates fail their source fingerprint")
    levels = np.load(levels_path, allow_pickle=False)
    # The embedded comparison must describe the actual completed summaries.
    summaries = []
    seen = set()
    for item in report["search"] + report.get("ablations", []):
        summary = item["summary"]
        path = output_root / summary["tag"] / f"summary_seed{summary['seed']}.json"
        if not path.exists() or read_json(path) != summary:
            raise ValueError(f"comparison is stale or lacks a completed source summary: {path}; regenerate 71")
        summaries.append(source_record(path))
        seen.add(path)
    selected_jobs = {prefix(job["tag"]): job for job in selection["selected"]} if selection else {}
    for group in report["groups"]:
        if group["configuration"] not in selected_jobs:
            raise ValueError("unselected configuration in the comparison")
        for seed_key, details in group["development_details"].items():
            seed = int(seed_key.removeprefix("seed"))
            job = {**selected_jobs[group["configuration"]], "tag": f"{group['configuration']}_s{seed}", "seed": seed}
            path = output_root / job["tag"] / f"summary_seed{seed}.json"
            if not path.exists():
                raise ValueError(f"development result lacks a completed source summary: {path}")
            summary = read_json(path)
            R.match_job(job, summary)
            if "development" not in summary:
                raise ValueError(f"development evaluation is unfinished: {path}")
            for channel in CHANNELS:
                for key in ("scores", "baseline_scores"):
                    if not np.isclose(summary["development"][key][channel]["J"], details[key][channel]["J"], rtol=1e-7, atol=1e-9):
                        raise ValueError(f"comparison development values are stale: {path}; regenerate 71")
            if path not in seen:
                summaries.append(source_record(path))
                seen.add(path)
    data = build_figure_data(report, manifest, selection, ablations, levels)
    data["sources"] = {"comparison": source_record(comparison), "manifest": source_record(manifest_path),
                       "selection": source_record(selection_path) if selection else None,
                       "ablations": source_record(ablation_path) if ablations else None,
                       "levels": level_source, "completed_summaries": summaries}
    return data


def draw_ratio_panel(ax, rows: list[dict], title: str) -> None:
    ax.set_title(title, loc="left", pad=12)
    ax.axvline(1., color="#858A8F", linewidth=.9, linestyle=(0, (3, 3)), zorder=1)
    labels = []
    if not rows:
        ax.text(.5, .5, "Pending frozen development evaluation", transform=ax.transAxes,
                ha="center", va="center", color="#73777B", fontsize=9)
        ax.set_yticks([])
    for i, row in enumerate(rows):
        labels.append(row["label"])
        if row["data"] is None:
            ax.text(1., i, "  pending", color="#73777B", va="center", fontsize=8)
            continue
        for channel, dy, marker in (("TEMP", -.12, "o"), ("SALT", .12, "s")):
            stats = row["data"]["channels"][channel]
            ax.errorbar(stats["ratio_mean"], i + dy, xerr=stats["ratio_std"],
                        fmt=marker, color=CHANNEL_COLORS[channel], markerfacecolor="white",
                        markersize=5, markeredgewidth=1.2, elinewidth=1.1, capsize=3,
                        label=channel if i == 0 else None, zorder=3)
    if rows:
        ax.set_yticks(range(len(rows)), labels)
        ax.set_ylim(len(rows) - .55, -.6)
    ax.set_xlabel("Standardized RMSE / anchor RMSE  (lower is better)")
    ax.grid(axis="x", color="#DEE0E2", linewidth=.6, zorder=0)
    # Include the matched reference even if every model improves substantially.
    lo, hi = ax.get_xlim()
    ax.set_xlim(min(lo, .995), max(hi, 1.005))


def render_figure(data: dict, output_prefix: Path) -> list[Path]:
    # The workspace's home configuration directory may be read-only.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/ocean-latent-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    style = {"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
             "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
             "axes.spines.top": False, "axes.spines.right": False,
             "axes.linewidth": .7, "xtick.major.width": .7, "ytick.major.width": .7,
             "svg.fonttype": "none", "savefig.facecolor": "white"}
    with plt.rc_context(style):
        fig, axs = plt.subplots(2, 2, figsize=(11.6, 8.8))
        fig.subplots_adjust(left=.135, right=.975, bottom=.135, top=.865, wspace=.55, hspace=.54)
        fig.text(.09, .947, "Ocean multimodal shared-state reconstruction", fontsize=16,
                 fontfamily="DejaVu Serif", color="#272D32")
        subtitle = "2016–2020 training  ·  2021 validation selection  ·  2022–2023 development"
        if data["pending"]:
            subtitle += "  ·  INCOMPLETE"
        fig.text(.09, .913, subtitle, fontsize=9, color="#61666B")
        ax = axs[0, 0]
        ax.set_title("a  Measured validation learning curves", loc="left", pad=12)
        for curve in data["curves"]:
            ax.plot(curve["steps"], curve["macro_z"], color=COLORS[curve["variant"]],
                    linestyle="--" if curve["large"] else "-", linewidth=1.2,
                    marker=".", markersize=3.2, label=curve["label"])
            ax.plot(curve["best_step"], curve["best_macro_z"], marker="s", markersize=4.4,
                    markerfacecolor="white", markeredgecolor=COLORS[curve["variant"]], zorder=4)
        ax.set_xlabel("Optimization step (including measured step 0)")
        ax.set_ylabel("Mean T/S standardized RMSE ↓")
        ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
        ax.grid(axis="y", color="#DEE0E2", linewidth=.6)
        if data["curves"]:
            ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper center", handlelength=2.1,
                      bbox_to_anchor=(.5, -.16), columnspacing=1., labelspacing=.45)
        if data["search_completed"] < data["search_registered"]:
            ax.text(.02, .98, f"{data['search_completed']}/{data['search_registered']} registered runs complete",
                    transform=ax.transAxes, va="top", color="#73777B", fontsize=8)
        draw_ratio_panel(axs[0, 1], data["configurations"], "b  Validation-frozen configurations · 3 seeds")
        control_variant = next((variant for variant in LABELS
                                if data["selected_neural"] and data["selected_neural"].startswith(variant + "_")), None)
        control_label = LABELS.get(control_variant, "Control backbone")
        draw_ratio_panel(axs[1, 0], data["controls"], f"c  {control_label} · fixed-OI controls")
        ax = axs[1, 1]
        ax.set_title(f"d  {control_label} · ensemble depth error", loc="left", pad=12)
        ax.axvline(0., color="#858A8F", linewidth=.9, linestyle=(0, (3, 3)))
        depth = data["depth"]
        if depth:
            for channel, marker in (("TEMP", "o"), ("SALT", "s")):
                ax.plot(depth["channels"][channel]["delta_J"], depth["depth_m"],
                        color=CHANNEL_COLORS[channel], marker=marker, markersize=3.5,
                        markerfacecolor="white", markeredgewidth=.9, linewidth=1.1)
            ax.text(0., 1.012, depth["label"] + " · three-seed ensemble",
                    transform=ax.transAxes, fontsize=7.2, color="#61666B", va="bottom")
        else:
            ax.text(.5, .5, "Pending selected-neural ensemble", transform=ax.transAxes,
                    ha="center", va="center", color="#73777B", fontsize=9)
        ax.set_ylim(1030., -15.)
        ax.set_ylabel("Observed depth (m)")
        ax.set_xlabel("Δ climatology-normalized RMSE")
        ax.grid(axis="y", color="#DEE0E2", linewidth=.6)
        handles = [Line2D([], [], marker=marker, linestyle="none", color=CHANNEL_COLORS[ch],
                          markerfacecolor="white", markersize=5, label=ch)
                   for ch, marker in (("TEMP", "o"), ("SALT", "s"))]
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.977, .965),
                   frameon=False, ncol=2, fontsize=8, handletextpad=.3, columnspacing=1.2)
        footer = ("Development years were used previously; these are retrospective reconstruction results, not an independent test.\n"
                  "b,c: standardized RMSE ratios, mean ± sample SD across three seeds.  d: ensemble error at 20 measured depth levels.\n"
                  "a: recorded evaluations only; squares mark validation-selected checkpoints. No arbitrary-depth/OOD claim.")
        if data["pending"]:
            footer += f"\nINCOMPLETE: {len(data['pending'])} registered result groups are pending; missing values are omitted."
        fig.text(.09, .052 if data["pending"] else .065, footer, fontsize=7.4,
                 color="#61666B", linespacing=1.55, va="center")
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        paths = [output_prefix.with_suffix(ext) for ext in (".png", ".svg")]
        fig.savefig(paths[0], dpi=300)
        fig.savefig(paths[1], metadata={"Title": "Measured ocean latent campaign", "Description": data["evidence_status"]})
        plt.close(fig)
    return paths


def caption(data: dict) -> str:
    selected = prefix(data["selected_neural"]) if data["selected_neural"] else "pending validation-only selection"
    return ("Ocean multimodal shared-state reconstruction. (a) Recorded 2021 validation macro_z evaluations, including "
            "the measured step-0 anchor; square markers identify validation-selected checkpoints. "
            "(b) Frozen architecture configurations: three-seed mean and sample standard deviation (ddof=1) of "
            "standardized RMSE divided by matched anchor RMSE, calculated separately for temperature and salinity. "
            f"(c) Component-control backbone {selected}, chosen by single-seed validation before replication, "
            "with the original OI parameters fixed: full, shared-latent "
            "removed, and added local correction removed. Removing the added local path retains original OI. "
            f"(d) Three-seed ensemble of {selected}, reporting ensemble minus anchor climatology-normalized RMSE at each of the "
            "20 actual depth levels. The ensemble uses matched scoring identities and the total-variance rule from report 71; "
            "its mean determines the plotted error. Lower ratios and negative RMSE differences indicate lower error. "
            "The backbone in panels c,d is the component-control choice; the final three-seed ensemble "
            "recommendation is selected separately by validation and may differ. "
            "2022–2023 were previously used for development, so these results do not establish independent test "
            "performance, arbitrary-depth reconstruction, out-of-distribution generalization, or forecasting. "
            + (f"The figure is incomplete; {len(data['pending'])} registered result groups are pending. " if data["pending"] else "")
            + "The accompanying data JSON records all plotted numbers, source paths, and fingerprints.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=OUTPUT / "comparison_20261007.json")
    parser.add_argument("--manifest", type=Path, default=OUTPUT / "campaign_20261007.json")
    parser.add_argument("--selection", type=Path, default=OUTPUT / "selected_20261007.json")
    parser.add_argument("--ablations", type=Path, default=OUTPUT / "ablations_20261007.json")
    parser.add_argument("--cache-manifest", type=Path, default=OUTPUT / "prepared_scenes/anc_satday_dfs/manifest.json")
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "reports/real_data/fig_latent_campaign_20261007")
    parser.add_argument("--allow-pending", action="store_true", help="render a clearly labelled incomplete progress figure")
    args = parser.parse_args()
    data = load_inputs(args.comparison, args.manifest, args.selection, args.ablations, args.cache_manifest, args.output_root)
    if data["pending"] and not args.allow_pending:
        raise SystemExit("refusing to render a final figure with unfinished registered results: " + "; ".join(data["pending"]))
    paths = render_figure(data, args.output_prefix)
    data_path = args.output_prefix.with_suffix(".data.json")
    data_path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    caption_path = args.output_prefix.with_suffix(".caption.txt")
    caption_path.write_text(caption(data))
    print(json.dumps({"figures": [str(path) for path in paths], "data": str(data_path),
                      "caption": str(caption_path), "pending": data["pending"]}))


if __name__ == "__main__":
    main()
