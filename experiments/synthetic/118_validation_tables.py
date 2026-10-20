"""The paper's tables on the selection years (CESM2 2004-2005, real 2021), from every finished run.

Validation scores only: the sealed tests (CESM2 2006-2007, real 2022-2023) are scored once by 109 and
113 after the method is frozen. For every row: RMSE at leads 0, 1, 3, 6 (temperature degC / salinity
PSU, mean ± sd over the finished seeds, every scored cell), the selection score (mean standardized
RMSE over leads, the 8000-cell draw each run chose its checkpoint by), and from the calibrations of
``116`` (fitted on the same years, so in-sample) the temperature CRPS and 95 % coverage at leads 0
and 6 before -> after calibration. Rows still running show their finished seeds.

  .venv/bin/python experiments/synthetic/118_validation_tables.py   # writes reports/paper/validation_tables_20261020.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/paper/validation_tables_20261020.md"
CALIBRATIONS = ROOT / "outputs/calibration_20261020"
SEEDS = (1234, 1235, 1236)
LEADS = ("0", "1", "3", "6")
PILOT = Path("/DATA2/linke/ocean_project_ka/outputs/grid_ka_70k_20261019")
DEV = Path("/DATA2/linke/ocean_project_ka_dev/outputs")
FORMAL, REAL = DEV / "formal_20261019", DEV / "real_global_20261019"
POSEIDON = Path("/DATA2/linke/ocean_project_ka_dev2/outputs/poseidon_20261019")
REFERENCE = Path("/DATA2/linke/ocean_project/outputs/forecast_20261019")
SELECTION = ROOT / "outputs/selection_20261020"

CESM2 = {
    "Candidates (method selection, 115)": [
        ("A: U-Net + KA, joint background", PILOT, "gj_flow"),
        ("B: U-Net + KA, prior background", FORMAL, "c_gp_flow"),
        ("C: Poseidon-T + KA", POSEIDON, "p_ka")],
    "Baselines": [
        ("Anchored transformer (OI anchor + local refinement)", REFERENCE, "f_y40"),
        ("ARMOR3D-style (prior background + OI kernel)", FORMAL, "c_gp_oi"),
        ("Aardvark-style (no exact-value read-out)", PILOT, "gj_none"),
        ("Poseidon-T fine-tuned (no read-out)", POSEIDON, "p_none")],
    "Ablations": [
        ("A with softmax read-out", DEV / "grid_ka_70k_20261019", "gj_softmax"),
        ("A with the OI kernel", PILOT, "gj_oi"),
        ("A with a stationary kernel", PILOT, "gj_stat"),
        ("A without evolution", FORMAL, "c_noevo"),
        ("A without lead decay", FORMAL, "c_nodecay"),
        ("A on a 4-year window", FORMAL, "c_y4"),
        ("A with noise pretraining", FORMAL, "c_noise"),
        ("Anchored transformer, 4-year window", REFERENCE, "f_y4"),
        ("C with random Poseidon-T weights", SELECTION, "pr_random")]}
REAL_ROWS = {
    "Candidates (method selection, 115)": [
        ("A: U-Net + KA, joint background", REAL, "scratch"),
        ("B: U-Net + KA, prior background", SELECTION, "rb_prior"),
        ("C: Poseidon-T + KA", SELECTION, "rc_poseidon")],
    "Baselines": [
        ("ARMOR3D-style (prior background + OI kernel)", FORMAL, "r_gp_oi"),
        ("Aardvark-style (no exact-value read-out)", FORMAL, "r_none")],
    "Simulation pretraining (A)": [
        ("CESM2-pretrained, fine-tuned 5 years", REAL, "finetune"),
        ("noise-pretrained, fine-tuned 5 years", FORMAL, "r_finetune_noise"),
        ("CESM2-pretrained, zero shot", REAL, "zeroshot"),
        ("noise-pretrained, zero shot", FORMAL, "r_zeroshot_noise")],
    "Real-data budget (A)": [
        ("from scratch, 2 years", FORMAL, "r_scratch_y2"),
        ("CESM2-pretrained, fine-tuned 2 years", FORMAL, "r_finetune_y2"),
        ("from scratch, 1 year", FORMAL, "r_scratch_y1"),
        ("CESM2-pretrained, fine-tuned 1 year", FORMAL, "r_finetune_y1"),
        ("noise-pretrained, fine-tuned 1 year", FORMAL, "r_finetune_noise_y1")]}


def finished(folder, tag):
    out = []
    for seed in SEEDS:
        run = folder / f"{tag}_s{seed}"
        path = run / f"summary_seed{seed}.json"
        if path.exists():
            out.append((run, json.loads(path.read_text())))
    return out


def calibration(run):
    path = CALIBRATIONS / f"{run.resolve().parent.name}__{run.resolve().name}.json"
    return json.loads(path.read_text())["in_sample"] if path.exists() else None


def cells(summaries, lead, ch):
    values = np.array([s["forecast"]["best"]["by_lead"][lead]["scores"][ch]["rmse_physical"] for _, s in summaries])
    digits = 4 if ch == "TEMP" else 4
    spread = f" ± {values.std(ddof=1):.{digits}f}" if len(values) > 1 else ""
    return f"{values.mean():.{digits}f}{spread}"


def row(name, folder, tag):
    summaries = finished(folder, tag)
    if not summaries:
        return f"| {name} | running (0/3) |" + " |" * 6, None
    leads = " | ".join(f"{cells(summaries, l, 'TEMP')} / {cells(summaries, l, 'SALT')}" for l in LEADS)
    score = np.mean([s["validation"]["scores"]["macro_z"] for _, s in summaries])
    cal = [c for c in (calibration(run) for run, _ in summaries) if c is not None]
    if cal:
        pick = lambda lead, kind, key: np.mean([c[lead]["TEMP"][kind][key] for c in cal])
        crps = " / ".join(f"{pick(l, 'raw', 'crps'):.4f} → {pick(l, 'calibrated', 'crps'):.4f}" for l in ("0", "6"))
        coverage = " / ".join(f"{pick(l, 'raw', 'coverage_95'):.2f} → {pick(l, 'calibrated', 'coverage_95'):.2f}"
                              for l in ("0", "6"))
    else:
        crps = coverage = "–"
    seeds = f" ({len(summaries)}/3)" if len(summaries) < 3 else ""
    return f"| {name}{seeds} | {leads} | {score:.4f} | {crps} | {coverage} |", summaries[0][1]


def table(title, groups):
    lines = [f"## {title}", "",
             "| Row | lead 0 | lead 1 | lead 3 | lead 6 | selection score | T CRPS lead 0 / 6 (raw → calibrated) | "
             "T 95 % coverage lead 0 / 6 |", "|---|---|---|---|---|---|---|---|"]
    reference = None
    for group, rows in groups.items():
        lines.append(f"| **{group}** | | | | | | | |")
        for name, folder, tag in rows:
            text, summary = row(name, folder, tag)
            lines.append(text)
            reference = reference or summary
    if reference is not None:
        oi = " | ".join(" / ".join(f"{reference['forecast']['best']['by_lead'][l]['baseline_scores'][ch]['rmse_physical']:.4f}"
                                   for ch in ("TEMP", "SALT")) for l in LEADS)
        lines.append(f"| Persisted OI (same cells) | {oi} | | | |")
    return lines + [""]


def main():
    lines = ["# Validation tables (selection years only)", "",
             "Generated by `experiments/synthetic/118_validation_tables.py`. RMSE temperature degC / salinity PSU, "
             "mean ± sd over finished seeds (1234-1236), every scored cell of the selection years. Selection score: "
             "mean standardized RMSE over leads on the 8000-cell draw (lower is better). CRPS and coverage: "
             "temperature, in-sample on the same years (`116`). The sealed tests are not opened.", ""]
    lines += table("CESM2 OSSE, 2004-2005 (trained 1964-2003)", CESM2)
    lines += table("Real Argo + OISST/OISSS/DUACS SLA, 2021 (trained 2016-2020)", REAL_ROWS)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
