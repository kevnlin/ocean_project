"""Report of the gridded model: the CESM2 read-out ablation at 70k steps and the real-data runs.

Writes ``reports/synthetic/grid_ka_20261019.{md,json}`` from every finished summary:

  CESM2   ``103`` (ka-pilot worktree: gj_flow, gj_stat, gj_oi, gj_none; ka-dev: gj_softmax) and
          the 70,000-step OI-anchored local transformer of ``95`` (``f_y40``, ``f_y4``), all scored
          on every cell of 2004-2005 with the checkpoint chosen on those years (the reference's
          protocol), mean and standard deviation over the finished seeds, with the persisted OI
          of the same cells
  real    ``107`` (scratch, zeroshot, finetune) on every cell of 2021 (the selection year), the
          persisted real OI of the same cells, and the tuned OI of ``106``

Unfinished runs are listed. Temperature in degC, salinity in PSU.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PILOT = Path("/DATA2/linke/ocean_project_ka/outputs/grid_ka_70k_20261019")
DEV = ROOT / "outputs/grid_ka_70k_20261019"
REFERENCE = Path("/DATA2/linke/ocean_project/outputs/forecast_20261019")
REAL = ROOT / "outputs/real_global_20261019"
REAL_OI = ROOT / "outputs/audit/real/fixed_baselines_real_global/summary.json"
REPORT = ROOT / "reports/synthetic/grid_ka_20261019"
SEEDS = (1234, 1235, 1236)
LEADS = ("0", "1", "3", "6")
CH = ("TEMP", "SALT")
CESM2_ROWS = (("gj_flow", PILOT, "kriging attention, flow kernel (the method)"),
              ("gj_stat", PILOT, "kriging attention, learned stationary kernel"),
              ("gj_oi", PILOT, "kriging attention, frozen OI kernel"),
              ("gj_softmax", DEV, "softmax attention over the same innovations"),
              ("gj_none", PILOT, "no read-out: gridded background alone"),
              ("f_y40", REFERENCE, "reference: OI-anchored local transformer, 40-year window"),
              ("f_y4", REFERENCE, "reference: OI-anchored local transformer, 4-year window"))
REAL_ROWS = (("scratch", "trained on real 2016-2020 only"),
             ("zeroshot", "the CESM2 model, no update"),
             ("finetune", "the CESM2 model fine-tuned on real 2016-2020"))


def summary(folder, tag, seed):
    path = folder / f"{tag}_s{seed}" / f"summary_seed{seed}.json"
    return json.loads(path.read_text()) if path.exists() else None


def by_lead(s, key="scores"):
    lead = s["forecast"]["best"]["by_lead"]
    return {l: {ch: lead[l][key][ch]["rmse_physical"] for ch in CH} for l in LEADS if l in lead}


def collect(folder, tag):
    runs = {seed: summary(folder, tag, seed) for seed in SEEDS}
    done = {seed: s for seed, s in runs.items() if s is not None}
    if not done:
        return None
    model = [by_lead(s) for s in done.values()]
    oi = [by_lead(s, "baseline_scores") for s in done.values()]
    stat = lambda rows: {l: {ch: (float(np.mean([r[l][ch] for r in rows])), float(np.std([r[l][ch] for r in rows])))
                             for ch in CH} for l in rows[0]}
    return {"seeds": sorted(done), "missing": [seed for seed in SEEDS if seed not in done],
            "model": stat(model), "persisted_oi": stat(oi), "best_steps": [s["best_step"] for s in done.values()],
            "hours": [round(s["runtime_s"] / 3600, 1) for s in done.values()]}


def cell(value):
    (t, ts), (s, ss) = value["TEMP"], value["SALT"]
    return f"{t:.4f} ± {ts:.4f} / {s:.4f} ± {ss:.4f}"


def change(a, b):
    return " / ".join(f"{100 * (a[ch][0] - b[ch][0]) / b[ch][0]:+.1f} %" for ch in CH)


def table(rows, title, note):
    lines = [f"## {title}", "", note, ""]
    header = "| run | seeds | " + " | ".join(f"lead {l}" for l in LEADS) + " |"
    lines += [header, "|" + "---|" * (len(LEADS) + 2)]
    oi = next((r["persisted_oi"] for _, r in rows.values() if r), None)
    for name, (description, r) in rows.items():
        if r is None:
            lines.append(f"| {name}: {description} | running | " + " | ".join("–" for _ in LEADS) + " |")
            continue
        cells = [cell(r["model"][l]) + (f"<br>vs persisted OI {change(r['model'][l], r['persisted_oi'][l])}")
                 if l in r["model"] else "–" for l in LEADS]
        lines.append(f"| {name}: {description} | {len(r['seeds'])} | " + " | ".join(cells) + " |")
    if oi:
        lines.append("| persisted OI (same cells) | – | " + " | ".join(cell(oi[l]) if l in oi else "–" for l in LEADS) + " |")
    return lines


def main():
    cesm2 = {name: (description, collect(folder, name)) for name, folder, description in CESM2_ROWS}
    real = {name: (description, collect(REAL, name)) for name, description in REAL_ROWS}
    lines = ["# Gridded state model with kriging attention", "",
             "Temperature °C / salinity PSU, RMSE on every scored cell, mean ± standard deviation over seeds.", ""]
    lines += table({k: v for k, v in cesm2.items()}, "CESM2: read-out ablation at 70,000 steps",
                   "Leads 0/1/3/6 months, 40-year training window 1964-2003 (f_y4: 2000-2003), checkpoint chosen "
                   "on 2004-2005 and scored on every cell of 2004-2005 (the reference's protocol; these years are "
                   "the selection years, not a sealed test).")
    lines += [""]
    lines += table({k: v for k, v in real.items()}, "Real Argo + satellites (2021)",
                   "Inputs of month t, held-out floats of month t + lead; checkpoint chosen on 2021 and scored on "
                   "every cell of 2021. 2022-2023 is scored once at the end of the study.")
    if REAL_OI.exists():
        oi = json.loads(REAL_OI.read_text())["baselines"]
        lines += ["", "Tuned OI of 106 (lead 0): " + "; ".join(
            f"{split} {oi['oi']['scores'][split]['TEMP']['rmse_physical']:.4f} / "
            f"{oi['oi']['scores'][split]['SALT']['rmse_physical']:.4f} (climatology "
            f"{oi['climatology']['scores'][split]['TEMP']['rmse_physical']:.4f} / "
            f"{oi['climatology']['scores'][split]['SALT']['rmse_physical']:.4f})"
            for split in ("validation", "development")) + "."]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    REPORT.with_suffix(".json").write_text(json.dumps({"cesm2": {k: v[1] for k, v in cesm2.items()},
                                                       "real": {k: v[1] for k, v in real.items()}}, indent=1) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
