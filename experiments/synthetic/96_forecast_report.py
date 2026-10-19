"""Tables of the forecast runs (``95_forecast_campaign.py``): RMSE by lead, temperature and salinity.

Reads every finished ``summary_seed*.json`` under ``outputs/forecast_20261019`` and writes
``reports/synthetic/forecast_20261019.md`` and ``.json``:

  by lead     RMSE on every cell of the 2004-2005 target months at leads 0, 1, 3 and 6 months,
              temperature (degC) and salinity (PSU) separately, mean and standard deviation over
              seeds, for the checkpoint chosen on those years and for the model of the last step
              (no choice involved), against persisted OI on the same cells
  by year     the same for 2004 and 2005 apart
  curves      the 2004-2005 score of every lead during training

Runs still training are left out. Run it at any time.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/forecast_20261019"
REPORT = ROOT / "reports/synthetic/forecast_20261019"
CH = ("TEMP", "SALT")
UNITS = {"TEMP": "°C", "SALT": "PSU"}
NAME = re.compile(r"^f_y(\d+)_s(\d+)$")


def runs(output=OUTPUT):
    found = {}
    for path in sorted(output.glob("f_y*_s*/summary_seed*.json")):
        match = NAME.match(path.parent.name)
        summary = json.loads(path.read_text())
        if match and "forecast" in summary:
            found.setdefault(int(match[1]), []).append(summary)
    return found


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), (float(values.std(ddof=1)) if len(values) > 1 else None)


def collect(found, which="best"):
    """{years: {lead: {ch: (mean, sd), "oi_<ch>": value, "by_year": {...}}}} for one checkpoint."""
    table = {}
    for years, summaries in sorted(found.items()):
        usable = [s for s in summaries if s["forecast"].get(which) is not None]
        if not usable:
            continue
        leads = sorted(usable[0]["forecast"][which]["by_lead"], key=int)
        table[years] = {"seeds": [s["seed"] for s in usable]}
        for lead in leads:
            parts = [s["forecast"][which]["by_lead"][lead] for s in usable]
            row = {ch: mean_sd([p["scores"][ch]["rmse_physical"] for p in parts]) for ch in CH}
            row.update({f"oi_{ch}": parts[0]["baseline_scores"][ch]["rmse_physical"] for ch in CH})
            row["by_year"] = {y: {**{ch: mean_sd([p["by_year"][y]["scores"][ch]["rmse_physical"] for p in parts]) for ch in CH},
                                  **{f"oi_{ch}": parts[0]["by_year"][y]["baseline_scores"][ch]["rmse_physical"] for ch in CH}}
                              for y in sorted(parts[0]["by_year"])}
            table[years][int(lead)] = row
        table[years]["best_step"] = [s["best_step"] for s in usable]
        table[years]["hours"] = mean_sd([s["runtime_s"] / 3600 for s in usable])
    return table


def fmt(stat, digits=4):
    if stat is None:
        return "–"
    mean, sd = stat
    return f"{mean:.{digits}f}" + (f" ± {sd:.{digits}f}" if sd is not None else "")


def change(new, old):
    return f"{100 * (new / old - 1):+.1f} %"


def lead_table(table, ch, title):
    windows = sorted(table)
    leads = sorted({k for t in table.values() for k in t if isinstance(k, int)})
    lines = [f"**{title}**", "",
             "| Lead | Persisted OI | " + " | ".join(f"{y} years" for y in windows) + " | "
             + " | ".join(f"{y} years vs OI" for y in windows) + " |",
             "|---|---|" + "---|" * (2 * len(windows))]
    for lead in leads:
        oi = next(table[y][lead][f"oi_{ch}"] for y in windows if lead in table[y])
        cells = [fmt(table[y][lead][ch]) if lead in table[y] else "–" for y in windows]
        gains = [change(table[y][lead][ch][0], oi) if lead in table[y] else "–" for y in windows]
        lines.append(f"| {lead} month{'s' * (lead != 1)} | {oi:.4f} | " + " | ".join(cells) + " | " + " | ".join(gains) + " |")
    return lines


def markdown(best, last):
    lines = ["# Forecast skill by lead, CESM2, Argo + surface inputs", "",
             "Zihao's shared T/S alignment model at the medium size (128 latent slots, width 256), Argo plus SST, SSS "
             "and steric SSH, the local OI-error scale by depth × 5° cell, 70,000 steps of 1,024 queries. One model "
             "for every lead: the inputs are the floats and satellite fields of month t, the target is the T/S field "
             "at the query profiles of month t + lead. Lead 0 is reconstruction. The training windows end in 2003 "
             "(4 years: 2000–2003, 40 years: 1964–2003); the normalisation, climatology, OI anchor and scale table "
             "are the 2000–2003 fits for both. The baseline at every lead is persisted OI: the frozen OI of month t "
             "used as the field of month t + lead.", "",
             "RMSE on every query cell of the 2004–2005 target months, mean ± standard deviation over seeds.", ""]
    if not best:
        return "\n".join(lines + ["No run has finished yet."]) + "\n"
    seeds = {y: len(t["seeds"]) for y, t in best.items()}
    lines += [f"Seeds finished: " + ", ".join(f"{y} years {n}" for y, n in seeds.items()) + ".", "",
              "## Checkpoint chosen on 2004–2005", ""]
    for ch in CH:
        lines += lead_table(best, ch, f"{'Temperature' if ch == 'TEMP' else 'Salinity'} RMSE ({UNITS[ch]})") + [""]
    if last:
        lines += ["## Model of the last step", "",
                  "No checkpoint choice: the scores the reported years did not select.", ""]
        for ch in CH:
            lines += lead_table(last, ch, f"{'Temperature' if ch == 'TEMP' else 'Salinity'} RMSE ({UNITS[ch]})") + [""]
    lines += ["## By target year", "", "Chosen checkpoint, temperature / salinity.", "",
              "| Window | Lead | 2004 | OI 2004 | 2005 | OI 2005 |", "|---|---|---|---|---|---|"]
    for years, t in sorted(best.items()):
        for lead in sorted(k for k in t if isinstance(k, int)):
            row = t[lead]["by_year"]
            cell = lambda y: f"{fmt(row[y]['TEMP'])} / {fmt(row[y]['SALT'])}" if y in row else "–"
            oi = lambda y: f"{row[y]['oi_TEMP']:.4f} / {row[y]['oi_SALT']:.4f}" if y in row else "–"
            lines.append(f"| {years} years | {lead} | {cell('2004')} | {oi('2004')} | {cell('2005')} | {oi('2005')} |")
    lines += ["", "## Runs", "", "| Window | best steps | hours a run |", "|---|---|---|"]
    for years, t in sorted(best.items()):
        lines.append(f"| {years} years | {', '.join(map(str, t['best_step']))} | {fmt(t['hours'], 1)} |")
    return "\n".join(lines) + "\n"


def main():
    found = runs()
    best, last = collect(found, "best"), collect(found, "last")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.with_suffix(".md").write_text(markdown(best, last))
    REPORT.with_suffix(".json").write_text(json.dumps(
        {"best": {str(y): {str(k): v for k, v in t.items()} for y, t in best.items()},
         "last": {str(y): {str(k): v for k, v in t.items()} for y, t in last.items()}}, indent=2) + "\n")
    print(f"{sum(len(v) for v in found.values())} finished runs -> {REPORT}.md", flush=True)


if __name__ == "__main__":
    main()
