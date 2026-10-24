"""Report of the size study (spec ``docs/superpowers/specs/2026-10-23-ka-size-study-design.md``).

Reads every finished ``summary_seed*.json`` (200,000 steps) under ``outputs/size_20261023`` and writes
``reports/synthetic/size_20261023.md`` and ``.json``:

  sizes     parameters, hours a run, best steps, and the score -- the mean over leads of the
            per-variable standardised RMSE of the chosen checkpoint on every cell -- on 2004, which
            chose the checkpoints, and on 2005, held out of that choice; mean ± sd over seeds
  decision  "bigger helps" if x1.75's 2005 score is below x1's by more than the larger of their seed
            standard deviations; the best size is the lowest 2004 score (choosing on 2005 would end
            its being held out); computed once every size has all three seeds
  by lead   every size's 2005 RMSE at leads 0, 1, 3 and 6 months, temperature (°C) and salinity (PSU),
            beside persisted OI and climatology (a zero anomaly); the best size's rows are the forecast

The convergence curves are ``171_size_curves.py``'s figure, linked here. Runs still training are left
out. Run it at any time.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/size_20261023"
REPORT = ROOT / "reports/synthetic/size_20261023"
STEPS, SEEDS = 200000, 3
SELECT, HELD_OUT = "2004", "2005"
CH = ("TEMP", "SALT")
UNITS = {"TEMP": "°C", "SALT": "PSU"}


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments/synthetic" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


forecast_report = _script("96_forecast_report")       # climatology, mean_sd
size_curves = _script("171_size_curves")              # SIZES, FIGURE
SIZES = size_curves.SIZES


def runs(output=OUTPUT):
    found = {}
    for path in sorted(output.glob("y100_x*_s*/summary_seed*.json")):
        arm = path.parent.name.rsplit("_s", 1)[0]
        summary = json.loads(path.read_text())
        if arm in SIZES and summary.get("completed_steps") == STEPS and "forecast" in summary:
            summary["arrays"] = path.parent / f"forecast_seed{summary['seed']}.npz"
            found.setdefault(arm, []).append(summary)
    return {arm: found[arm] for arm in SIZES if arm in found}


def score(summary, year):
    """The chosen checkpoint's mean over leads of the per-variable standardised RMSE on every cell of ``year``."""
    by_lead = summary["forecast"]["best"]["by_lead"]
    return float(np.mean([v["by_year"][year]["scores"]["macro_z"] for v in by_lead.values()]))


def decide(scores):
    """``scores`` {arm: {year: [one per seed]}} -> the spec's decision."""
    m = {arm: {y: forecast_report.mean_sd(v) for y, v in s.items()} for arm, s in scores.items()}
    small, large = m["y100_x1"][HELD_OUT], m["y100_x175"][HELD_OUT]
    spread = max(small[1] or 0., large[1] or 0.)
    return {"bigger_helps": bool(large[0] < small[0] - spread), "gain": small[0] - large[0], "seed_spread": spread,
            "best": min(m, key=lambda arm: m[arm][SELECT][0])}


def by_lead(summaries, year=HELD_OUT):
    leads = sorted(summaries[0]["forecast"]["best"]["by_lead"], key=int)
    clim = forecast_report.climatology(summaries[0]["arrays"])      # every run is scored on the same cells
    first = summaries[0]["forecast"]["best"]["by_lead"]
    return {lead: {**{ch: forecast_report.mean_sd([s["forecast"]["best"]["by_lead"][lead]["by_year"][year]["scores"][ch]["rmse_physical"]
                                                   for s in summaries]) for ch in CH},
                   **{f"oi_{ch}": first[lead]["by_year"][year]["baseline_scores"][ch]["rmse_physical"] for ch in CH},
                   **{f"clim_{ch}": clim[int(lead)]["by_year"][year][ch] for ch in CH}} for lead in leads}


def text(value, digits=4):
    mean, sd = value
    return f"{mean:.{digits}f}" + ("" if sd is None else f" ± {sd:.{digits}f}")


def markdown(found, scores, decision, leads):
    out = ["# KA-Ocean model size on 100 CESM2 years", "",
           "KA-Ocean (method A: U-Net state, lead evolution, `ka_flow` kriging read-out) on `cesm2_vlong_uniform`, "
           "training window 1904-2003, 200,000 steps of 1,024 queries, leads 0, 1, 3 and 6 months in one model. "
           "Each run's checkpoint is chosen on target year 2004; 2005 is held out of that choice. "
           "Spec: `docs/superpowers/specs/2026-10-23-ka-size-study-design.md`.", "",
           "Seeds finished: " + ", ".join(f"{SIZES[a][0]} {len(found.get(a, []))}" for a in SIZES) + ".", "",
           "## Sizes", "", "| size | parameters | hours a run | best steps | 2004 score (chose) | 2005 score (held out) |",
           "|---|---|---|---|---|---|"]
    for arm, summaries in found.items():
        hours = forecast_report.mean_sd([s["runtime_s"] / 3600 for s in summaries])
        out.append(f"| {SIZES[arm][0]} | {summaries[0]['params'] / 1e6:.1f}M | {text(hours, 1)} | "
                   f"{', '.join(str(s['best_step']) for s in summaries)} | {text(forecast_report.mean_sd(scores[arm][SELECT]))} | "
                   f"{text(forecast_report.mean_sd(scores[arm][HELD_OUT]))} |")
    out += ["", "Score: mean over leads of the per-variable standardised RMSE on every cell, mean ± sd over seeds.", "", "## Decision", ""]
    if decision is None:
        out.append("Waits until every size has all three seeds.")
    else:
        out += [f"* Bigger helps: **{'yes' if decision['bigger_helps'] else 'no'}** -- x1.75's 2005 score is "
                f"{decision['gain']:+.4f} below x1's; the larger seed sd is {decision['seed_spread']:.4f}.",
                f"* Best size (lowest 2004 score): **{SIZES[decision['best']][0]}**."]
    for ch in (CH if leads else ()):                  # nothing finished yet: no lead tables
        out += ["", f"## 2005 by lead: {'temperature' if ch == 'TEMP' else 'salinity'} RMSE ({UNITS[ch]})", "",
                "| lead | persisted OI | climatology | " + " | ".join(SIZES[a][0] for a in leads) + " |",
                "|---|---|---|" + "---|" * len(leads)]
        first = next(iter(leads.values()))
        for lead in first:
            cells = [text(leads[a][lead][ch]) + (" **best**" if decision and a == decision["best"] else "") for a in leads]
            out.append(f"| {lead} | {first[lead][f'oi_{ch}']:.4f} | {first[lead][f'clim_{ch}']:.4f} | " + " | ".join(cells) + " |")
    out += ["", "## Convergence", "", f"![training curves]({size_curves.FIGURE.name})", "",
            "`171_size_curves.py`: training MSE and the 2004 validation score per run; a best step after 180k means still improving."]
    return "\n".join(out) + "\n"


def main():
    found = runs()
    scores = {arm: {y: [score(s, y) for s in summaries] for y in (SELECT, HELD_OUT)} for arm, summaries in found.items()}
    complete = len(found) == len(SIZES) and all(len(v) == SEEDS for v in found.values())
    decision = decide(scores) if complete else None
    leads = {arm: by_lead(summaries) for arm, summaries in found.items()}
    REPORT.with_suffix(".md").write_text(markdown(found, scores, decision, leads))
    REPORT.with_suffix(".json").write_text(json.dumps({"scores": scores, "decision": decision, "by_lead": leads,
                                                       "seeds": {a: [s["seed"] for s in v] for a, v in found.items()}}, indent=2) + "\n")
    print(REPORT.with_suffix(".md").relative_to(ROOT))


if __name__ == "__main__":
    main()
