"""Report of the FuXi comparison and the GLORYS12 transfer
(spec ``docs/superpowers/specs/2026-10-26-fuxi-glorys-comparison-design.md``).

Reads KA-Ocean x1.375's summaries (the size study, ``outputs/size_20261023``), the FuXi-Ocean-style runs'
(``outputs/fuxi_20261026``), ``173``'s transfer results and, for ``g_ka``, ``159``'s main sealed results;
writes ``reports/synthetic/fuxi_transfer_20261026.md`` and ``.json``. Every table names each model with its
parameter count. A section whose inputs are not there yet says what it waits for.

  .venv/bin/python experiments/synthetic/174_fuxi_transfer_report.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/synthetic/fuxi_transfer_20261026"
TRANSFER = ROOT / "outputs/transfer_20261026/results.json"
MAIN = ROOT / "outputs/glorys_20261021/glorys12_test/sealed/results.json"
STEPS, SEEDS, HELD_OUT = 200000, 3, "2005"
CH, LEADS = ("TEMP", "SALT"), ("0", "1", "3", "6")
UNITS = {"TEMP": "°C", "SALT": "PSU"}
VARIABLE = {"TEMP": "temperature", "SALT": "salinity"}
#: CESM2-trained arm -> where its runs are
ARMS = {"y100_x1375": ROOT / "outputs/size_20261023", "y100_fuxi": ROOT / "outputs/fuxi_20261026"}
NAMES = {"y100_x1375": "KA-Ocean", "y100_fuxi": "FuXi-Ocean-style", "g_ka": "g_ka (KA-Ocean, GLORYS-trained)"}


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments/synthetic" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


size_report = _script("170_size_report")          # score, by_lead, text
mean_sd, text = size_report.forecast_report.mean_sd, size_report.text


def runs():
    """{arm: its finished summaries}, each with the path of its saved forecast arrays."""
    found = {}
    for arm, folder in ARMS.items():
        for path in sorted(folder.glob(f"{arm}_s*/summary_seed*.json")):
            summary = json.loads(path.read_text())
            if summary.get("completed_steps") == STEPS and "forecast" in summary:
                summary["arrays"] = path.parent / f"forecast_seed{summary['seed']}.npz"
                found.setdefault(arm, []).append(summary)
    return found


def verdict(ka, fuxi):
    """The spec's rule on per-seed scores: a difference counts only beyond the larger seed sd."""
    (mk, sk), (mf, sf) = mean_sd(ka), mean_sd(fuxi)
    spread = max(sk or 0., sf or 0.)
    word = ("KA better" if mk < mf - spread else "FuXi-style better" if mf < mk - spread
            else "no difference beyond the seed spread")
    return {"verdict": word, "ka": mk, "fuxi": mf, "fuxi_minus_ka": mf - mk, "seed_spread": spread}


def g_ka_params(entries):
    """g_ka's parameter count, from its run's summary beside the registered checkpoint."""
    e = entries[0]
    return json.loads((Path(e["checkpoint"]).parent / f"summary_seed{e['seed']}.json").read_text())["params"]


def transfer(results, main):
    """GLORYS 2019-2020: RMSE by lead (mean, sd over seeds) per arm and variable, persisted OI and the
    climatology, the per-seed scores of the CESM2-trained arms, the parameter counts."""
    arms = {}
    for e in results["entries"]:
        arms.setdefault(e["tag"].rsplit("_s", 1)[0], []).append(e)
    rows = {**arms, "g_ka": [e for e in main["entries"] if e["tag"].startswith("g_ka_")]}
    first = results["entries"][0]["by_lead"]
    table = {lead: {**{f"oi_{ch}": first[lead]["all"][ch]["persisted_oi"]["rmse"] for ch in CH},
                    **{f"clim_{ch}": first[lead]["all"][ch]["climatology"]["rmse"] for ch in CH},
                    **{f"{arm}_{ch}": mean_sd([e["by_lead"][lead]["all"][ch]["model"]["rmse"] for e in entries])
                       for arm, entries in rows.items() for ch in CH}} for lead in LEADS}
    scores = {arm: [float(np.mean([e["macro_z"][lead] for lead in LEADS])) for e in entries] for arm, entries in arms.items()}
    return {"table": table, "scores": scores, "arms": list(rows),
            "params": {**{arm: entries[0]["params"] for arm, entries in arms.items()}, "g_ka": g_ka_params(rows["g_ka"])}}


def verdicts(found, glorys):
    if not (len(found) == len(ARMS) and all(len(s) == SEEDS for s in found.values())):
        return None
    out = {"cesm2_2005": verdict(*([size_report.score(s, HELD_OUT) for s in found[arm]] for arm in ARMS))}
    if glorys is not None:
        out["glorys_2019_2020"] = verdict(glorys["scores"]["y100_x1375"], glorys["scores"]["y100_fuxi"])
    return out


def markdown(found, glorys):
    """The report; ``found`` the finished CESM2 runs by arm, ``glorys`` ``transfer()``'s or None."""
    params = {**({} if glorys is None else glorys["params"]), **{arm: s[0]["params"] for arm, s in found.items()}}
    head = lambda arm: f"{NAMES[arm]} ({params[arm] / 1e6:.1f}M)"
    out = ["# KA-Ocean against a FuXi-Ocean-style model, and both in transfer to GLORYS12", "",
           "KA-Ocean x1.375 (the size study's choice) and a FuXi-Ocean-style model (Huang et al., arXiv:2506.03210, "
           "re-implemented: its code and weights are unreleased), trained alike on `cesm2_vlong_uniform` 1904-2003: "
           "200,000 steps of 1,024 queries, leads 0, 1, 3 and 6 months in one model, the checkpoint chosen on 2004. "
           "Spec: `docs/superpowers/specs/2026-10-26-fuxi-glorys-comparison-design.md`.", "",
           "## Models", "", "| model | parameters | trained on | seeds | hours a run | best steps |", "|---|---|---|---|---|---|"]
    for arm, summaries in found.items():
        hours = mean_sd([s["runtime_s"] / 3600 for s in summaries])
        out.append(f"| {NAMES[arm]} | {params[arm]:,} | 100 CESM2 years, 200k steps | {len(summaries)} | {text(hours, 1)} | "
                   f"{', '.join(str(s['best_step']) for s in summaries)} |")
    if glorys is not None:
        out.append(f"| {NAMES['g_ka']} | {params['g_ka']:,} | 20 GLORYS years, 100k steps | 3 | | |")
    out += ["", "## Verdicts", ""]
    v = verdicts(found, glorys)
    if v is None:
        out.append("Waits until both models have all three seeds: "
                   + ", ".join(f"{NAMES[a]} {len(found.get(a, []))}" for a in ARMS) + ".")
    else:
        for key, title in (("cesm2_2005", "CESM2 2005 (held out)"), ("glorys_2019_2020", "GLORYS12 2019-2020 (transfer)")):
            if key in v:
                out.append(f"* {title}: **{v[key]['verdict']}**; score KA-Ocean {v[key]['ka']:.4f}, FuXi-Ocean-style "
                           f"{v[key]['fuxi']:.4f}, larger seed sd {v[key]['seed_spread']:.4f}.")
    out += ["", "Score: mean over leads of the per-variable standardised RMSE (per-depth z units of the scored cohort), "
            "mean over seeds; a difference counts only beyond the larger seed sd."]
    leads = {arm: size_report.by_lead(s) for arm, s in found.items()}
    for ch in CH:
        out += ["", f"## CESM2 2005 (held out): {VARIABLE[ch]} RMSE ({UNITS[ch]})", ""]
        if not leads:
            out.append("Waits for finished runs.")
            continue
        first = next(iter(leads.values()))
        out += ["| lead | persisted OI | climatology | " + " | ".join(head(a) for a in leads) + " |",
                "|---" * (3 + len(leads)) + "|"]
        out += [f"| {lead} | {first[lead][f'oi_{ch}']:.4f} | {first[lead][f'clim_{ch}']:.4f} | "
                + " | ".join(text(leads[a][lead][ch]) for a in leads) + " |" for lead in LEADS]
    if glorys is None:
        out += ["", "## GLORYS12 2019-2020 (transfer)", "", "Waits for `173_glorys_transfer.py --score`."]
    else:
        order = [a for a in ("g_ka", "y100_x1375", "y100_fuxi") if a in glorys["arms"]]
        for ch in CH:
            out += ["", f"## GLORYS12 2019-2020 (transfer): {VARIABLE[ch]} RMSE ({UNITS[ch]})", "",
                    "| lead | persisted OI | climatology | " + " | ".join(head(a) for a in order) + " |",
                    "|---" * (3 + len(order)) + "|"]
            out += [f"| {lead} | {glorys['table'][lead][f'oi_{ch}']:.4f} | {glorys['table'][lead][f'clim_{ch}']:.4f} | "
                    + " | ".join(text(glorys["table"][lead][f"{a}_{ch}"]) for a in order) + " |" for lead in LEADS]
        transferred = [a for a in ("y100_x1375", "y100_fuxi") if a in glorys["arms"]]
        out += ["", "g_ka trained on 20 GLORYS years (the in-domain reference); KA-Ocean and FuXi-Ocean-style never saw GLORYS.",
                "", "## Transfer cost: GLORYS RMSE as a ratio to g_ka's", "",
                "| lead | " + " | ".join(f"{head(a)} {VARIABLE[ch]}" for a in transferred for ch in CH) + " |",
                "|---" * (1 + 2 * len(transferred)) + "|"]
        out += [f"| {lead} | " + " | ".join(f"{glorys['table'][lead][f'{a}_{ch}'][0] / glorys['table'][lead][f'g_ka_{ch}'][0]:.3f}"
                                           for a in transferred for ch in CH) + " |" for lead in LEADS]
    out += ["", "## Protocol and asymmetries", "",
            "* Equal: data, inputs (one month of profiles and surface grids), steps, queries, objective, optimiser, "
            "selection on 2004, the read-out after the state. FuXi-Ocean-style has no kriging read-out (KA's own).",
            "* Left out of FuXi-Ocean: Mixture-of-Time (one input month), autoregression (one pass per lead), the "
            "Charbonnier loss and its two training stages. Its own size is unpublished; it is matched to KA's.",
            "* KA-Ocean x1.375 won a three-size selection on 2004 (gain 0.0015, 0.2 %); FuXi-Ocean-style has one size.",
            "* GLORYS: a declared extra pass on the sealed 2019-2020 cells; GLORYS's own statistics; g_ka's rows are "
            "the 2026-10-22 main pass's, not re-scored. g_ka is smaller and trained half as long: a reference, not a competitor."]
    return "\n".join(out) + "\n"


def main():
    found = runs()
    glorys = (transfer(json.loads(TRANSFER.read_text()), json.loads(MAIN.read_text())) if TRANSFER.exists() else None)
    REPORT.with_suffix(".md").write_text(markdown(found, glorys))
    cesm2 = {arm: {"scores_2005": [size_report.score(s, HELD_OUT) for s in v], "by_lead": size_report.by_lead(v),
                   "params": v[0]["params"], "seeds": [s["seed"] for s in v]} for arm, v in found.items()}
    REPORT.with_suffix(".json").write_text(json.dumps({"cesm2_2005": cesm2, "glorys_2019_2020": glorys,
                                                       "verdicts": verdicts(found, glorys)}, indent=2) + "\n")
    print(REPORT.with_suffix(".md").relative_to(ROOT))


if __name__ == "__main__":
    main()
