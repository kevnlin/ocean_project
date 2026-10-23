"""Report of the kriging-attention pilot (``98_kriging_pilot_campaign.py``).

Reads every finished arm's summary and writes ``reports/synthetic/ka_pilot_20261019.{md,json}``:
2004 validation RMSE (the checkpoint year), 2005 development RMSE, the change against the
frozen OI and against the reference arm (softmax with the OI anchor), and what the kriging
arms learned (share of negative weights, length scales, gate). Unfinished arms are listed
as running. One seed: a screen, not a result.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/ka_pilot_20261019"
REPORT = ROOT / "reports/synthetic/ka_pilot_20261019"
REFERENCE_ARM = "sm_oi"
CH = ("TEMP", "SALT")


def _mean(value):
    if isinstance(value, dict):
        return value.get("mean")
    if isinstance(value, list):
        flat = [x for row in value for x in (row if isinstance(row, list) else [row])]
        return sum(flat) / len(flat) if flat else None
    return value


def diagnostics(validation):
    months = validation.get("diagnostics") or []
    out = {}
    for name in ("kriging_negative_share", "kriging_length_scale_km", "local_gate"):
        rows = [m[name] for m in months if name in m]
        if not rows:
            continue
        if name == "kriging_negative_share":
            out[name] = [sum(r[i] for r in rows) / len(rows) for i in range(2)]
        else:
            values = [_mean(r) for r in rows]
            out[name] = sum(values) / len(values)
    return out


def arm_result(path):
    summary = json.loads(path.read_text())
    val, dev = summary["validation"], summary.get("development")
    row = {"tag": summary["tag"], "local_weighting": summary["local_weighting"],
           "oi_anchor": summary["oi_anchor"], "best_step": summary["best_step"],
           "steps": summary["completed_steps"], "hours": summary["runtime_s"] / 3600,
           "params": summary["params"],
           "validation": {ch: val["scores"][ch]["rmse_physical"] for ch in CH},
           "validation_macro_z": val["scores"]["macro_z"],
           "validation_oi": {ch: val["baseline_scores"][ch]["rmse_physical"] for ch in CH},
           "diagnostics": diagnostics(val)}
    if dev is not None:
        row["development"] = {ch: dev["scores"][ch]["rmse_physical"] for ch in CH}
        row["development_oi"] = {ch: dev["baseline_scores"][ch]["rmse_physical"] for ch in CH}
    return row


def change(a, b):
    return {ch: 100. * (a[ch] - b[ch]) / b[ch] for ch in CH}


def main():
    campaign = json.loads((OUTPUT / "campaign.json").read_text())
    rows, running = [], []
    for tag in campaign["jobs"]:
        path = OUTPUT / tag / f"summary_seed{tag.rsplit('_s', 1)[1]}.json"
        if path.exists():
            rows.append(arm_result(path))
        else:
            running.append(tag)
    reference = next((r for r in rows if r["tag"].startswith(REFERENCE_ARM + "_")), None)
    for r in rows:
        r["validation_vs_oi_pct"] = change(r["validation"], r["validation_oi"])
        if reference is not None:
            r["validation_vs_reference_pct"] = change(r["validation"], reference["validation"])
        if "development" in r:
            r["development_vs_oi_pct"] = change(r["development"], r["development_oi"])
    fmt = lambda d: f"{d['TEMP']:.4f} / {d['SALT']:.4f}"
    pct = lambda d: f"{d['TEMP']:+.1f} % / {d['SALT']:+.1f} %"
    lines = ["# Kriging attention pilot, CESM2, Argo + surface, lead 0", "",
             "Runner 97 at the long-record reference: the medium model (128 slots, width 256), the "
             "4-year window 2000-2003, 15,000 steps of 1,024 queries, seed 1234. Checkpoint chosen on "
             "2004; 2005 scored once at the end. The arms change only how the local path weighs the "
             "32 nearest profiles and whether the model sees the frozen OI. One seed: a screen, not a result.",
             "", "## RMSE, temperature °C / salinity PSU", "",
             "| arm | local weighting | OI anchor | 2004 | vs OI | vs softmax + OI | 2005 | 2005 vs OI | best step |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['tag']} | {r['local_weighting']} | {'yes' if r['oi_anchor'] else 'no'} | "
                     f"{fmt(r['validation'])} | {pct(r['validation_vs_oi_pct'])} | "
                     f"{pct(r['validation_vs_reference_pct']) if reference else '–'} | "
                     f"{fmt(r['development']) if 'development' in r else '–'} | "
                     f"{pct(r['development_vs_oi_pct']) if 'development' in r else '–'} | {r['best_step']} |")
    if rows:
        oi = rows[0]["validation_oi"]
        lines += ["", f"Frozen OI on the same cells: 2004 {fmt(oi)}"
                  + (f"; 2005 {fmt(rows[0]['development_oi'])}" if "development_oi" in rows[0] else "") + "."]
    lines += ["", "## What the kriging arms learned (2004 validation, mean over months)", "",
              "| arm | negative weights T / S | length scale at the query, km | local gate | hours |",
              "|---|---|---|---|---|"]
    for r in rows:
        d = r["diagnostics"]
        neg = d.get("kriging_negative_share")
        negative = f"{100 * neg[0]:.0f} % / {100 * neg[1]:.0f} %" if neg else "–"
        scale = d.get("kriging_length_scale_km")
        length = f"{scale:.0f}" if scale is not None else "–"
        gate = f"{d['local_gate']:.3f}" if "local_gate" in d else "–"
        lines.append(f"| {r['tag']} | {negative} | {length} | {gate} | {r['hours']:.1f} |")
    if running:
        lines += ["", "Still running: " + ", ".join(running) + "."]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    REPORT.with_suffix(".json").write_text(json.dumps({"rows": rows, "running": running}, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
