"""Build a concise English report from the audited validation snapshot."""
from __future__ import annotations

from datetime import datetime
import html
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

FOLDER = Path(__file__).resolve().parent
DATA = FOLDER / "concise_english_comparison_20261007.json"
FIGURE = FOLDER / "fig_matched_architecture_20261007.svg"
DESTINATION = FOLDER / "matched_architecture_brief_20261007.en.html"
MARKDOWN = FOLDER / "matched_architecture_brief_20261007.en.md"


def escape(value):
    return html.escape(str(value))


def statistic(row, channel, metric="rmse"):
    if row.get("seed_mean_sd"):
        value = row["seed_mean_sd"][channel][metric]
        return f"{value['mean']:.6f} ± {value['sd']:.6f}"
    return f"{row['physical_metrics'][channel][metric]:.6f}"


def budget(row):
    if row.get("seed_mean_sd"):
        return "3 seeds · 15k steps"
    if row["family"] == "prior_learned_oi":
        return "1 seed · 1.5k steps"
    if row["family"] == "fixed_pointwise_mlp":
        return "1 seed · 30 epochs"
    return "Fixed method"


def table(headers, rows, *, classes=None):
    head = "".join(f"<th>{escape(c)}</th>" for c in headers)
    body = []
    for i, row in enumerate(rows):
        style = f' class="{classes[i]}"' if classes else ""
        body.append(f"<tr{style}>" + "".join(f"<td>{escape(c)}</td>" for c in row) + "</tr>")
    return '<div class="table-wrap"><table><thead><tr>' + head + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"


def md_table(headers, rows):
    clean = lambda v: str(v).replace("|", "\\|")
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(clean(c) for c in row) + " |" for row in rows])


def main():
    data = json.loads(DATA.read_text())
    campaign = data["campaign"]
    complete = [r for r in data["comparison_rows"] if r["status"] == "complete_three_seeds"]
    pending = [r for r in data["comparison_rows"] if r["status"] != "complete_three_seeds"]
    assert len(data["comparison_rows"]) == 13 and campaign["registered_runs"] == 39
    assert all(r["seeds_completed"] == 3 and r["steps_per_run"] == 15000 for r in complete)
    assert all(r["seed_mean_sd"] is None for r in pending)
    assert data["validation_protocol"]["n_valid_per_variable"] == {"TEMP": 92517, "SALT": 92517}
    local_time = datetime.fromisoformat(data["as_of_utc"].replace("Z", "+00:00")).astimezone(ZoneInfo("America/Los_Angeles"))
    snapshot = local_time.strftime("%d %b %Y, %H:%M PDT")
    status = f"Validation snapshot · {campaign['completed_training_runs']}/{campaign['registered_runs']} training runs complete · {snapshot}"
    title = "CESM2 multimodal ocean reconstruction"
    objective = ("Reconstruct temperature and salinity from sparse Argo profiles and optional SST, SSS and steric sea height, "
                 "then return their means and marginal uncertainty at a requested location, native depth and month.")
    scope = ("This experiment evaluates retrospective monthly reconstruction at 20 depths (5–984.7 m). "
             "The final architecture has not yet been selected.")
    protocol = [
        ["Data split", "2000–2003 training; 2004 validation; previously used 2005 development"],
        ["Common observations", "6,080 input profiles/month; 4,256 training sources after target holdout"],
        ["Budget", "13 family/input settings × 3 seeds × 15,000 steps; 1,024 training queries/step"],
        ["Scoring", "92,517 validation values per variable; 351,895 development values per variable"],
        ["Targets", "Anomalies at each Argo profile’s own position; train-year climatology and normalization"],
    ]
    architecture = ("Depth-resolved observation encoders feed a shared latent Transformer and independent coordinate-query decoder. "
                    "Fixed spherical optimal interpolation (OI) supplies a numerical mean anchor. A separate 32-profile neighborhood "
                    "reads original T/S innovations at the query depth; a signed gate controls its correction. "
                    "The uncertainty head uses the query representation and local evidence statistics.")
    variants = ("Dense64 uses width 64 and 64 latents. Large variants use width 192, 96 latents and six latent blocks. "
                "Soft MoE replaces only the latent feed-forward layers with four experts × two slots; every expert executes. "
                "Local Transformer instead adds attention within each query’s own neighborhood and updates that query representation. "
                "Its registered arm is Argo-only. Queries do not attend to other prediction queries.")
    differences = [
        ["Observation representation", "Embeddings pooled into four depth bands", "One observation row per native depth"],
        ["Numerical mean", "Learned token-model estimate", "Frozen OI + zero-initialized residual and local gate"],
        ["Local evidence", "Refiner reads compressed embeddings", "Direct numerical innovations at the query depth"],
        ["Shared backbone", "Perceiver-style 64-slot latent already present", "Dense / Soft MoE; optional query-local Transformer"],
        ["Uncertainty", "No learned Gaussian scale", "Query- and evidence-conditioned marginal Gaussian scale"],
    ]
    old_note = ("Old-to-new comparisons also change capacity, optimization and loss (variable-balanced MSE + 0.02 Gaussian NLL). "
                "Only the registered Soft MoE full/latent-off and full/local-off controls isolate those pathways; "
                "their conclusions do not transfer directly to Local Transformer.")
    rows = data["fixed_baseline_rows"] + complete
    result_headers = ["Method (Argo-only)", "Per-run parameters", "Budget", "T RMSE (°C) ↓", "S RMSE (PSU) ↓"]
    result_rows = [[r["method"], f"{r['parameters_per_run']:,}", budget(r), statistic(r, "TEMP"), statistic(r, "SALT")] for r in rows]
    result_classes = ["highlight" if r["family"] == "local_transformer192" else "" for r in rows]
    stat_note = ("All values use the same 2004 locations and canonical float64 truth. Neural rows show mean ± sample SD across "
                 "three completed seeds at their validation-selected checkpoints; these are not prediction-ensemble metrics. "
                 "Fixed/single-seed baselines have no invented seed SD. Parameter totals include bypassed ablation modules.")
    no_local = ("Soft MoE without the added local path selects the initial OI checkpoint in all three seeds (best step = 0), "
                "which explains its identical RMSE to classical OI.")
    local = next(r for r in complete if r["family"] == "local_transformer192")
    prior = next(r for r in data["fixed_baseline_rows"] if r["family"] == "prior_learned_oi")
    gains = {ch: 100 * (1 - local["seed_mean_sd"][ch]["rmse"]["mean"] / prior["physical_metrics"][ch]["rmse"]) for ch in ("TEMP", "SALT")}
    finding = (f"Among the completed three-seed configurations, Local Transformer has the lowest validation RMSE. "
               f"Its seed-mean error is {gains['TEMP']:.2f}% lower for temperature and {gains['SALT']:.2f}% lower for salinity "
               "than the prior learned OI. This is a validation trend, not a final architecture decision or development-set improvement.")
    pending_headers = ["Incomplete comparison", "Inputs", "Complete seeds / 3"]
    pending_rows = [[r["method"], r["inputs"], f"{r['seeds_completed']}/3"] for r in pending]
    pending_note = ("Previous 64-slot, official 4DVarNet task adaptation and multimodal comparisons remain incomplete. "
                    "Partial-seed averages are withheld. After all 39 runs finish, validation ensembles freeze the selection "
                    "before new 2005 predictions are scored. The final report also includes MAE, bias, R², correlation, "
                    "depth-resolved errors, Gaussian NLL/CRPS, coverage and paired confidence intervals.")
    contribution = ("The candidate contribution is how OI-anchored means, original depth-resolved numerical evidence and shared "
                    "multimodal context work together. Its value requires strong-baseline and pathway-control evidence. "
                    "Shared-latent query architectures and Soft MoE are established components.")
    limitations = ("Synthetic SST/SSS equal noiseless 5 m truth; steric sea height is derived from the target T/S state. "
                   "These idealized inputs do not establish real-satellite gains. This snapshot does not establish forecasting, "
                   "arbitrary-depth interpolation, exact Bayesian updates, duplicate-evidence consistency or external SOTA.")
    sources = ('Architecture sources: <a href="https://arxiv.org/abs/2107.14795">Perceiver IO</a> and '
               '<a href="https://arxiv.org/abs/2308.00951">Soft MoE</a>. '
               'Closest-domain comparisons and claim limits: <a href="literature_novelty_audit_20261007.md">12-paper literature audit</a>. '
               'Protocol: <a href="matched_architecture_protocol_20261007.md">full implementation protocol</a>. '
               'Data: <a href="concise_english_comparison_20261007.csv">comparison CSV</a> and '
               '<a href="concise_english_comparison_20261007.json">audited snapshot</a>.')
    svg = FIGURE.read_text()
    svg = re.sub(r"<\?xml[^>]*\?>", "", svg).strip()
    css = """
    :root {color-scheme:light; --ink:#192b39; --muted:#62717e; --line:#dce3e8; --accent:#2b6175;}
    * {box-sizing:border-box} body {margin:0; background:#f2f4f6; color:var(--ink); font:15px/1.6 Arial,Helvetica,sans-serif;}
    main {max-width:1180px; margin:30px auto; padding:42px 48px; background:white; border:1px solid var(--line);}
    .eyebrow {text-transform:uppercase; font-size:11px; letter-spacing:1.7px; color:var(--accent); font-weight:700;}
    h1 {font-size:34px; line-height:1.18; margin:8px 0 12px; letter-spacing:-.5px;}
    .status {font-size:12px; color:var(--muted); border-bottom:1px solid var(--line); padding-bottom:20px; margin-bottom:22px;}
    h2 {font-size:20px; margin:26px 0 10px; font-weight:700;} p {margin:9px 0 12px;}
    .table-wrap {overflow-x:auto; margin:13px 0;} table {border-collapse:collapse; width:100%; font-size:12px; line-height:1.5;}
    th {background:#f3f6f8; color:#364b5a; text-align:left; padding:10px; border-bottom:1px solid #bbcbd5; font-weight:700;}
    td {padding:9px 10px; border-bottom:1px solid #e6ebef; vertical-align:top; font-variant-numeric:tabular-nums;}
    .highlight {background:#edf6f4;} .highlight td:first-child {font-weight:700;} figure {margin:16px 0 10px;}
    figure svg {display:block; width:100%; height:auto;} figcaption,.note {color:var(--muted); font-size:12px; line-height:1.5;}
    .finding {border-left:3px solid #3a8177; background:#f4f9f7; padding:12px 15px; font-size:14px;}
    a {color:var(--accent);} footer {border-top:1px solid var(--line); margin-top:22px; padding-top:15px; font-size:11px; color:var(--muted);}
    @media(max-width:760px) {main {margin:0; padding:25px 18px; border:none;} h1 {font-size:28px;}}
    @media print {body {background:white;} main {max-width:none; margin:0; padding:0; border:0;} h2 {break-after:avoid;}
      tr,figure,.finding {break-inside:avoid;} th {background:#eee!important;} @page {size:A4; margin:15mm;}}
    """
    sections = [f'<div class="eyebrow">Concise method and comparison report</div><h1>{escape(title)}</h1><div class="status">{escape(status)}</div>',
        f'<p>{escape(objective)}</p><p class="note">{escape(scope)}</p>',
        '<h2>1. Common experimental setting</h2>', table(["Item", "Registered protocol"], protocol),
        '<h2>2. Implemented architecture</h2>', f'<p>{escape(architecture)}</p>',
        f'<figure>{svg}<figcaption>Implemented candidate system, not the final selected model. Surface inputs are optional; the Local Transformer variant is Argo-only in this campaign. '
        '<a href="fig_matched_architecture_20261007.svg">Editable SVG</a> · <a href="fig_matched_architecture_20261007.png">PNG</a> · <a href="fig_matched_architecture_20261007.pdf">Vector PDF</a>.</figcaption></figure>',
        f'<p>{escape(variants)}</p>', '<h2>3. What changed from the previous implementation</h2>',
        table(["Component", "Previous 64-slot model", "Current candidates"], differences), f'<p class="note">{escape(old_note)}</p>',
        '<h2>4. Measured validation comparison</h2>', table(result_headers,result_rows,classes=result_classes),
        f'<p class="note">{escape(stat_note)}</p><p class="note">{escape(no_local)}</p><p class="finding">{escape(finding)}</p>',
        '<h2>5. Comparisons still in progress</h2>', table(pending_headers,pending_rows), f'<p class="note">{escape(pending_note)}</p>',
        '<h2>6. Contribution and interpretation</h2>', f'<p>{escape(contribution)} '
        '<a href="https://arxiv.org/abs/2107.14795">Perceiver IO</a>; <a href="https://arxiv.org/abs/2308.00951">Soft MoE</a>.</p>',
        f'<p class="note">{escape(limitations)}</p>', f'<footer>{sources}</footer>']
    document = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' + f'<title>{escape(title)}</title><style>{css}</style></head><body><main>' + "\n".join(sections) + '</main></body></html>\n'
    DESTINATION.write_text(document)
    md = [f"# {title}", "", status, "", objective, "", scope, "", "## Common experimental setting", "",
        md_table(["Item","Registered protocol"],protocol), "", "## Implemented architecture", "", architecture, "",
        "![Implemented candidate architecture](fig_matched_architecture_20261007.png)", "", "[Editable SVG](fig_matched_architecture_20261007.svg) · [Vector PDF](fig_matched_architecture_20261007.pdf).", "", variants, "",
        "## What changed from the previous implementation", "", md_table(["Component","Previous 64-slot model","Current candidates"],differences), "",old_note,"",
        "## Measured validation comparison", "", md_table(result_headers,result_rows), "",stat_note,"",no_local,"",finding,"",
        "## Comparisons still in progress", "",md_table(pending_headers,pending_rows),"",pending_note,"",
        "## Contribution and interpretation","",contribution + " [Perceiver IO](https://arxiv.org/abs/2107.14795); [Soft MoE](https://arxiv.org/abs/2308.00951).","", limitations,"",
        "Sources: [full protocol](matched_architecture_protocol_20261007.md), [comparison CSV](concise_english_comparison_20261007.csv), [audited data](concise_english_comparison_20261007.json), [literature audit](literature_novelty_audit_20261007.md).",""]
    MARKDOWN.write_text("\n".join(md))
    print(json.dumps({"report":str(DESTINATION),"markdown":str(MARKDOWN),"as_of":snapshot,"rows":len(rows)},ensure_ascii=False))


if __name__ == "__main__":
    main()
