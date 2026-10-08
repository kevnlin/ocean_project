"""Concise live report: complete matched evidence and pending mechanism tests.

No training or model-selection action is performed by this reporter. New 2005
arrays remain unread until the registered validation selection is frozen and
revalidated by the canonical matched reporter.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import html
import importlib.util
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location("canonical_matched_report", ROOT / "experiments/synthetic/53_matched_reconstruction_report.py")
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)
FOLDER = ROOT / "reports/synthetic"
STEM = "innovation_contribution_report_20261007.en"
FIGURE = "fig_innovation_architecture_20261007"
LABELS = {
    "climatology": "Climatology", "nearest_profile": "Nearest profile", "oi": "Fixed OI",
    "prior_learned_oi": "Prior learned OI", "fixed_pointwise_mlp": "Fixed pointwise MLP",
    "previous_token64": "Previous 64-slot", "dense64": "Dense64", "dense192": "Dense192",
    "soft_moe192": "Soft MoE192", "local_transformer192": "Local Transformer192",
    "soft_moe192_latent_off": "Soft MoE192 / latent off",
    "soft_moe192_local_off": "Soft MoE192 / local off", "official4dvarnet": "4DVarNet task adaptation",
    "local_control": "Matched local control", "local_latent_off": "Local / latent off",
    "local_off": "Local / local off", "profile_none": "Full profiles / no shift",
    "profile_shared": "Full profiles / shared T/S shift", "profile_independent": "Full profiles / independent shifts",
    "cov_diagonal": "Covariance / diagonal R", "cov_correlated": "Covariance / correlated R",
    "aligned_correlated": "Shared alignment + correlated R", "operator_direct": "Column / direct surface context",
    "operator_update": "Column / observation update", "all_modules": "All candidate modules",
}
HYPOTHESES = [
    ("Complete-profile context", "profile_none", "local_control", "Does complete-profile context add value to the same local backbone?"),
    ("Full-profile alignment", "profile_shared", "profile_none", "Does a shared displacement help beyond complete-profile context?"),
    ("Joint T/S displacement", "profile_shared", "profile_independent", "Does sharing the physical displacement help?"),
    ("Correlated observation error", "cov_correlated", "cov_diagonal", "Does source/profile correlation improve numerical updates?"),
    ("Alignment + covariance", "aligned_correlated", "cov_correlated", "Does alignment add value to the covariance update?"),
    ("Covariance after alignment", "aligned_correlated", "profile_shared", "Does correlated updating help already aligned profiles?"),
    ("Observation operators", "operator_update", "operator_direct", "Does an explicit residual operator help beyond direct surface context?"),
    ("Combined architecture", "all_modules", "local_control", "Do the candidate mechanisms improve the matched local control?"),
    ("Combined operator increment", "all_modules", "aligned_correlated", "Does the operator increment help the aligned correlated system?"),
    ("Local shared latent", "local_control", "local_latent_off", "Does the shared latent help this local architecture?"),
    ("Local numerical pathway", "local_control", "local_off", "Does local numerical evidence help this architecture?"),
]


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_text(path, text):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def scientific_sources(summary, tag):
    hashes = summary.get("source_sha256", summary.get("source_hashes", {}))
    if not hashes:
        raise ValueError(f"{tag}: scientific source hashes absent")
    for name, recorded in hashes.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if not path.is_file() or S.sha256(path) != recorded:
            raise ValueError(f"{tag}: scientific source mismatch: {name}")


def audit_job(job, ref, *, new=False):
    summary, contract = S.summary_contract(job)
    if new and not job.get("reuse") and summary.get("completed_steps") != 15000:
        raise S.PendingReport(f"{job['tag']}: explicit completed_steps=15000 required")
    scientific_sources(summary, job["tag"])
    prediction, canonical_contract = S.validation_prediction(job, ref)
    return prediction, summary, canonical_contract


def report_group(jobs, ref, origin):
    values, summaries, contracts, errors = [], [], [], []
    for job in sorted(jobs, key=lambda j: j["seed"]):
        try:
            prediction, summary, contract = audit_job(job, ref, new=origin == "new")
            values.append(prediction); summaries.append(summary); contracts.append(contract)
        except (S.PendingReport, ValueError, KeyError) as error:
            errors.append({"tag": job["tag"], "reason": str(error),
                           "status": "pending" if isinstance(error, S.PendingReport) else "audit_failed"})
    row = {"family": jobs[0]["family"], "mode": "surface" if jobs[0]["surface"] else "argo",
           "origin": origin, "seeds_completed": len(values), "seeds_expected": 3,
           "reuse": all(job.get("reuse", False) for job in jobs),
           "status": "complete_three_seeds" if len(values) == 3 and sorted(j["seed"] for j in jobs) == list(S.SEEDS) else "pending",
           "pending": errors, "tags": [j["tag"] for j in jobs]}
    if row["status"] != "complete_three_seeds":
        return row, None, None
    fingerprint_keys = ("cohort_fingerprint", "feature_fingerprint", "scene_fingerprint")
    if any(len({summary.get("data", {}).get(key) for summary in summaries}) > 1 for key in fingerprint_keys):
        raise ValueError(f"{row['family']}: data fingerprints differ across seeds")
    seed_metrics = [{"seed": job["seed"], **S.physical_metrics(value, ref)}
                    for job, value in zip(sorted(jobs, key=lambda j:j["seed"]), values)]
    combined = S.ensemble(values)
    row.update(seed_metrics=seed_metrics, seed_mean_sd=S.summarize_seeds(seed_metrics),
               ensemble=S.physical_metrics(combined, ref),
               parameters_per_run=[summary.get("params") for summary in summaries],
               runtime_s_per_run=[summary.get("runtime_s") for summary in summaries],
               best_steps=[summary.get("best_step") for summary in summaries],
               validation_contracts=contracts,
               uncertainty="Moment-matched Gaussian across three model seeds" if "std" in combined else "Unavailable")
    return row, combined, summaries


def groups(campaign):
    result = {}
    for job in campaign.get("jobs", []):
        result.setdefault((bool(job["surface"]), job["family"]), []).append(job)
    return result


def baseline_rows(old_path, old, baseline_dir, ref):
    rows, predictions = [], {}
    replay = read_json(baseline_dir / "replay_manifest.json")
    for name, path in S.baseline_candidates(baseline_dir, "validation").items():
        expected = replay["exports"]["validation"][name].get("sha256")
        if expected and S.sha256(path) != expected:
            raise ValueError(f"{name}: baseline export hash differs")
        arrays = S.load_arrays(path); S.assert_identity(arrays, ref, name)
        prediction = {"mean": np.asarray(arrays["mean"], np.float64)}
        rows.append({"family": name, "mode": "argo", "origin": "baseline", "status": "fixed",
                     "budget": "Fixed numerical baseline", "parameters_per_run": [], "runtime_s_per_run": [],
                     **S.physical_metrics(prediction, ref)})
        predictions[("baseline", "argo", name)] = prediction
    prior = S.prior_learned_oi_validation(old_path, old, ref)
    for name, digest in prior["registry"].get("source_hashes", {}).items():
        if S.sha256(ROOT / name) != digest:
            raise ValueError(f"prior learned OI replay source changed: {name}")
    arrays = S.load_arrays(prior["registry"]["arrays"]["validation"]["path"])
    S.assert_identity(arrays, ref, "prior_learned_oi")
    prediction = {"mean": np.asarray(arrays["mean"], np.float64)}
    rows.append({"family": "prior_learned_oi", "mode": "argo", "origin": "baseline", "status": "fixed",
                 "budget": "Existing 1 seed / 1,500 steps", "parameters_per_run": [prior.get("parameters")],
                 "runtime_s_per_run": [], **S.physical_metrics(prediction, ref)})
    predictions[("baseline", "argo", "prior_learned_oi")] = prediction
    fixed = S.fixed_mlp_validation(old_path, old, ref)
    scientific_sources(read_json(fixed["summary_path"]), "fixed_pointwise_mlp")
    arrays = S.load_arrays(fixed["validation_array"]); S.assert_identity(arrays, ref, "fixed_pointwise_mlp")
    prediction = {"mean": np.asarray(arrays["mean"], np.float64)}
    rows.append({"family": "fixed_pointwise_mlp", "mode": "argo", "origin": "baseline", "status": "fixed",
                 "budget": "Fixed 1 seed / 30 epochs", "parameters_per_run": [fixed["training_contract"].get("params")],
                 "runtime_s_per_run": [], **S.physical_metrics(prediction, ref)})
    predictions[("baseline", "argo", "fixed_pointwise_mlp")] = prediction
    return rows, predictions


def matched_settings(actual, control):
    config_keys = ("variant", "width", "n_latents", "n_heads", "n_blocks", "n_query_blocks", "n_sat_features")
    training_keys = ("steps", "queries", "context_profiles", "lr", "weight_decay", "nll_weight", "val_cells", "amp")
    differences = []
    for a, b in zip(actual, control):
        for section, keys in (("config", config_keys), ("training", training_keys)):
            for key in keys:
                if a.get(section, {}).get(key) != b.get(section, {}).get(key):
                    differences.append(f"seed {a['seed']}: {section}.{key}")
    return sorted(set(differences))


def verify_new_selection(choice, campaign_path, campaign, rows, predictions):
    """Recheck stable scientific contracts; summary/ZIP timestamps may change."""
    if choice.get("campaign_sha256") != S.sha256(campaign_path):
        raise ValueError("frozen new campaign hash differs")
    expected_rule = "minimum ensemble per-variable standardized RMSE on 2004; separate input arms"
    if choice.get("rule") != expected_rule:
        raise ValueError("frozen new selection rule differs")
    contracts = {tag: contract for row in rows if row.get("origin") == "new"
                 and row["status"] == "complete_three_seeds"
                 for tag, contract in zip(sorted(row["tags"], key=lambda t: next(j["seed"] for j in campaign["jobs"] if j["tag"]==t)),
                                          row["validation_contracts"])}
    frozen = choice.get("validation_contracts", [])
    if len(frozen) != len(campaign["jobs"]):
        raise ValueError("frozen new validation contract count differs")
    for job, old_contract in zip(campaign["jobs"], frozen):
        if job["tag"] not in contracts:
            raise S.PendingReport("every registered three-seed group must pass audit before development")
        current = contracts[job["tag"]]
        for key in ("training_contract_sha256", "checkpoint_sha256", "validation_content_sha256"):
            if old_contract.get(key) != current.get(key):
                raise ValueError(f"frozen scientific content differs: {job['tag']} / {key}")
    current_candidates = {(bool(c["surface"]), c["family"]): c["score"] for c in choice.get("candidates", [])}
    for (surface, family), jobs in groups(campaign).items():
        key = ("new", "surface" if surface else "argo", family)
        if key not in predictions or (surface, family) not in current_candidates:
            raise S.PendingReport("frozen candidate family is unavailable")
        # Frozen scores were calculated with the same canonical reference.
        row = next(r for r in rows if (r.get("origin"),r.get("mode"),r.get("family")) == key)
        if not np.isclose(row["ensemble"]["mean_standardized_rmse"], current_candidates[(surface,family)], rtol=0, atol=1e-12):
            raise ValueError("frozen validation candidate score changed")
    if len(current_candidates) != len(groups(campaign)):
        raise ValueError("frozen candidate set differs")
    for surface in (False, True):
        selected = choice.get("selected", {}).get(str(surface))
        candidates = [c for c in choice["candidates"] if bool(c["surface"]) == surface]
        if not candidates or selected != min(candidates, key=lambda c:c["score"]):
            raise ValueError("frozen recommendation does not match validation criterion")


def operational_snapshot(campaign_path, campaign=None):
    """Read live queue/verification/diagnostic metadata without scoring arrays."""
    campaign = campaign if campaign is not None else (read_json(campaign_path) if campaign_path.is_file() else {"jobs":[]})
    parent = campaign_path.parent
    status_path, service_path = parent/'status.json', parent/'service_health.json'
    smoke_path, verification_path = parent/'engineering_smoke.json', parent/'verification.json'
    status = read_json(status_path) if status_path.is_file() else {
        "phase":"registered / launch not yet confirmed", "running":{},
        "pending":[job['tag'] for job in campaign.get('jobs',[]) if not job.get('reuse')]}
    service = read_json(service_path) if service_path.is_file() else {}
    started_at = status.get('started_at_utc')
    if not started_at and service.get('healthy'):
        started_at=service.get('started_at_utc')
    verification=read_json(verification_path) if verification_path.is_file() else {}
    if verification:
        tests=f"{verification.get('unique_focused_tests_passed',0)} unique focused tests passed; {verification.get('sandbox_cuda_tests_skipped',0)} CUDA tests skipped in the restricted sandbox"
    else:
        tests='Focused test verification artifact is pending'
    diagnostics=[]
    for job in campaign.get('jobs',[]):
        path=Path(job['output'])/f"stress_seed{job['seed']}.json"
        if path.is_file():
            diagnostic=read_json(path)
            checkpoint=Path(diagnostic.get('checkpoint',''))
            if not checkpoint.is_file() or S.sha256(checkpoint)!=diagnostic.get('checkpoint_sha256'):
                diagnostics.append({'tag':job['tag'],'family':job['family'],'mode':'surface' if job['surface'] else 'argo',
                                    'path':str(path),'status':'audit failed: stress checkpoint identity differs'})
            else:
                diagnostics.append({'tag':job['tag'],'family':job['family'],'mode':'surface' if job['surface'] else 'argo',
                    'path':str(path),'status':'individual fixed-checkpoint capped diagnostic; no family aggregation',
                    'seed':diagnostic.get('seed'),'n_queries':diagnostic.get('n_queries'),
                    'conditions':diagnostic.get('conditions',{}),'duplicate_component_probe':diagnostic.get('duplicate_component_probe')})
    return {'execution_status':status,'formal_training_started_at_utc':started_at,
        'engineering_validation':{'smoke':read_json(smoke_path) if smoke_path.is_file() else {},
            'verification':verification,'source':str(smoke_path),'verification_source':str(verification_path),
            'test_result':tests,'scope':'Engineering verification only; no efficacy or contribution evidence'},
        'source_robustness':{'schedule':'after all registered training runs complete; before development scoring',
            'script':str(ROOT/'experiments/synthetic/65_innovation_stress.py'),'status_source':str(status_path),
            'scope':'2004 capped source-only robustness diagnostics, fixed selected checkpoints; separate from formal three-seed contrasts',
            'conditions':campaign.get('stress_tests',[]),'artifacts':diagnostics}}


def create_snapshot(campaign_path, old_path, baseline_dir, selection_path, draws):
    old = read_json(old_path)
    new = read_json(campaign_path) if campaign_path.is_file() else {"jobs": []}
    reference = S.load_reference(baseline_dir, "validation")
    rows, predictions = baseline_rows(old_path, old, baseline_dir, reference)
    summary_map = {}
    counts = {}
    for origin, campaign in (("old", old), ("new", new)):
        completed = 0
        reused_completed = 0
        for (_, _), jobs in sorted(groups(campaign).items()):
            row, prediction, summaries = report_group(jobs, reference, origin)
            rows.append(row); completed += row["seeds_completed"]
            if row["reuse"]:
                reused_completed += row["seeds_completed"]
            key = (origin, row["mode"], row["family"])
            if prediction is not None:
                predictions[key] = prediction; summary_map[key] = summaries
        counts[origin] = {"registered": len(campaign["jobs"]), "completed": completed,
                          "reused_completed": reused_completed,
                          "fresh_completed": completed-reused_completed,
                          "fresh_registered": sum(not job.get("reuse",False) for job in campaign["jobs"]),
                          "complete_groups": sum(r.get("origin") == origin and r["status"] == "complete_three_seeds" for r in rows)}
    contrasts = {}
    local = ("old", "argo", "local_transformer192")
    for name, key in (("Dense192", ("old", "argo", "dense192")),
                      ("Prior learned OI", ("baseline", "argo", "prior_learned_oi"))):
        if local in predictions and key in predictions:
            contrasts["Local Transformer192 vs " + name] = {
                "scope": "Old campaign / system comparison / 2004 validation",
                "paired_month_ci": S.paired_contrast(predictions[local], predictions[key], reference, draws=draws)}
    mechanisms = []
    hypotheses = HYPOTHESES
    if new.get("paired_contrasts"):
        titles = {(actual,control):title for title,actual,control,_ in HYPOTHESES}
        hypotheses = [(titles.get((contrast['candidate'],contrast['control']), contrast['question']),
                       contrast['candidate'],contrast['control'],contrast['question']) for contrast in new['paired_contrasts']]
    registered_keys = {("surface" if surface else "argo", family) for surface,family in groups(new)}
    for mode in ("argo", "surface"):
        for title, actual, control, question in hypotheses:
            if registered_keys and ((mode,actual) not in registered_keys or (mode,control) not in registered_keys):
                continue
            keys = (("new", mode, actual), ("new", mode, control))
            item = {"mechanism": title, "mode": mode, "actual": actual, "control": control,
                    "question": question, "status": "pending: full matched three-seed contrast required"}
            if all(key in predictions for key in keys):
                differences = matched_settings(summary_map[keys[0]], summary_map[keys[1]])
                paired = S.paired_contrast(predictions[keys[0]], predictions[keys[1]], reference, draws=draws)
                item.update(paired_month_ci=paired, unmatched_settings=differences)
                supported = not differences and all(paired[ch]["ci95_delta_rmse"][1] < 0 for ch in S.CHANNELS)
                item["status"] = "validation evidence supports this contrast" if supported else (
                    "complete, but capacity/training mismatch prevents attribution" if differences else "complete: benefit not established for both variables")
            mechanisms.append(item)
    development = {"status": "withheld: validation selection is not frozen", "rows": []}
    if campaign_path.is_file() and selection_path.is_file():
        try:
            choice = read_json(selection_path)
            verify_new_selection(choice, campaign_path, new, rows, predictions)
            dev_ref = S.load_reference(baseline_dir, "development")
            for (_, _), jobs in sorted(groups(new).items()):
                values = []
                for job in sorted(jobs, key=lambda j:j["seed"]):
                    arrays = S.load_arrays(S.prediction_path(job, "development"))
                    S.assert_identity(arrays, dev_ref, job["tag"])
                    value = {"mean": np.asarray(arrays["mean"], np.float64)}
                    if "std" in arrays:
                        value["std"] = np.asarray(arrays["std"], np.float64)
                    values.append(value)
                development["rows"].append({"family": jobs[0]["family"], "mode": "surface" if jobs[0]["surface"] else "argo",
                                           **S.physical_metrics(S.ensemble(values), dev_ref)})
            development["status"] = "2005 development after revalidated frozen selection; previously used test year"
        except (S.PendingReport, ValueError, KeyError) as error:
            development["status"] = "withheld/incomplete: " + str(error)
            development["rows"] = []
    return {"schema_version": 1, "as_of_utc": datetime.now(timezone.utc).isoformat(),
            "split": "2004 validation", "counts": counts, "rows": rows, "contrasts": contrasts,
            "mechanisms": mechanisms, "development": development,
            **operational_snapshot(campaign_path,new),
            "protocol": {"training_years": [2000, 2003], "validation_year": 2004,
                "inputs_per_month": 6080, "validation_values_per_variable": 92517,
                "new_budget": "15,000 steps; seeds 1234/1235/1236", "bootstrap_draws": draws,
                "truth": "canonical original float64 truth; exact scoring identities and training normalization",
                "aggregation": "neural main table = seed mean ± sample SD; contrasts = prediction ensemble / whole-month paired bootstrap",
                "pilots": "excluded from contribution evidence"},
            "sources": {"original_campaign": str(old_path), "new_campaign": str(campaign_path),
                "selection": str(selection_path), "canonical_reference": str(baseline_dir / "oi_validation.npz"),
                "reference_sha256": S.sha256(baseline_dir / "oi_validation.npz"), "reporter_sha256": S.sha256(Path(__file__))}}


def fmt(value, digits=5):
    return "—" if value is None else f"{value:.{digits}f}"


def statistic(row, channel, metric):
    if row.get("seed_mean_sd"):
        value = row["seed_mean_sd"][channel][metric]
        return fmt(value["mean"]) + " ± " + fmt(value["sd"])
    return fmt(row["physical_metrics"][channel].get(metric))


def table(headers, rows):
    esc = lambda x: html.escape(str(x))
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{esc(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{esc(c)}</td>' for c in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def md_table(headers, rows):
    clean = lambda x: str(x).replace("|", "\\|")
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |'] + ['| ' + ' | '.join(clean(c) for c in row) + ' |' for row in rows])


def architecture_svg():
    # Native SVG text, paths and rounded rectangles remain individually editable.
    body = ['<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="670" viewBox="0 0 1280 670">',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="#526879"/></marker></defs>',
        '<rect width="1280" height="670" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#203749} .title{font-size:23px;font-weight:700}.label{font-size:16px;font-weight:700}.small{font-size:14px}.note{font-size:13px;fill:#647988}.arrow{fill:none;stroke:#526879;stroke-width:2;marker-end:url(#arrow)}</style>',
        '<text x="30" y="35" class="title">OI-anchored ocean reconstruction and candidate mechanisms</text>',
        '<text x="30" y="58" class="note">Current backbone above; isolated experimental additions below. Every target query is decoded independently.</text>']

    def box(x, y, w, h, title, lines, experimental=False):
        fill, stroke = ("#fff6e9", "#bf9150") if experimental else ("#edf4f7", "#7399ad")
        body.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        body.append(f'<text x="{x+15}" y="{y+27}" class="label">{html.escape(title)}</text>')
        for i, line in enumerate(lines):
            body.append(f'<text x="{x+15}" y="{y+52+i*21}" class="small">{html.escape(line)}</text>')

    def arrow(path, dashed=False):
        body.append(f'<path d="{path}" class="arrow"' + (' stroke-dasharray="6 5"' if dashed else '') + '/>')

    box(30, 90, 245, 115, "Observed source profiles", ["Raw depth-resolved T/S", "Position, month, validity", "Complete source pool"])
    box(325, 90, 255, 115, "Shared observation latent", ["Depth observation encoder", "Dense Transformer / Soft MoE", "Established backbone"])
    box(635, 90, 270, 115, "Coordinate-query decoder", ["Own local neighborhood", "Query-local Transformer", "Numerical T/S innovations"])
    box(970, 135, 275, 135, "Predicted T/S", ["Frozen OI + learned correction", "At requested native depth", "Learned marginal uncertainty"])
    box(325, 235, 255, 92, "Frozen spherical OI", ["Numerical interpolation anchor", "Same input profiles / depths"])
    box(635, 235, 270, 92, "Optional surface context", ["CESM2 SST / SSS / steric SSH", "Idealized inputs derived from truth"])
    arrow("M275 146 H325"); arrow("M580 146 H635"); arrow("M905 146 H935 V177 H970")
    arrow("M150 205 V281 H325"); arrow("M580 281 H600 V343 H945 V228 H970"); arrow("M770 235 V205")
    body.append('<path d="M30 360 H1245" stroke="#d9e1e6" stroke-width="1.5"/>')
    box(30, 417, 390, 157, "1 · Complete-profile alignment", ["Encode observed absolute T/S", "Bounded shared or independent shift", "Subtract source climatology at destination", "Control: full profiles with no shift"], True)
    box(445, 417, 390, 157, "2 · Correlation-aware numerical update", ["PSD state B and observation-error R", "Depth / T-S / profile / source correlation", "Structured precision + Woodbury solve", "Control: diagonal observation error"], True)
    box(860, 417, 385, 157, "3 · Observation-operator residual", ["Exact native-depth SST / SSS mapping", "TEOS-10 SSH: train-climate Jacobian", "Same gain updates mean and variance", "Control: direct surface context"], True)
    arrow("M225 417 V350 H610 V222 H685 V205", True)
    arrow("M640 417 V352 H620 V213 H860 V205", True)
    arrow("M1060 417 V300 H1110 V270", True)
    body.append('<rect x="25" y="373" width="1230" height="29" fill="white"/>')
    body.append('<text x="30" y="394" class="label">Experimental additions: contributions remain hypotheses until matched tests complete</text>')
    body.append('<text x="30" y="614" class="small">OI and learned local states reuse source evidence: this diagram does not claim independent sequential Bayesian assimilation.</text>')
    body.append('<text x="30" y="639" class="note">Candidate covariance diagnostics and a linearized operator are conditional mechanisms; final uncertainty needs calibration tests.</text>')
    body.append('</svg>')
    return '\n'.join(body)


def write_outputs(data, report_dir):
    report_dir.mkdir(parents=True, exist_ok=True)
    figure = report_dir / (FIGURE + '.svg')
    atomic_text(figure, architecture_svg())
    rendered = []
    try:
        import cairosvg
        cairosvg.svg2png(url=str(figure), write_to=str(report_dir / (FIGURE + '.png')), output_width=1920)
        cairosvg.svg2pdf(url=str(figure), write_to=str(report_dir / (FIGURE + '.pdf')))
        rendered = ['png', 'pdf']
    except (ImportError, OSError) as error:
        data['render_note'] = str(error)
    data['figure_formats'] = ['svg'] + rendered
    complete = [r for r in data['rows'] if r['status'] in ('fixed', 'complete_three_seeds')]
    headers = ['Method', 'Inputs', 'T RMSE (°C)', 'S RMSE (PSU)', 'T MAE', 'S MAE', 'T bias', 'S bias']
    main_rows = [[LABELS.get(r['family'], r['family']) + (' (reused)' if r.get('reuse') else ''), 'Argo + surface' if r['mode']=='surface' else 'Argo',
                 *[statistic(r, ch, metric) for metric in ('rmse', 'mae', 'mean_bias') for ch in S.CHANNELS]] for r in complete]
    cost_headers = ['Method / inputs', 'Parameters / seed', 'Minutes / seed', 'Ensemble T / S CRPS', 'Ensemble T / S 95% coverage']
    cost_rows = []
    for row in complete:
        metrics = row.get('ensemble', row).get('physical_metrics')
        params = row.get('parameters_per_run', []); runtimes = [v for v in row.get('runtime_s_per_run', []) if v is not None]
        p = ', '.join(str(v) for v in sorted(set(params))) if params and all(v is not None for v in params) else '—'
        runtime = f'{min(runtimes)/60:.1f}–{max(runtimes)/60:.1f}' if len(runtimes)==3 else '—'
        crps = ' / '.join(fmt(metrics[ch].get('crps')) for ch in S.CHANNELS)
        coverage = ' / '.join('—' if metrics[ch].get('coverage_95') is None else f"{100*metrics[ch]['coverage_95']:.1f}%" for ch in S.CHANNELS)
        cost_rows.append([LABELS.get(row['family'],row['family'])+' / '+row['mode'],p,runtime,crps,coverage])
    pending_rows = [[r['origin'], LABELS.get(r['family'],r['family']),r['mode'],f"{r['seeds_completed']}/3",
                    'Audit failed: '+next(e['reason'] for e in r['pending'] if e['status']=='audit_failed') if any(e['status']=='audit_failed' for e in r['pending']) else 'Awaiting full-budget artifacts'] for r in data['rows'] if r['status']=='pending']
    ci_rows = []
    for name, item in data['contrasts'].items():
        ci_rows.append([name, *[f"{fmt(item['paired_month_ci'][ch]['delta_rmse'],6)} [{fmt(item['paired_month_ci'][ch]['ci95_delta_rmse'][0],6)}, {fmt(item['paired_month_ci'][ch]['ci95_delta_rmse'][1],6)}]" for ch in S.CHANNELS]])
    mechanism_rows = [[m['mechanism'],m['mode'],m['actual']+' vs '+m['control'],m['status']] for m in data['mechanisms']]
    snapshot = datetime.fromisoformat(data['as_of_utc']).astimezone(ZoneInfo('America/Los_Angeles')).strftime('%d %b %Y, %H:%M PDT')
    old,new = data['counts']['old'],data['counts']['new']
    status = f"{snapshot} · Original campaign {old['completed']}/{old['registered']} audited runs · New campaign {new['fresh_completed']}/{new['fresh_registered']} fresh runs complete + {new['reused_completed']} reused controls"
    execution=data['execution_status']
    running_count=len(execution.get('running',{})); pending_count=len(execution.get('pending',[]))
    execution_note=f"Queue: {execution.get('phase','unknown')} · {running_count} running · {pending_count} pending in the latest queue status."
    if data.get('formal_training_started_at_utc'):
        execution_note+=' Formal training started: '+str(data['formal_training_started_at_utc'])+'.'
    smoke=data['engineering_validation']['smoke']
    engineering_note = (f"Engineering verification: {len(smoke.get('results',[]))} recipes passed GPU forward/backward with {smoke.get('queries',0):,} queries and the complete source pool; "
                        + data['engineering_validation']['test_result'] + '. These checks and short smoke runs do not establish efficacy.') if smoke else 'Engineering verification artifacts are pending.'
    supported = [m for m in data['mechanisms'] if m['status']=='validation evidence supports this contrast']
    takeaway = ('Matched validation contrasts support: '+', '.join(m['mechanism']+' ('+m['mode']+')' for m in supported)+'. External novelty and generalization still require independent comparisons.' if supported else
                'Usable now: a measured OI-anchored reconstruction system and the Local Transformer validation trend. Full-profile alignment, correlation-aware updates, observation operators and their combination remain candidate contributions; no new mechanism result is claimed.')
    protocol = 'Train: 2000–2003. Validate: 2004, 92,517 values per variable, identical profile/depth/month identities and canonical float64 truth. Learned families: three seeds × 15,000 steps, 6,080 source profiles/month at evaluation. SST/SSS are noiseless 5 m CESM2 products; steric SSH is derived from T/S truth. The previously used 2005 year is development evidence.'
    stats = 'Main table reports the arithmetic seed mean ± sample SD, not ensemble RMSE. Fixed baselines have no seed SD. A learned row appears only after all three complete runs pass budget, checkpoint, source-hash, identity and normalization checks. Bias means prediction minus truth.'
    ci_note = f'Contrasts use the three-seed prediction ensemble and {data["protocol"]["bootstrap_draws"]:,} paired bootstrap draws of whole months; negative ΔRMSE favors Local Transformer. There are only 12 validation months. System comparisons do not isolate attention or latent necessity.'
    mechanism_note = 'Mechanism attribution requires the registered full three-seed matched control, consistent architecture/training settings, and a 95% paired-month ΔRMSE interval below zero for both variables. This is validation evidence; it does not establish external method novelty. Pilot runs are excluded. New development arrays stay unread until selection is frozen.'
    cost_note = 'Runtime is recorded per-run wall time, including validation; hardware/concurrency are not normalized. Parameters are allocated counts, not active FLOPs. Neural ensemble scales are moment-matched Gaussian summaries; no uncertainty is invented for deterministic baselines.'
    dev = data['development']
    robustness=data.get('source_robustness',{})
    stress_rows=[]
    for artifact in robustness.get('artifacts',[]):
        for condition,item in artifact.get('conditions',{}).items():
            metrics=item.get('metrics',{})
            stress_rows.append([artifact['tag'],artifact.get('seed'),condition,artifact.get('n_queries'),
                fmt(metrics.get('TEMP',{}).get('rmse')),fmt(metrics.get('SALT',{}).get('rmse'))])
    stress_note=f"Source robustness: {robustness.get('schedule','pending')}. {len(robustness.get('artifacts',[]))} individual diagnostic artifacts available. These capped fixed-checkpoint results stay separate; a single artifact is never presented as a three-seed family or formal mechanism contrast."
    stress_headers=['Checkpoint / run','Diagnostic seed','Condition','Capped queries','T RMSE (°C)','S RMSE (PSU)']
    stress_html=table(stress_headers,stress_rows) if stress_rows else '<p class="note">No completed source-robustness diagnostic is available yet.</p>'
    dev_table = ''
    dev_md = ''
    if dev['rows']:
        dh = ['Method','Inputs','Ensemble T RMSE (°C)','Ensemble S RMSE (PSU)']
        dr = [[LABELS.get(r['family'],r['family']),r['mode'],fmt(r['physical_metrics']['TEMP']['rmse']),fmt(r['physical_metrics']['SALT']['rmse'])] for r in dev['rows']]
        dev_table,dev_md = table(dh,dr),md_table(dh,dr)
    links = [('Original manifest', data['sources']['original_campaign']), ('New manifest',data['sources']['new_campaign']), ('Frozen selection',data['sources']['selection']), ('Canonical validation arrays',data['sources']['canonical_reference'])]
    link_html = ' · '.join(f'<a href="{html.escape(path)}">{name}</a>' for name,path in links)
    log_dir = str(Path(data['sources']['new_campaign']).parent)
    svg = figure.read_text()
    css = 'body{margin:0;background:#f3f5f6;color:#203342;font:15px/1.55 Arial,Helvetica,sans-serif}main{max-width:1320px;margin:26px auto;background:white;padding:34px 38px}h1{font-size:30px;margin:0 0 9px}h2{font-size:19px;margin:25px 0 9px}.status,.note,figcaption{color:#617582;font-size:12px}.takeaway{border-left:3px solid #598f81;background:#f0f7f5;padding:13px 16px}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:11px;line-height:1.5}th{text-align:left;background:#edf2f5;padding:8px}td{padding:7px 8px;border-bottom:1px solid #e0e7eb;font-variant-numeric:tabular-nums;white-space:nowrap}figure{margin:16px 0}figure svg{width:100%;height:auto}a{color:#2e6a83}summary{cursor:pointer;font-weight:bold}footer{margin-top:24px;font-size:11px;color:#617582}@media(max-width:700px){main{padding:23px 15px;margin:0}h1{font-size:25px}}@media print{body{background:white}main{margin:0;padding:0}tr,figure{break-inside:avoid}}'
    sections = [f'<h1>CESM2 ocean reconstruction: contribution experiments</h1><p class="status">{html.escape(status)}</p>',
        f'<p class="takeaway">{html.escape(takeaway)}</p><p class="note">{html.escape(protocol)}</p><p class="note">{html.escape(execution_note)}</p><p class="note">{html.escape(engineering_note)}</p>',
        '<figure>'+svg+f'<figcaption>Editable candidate architecture. <a href="{FIGURE}.svg">SVG</a> · <a href="{FIGURE}.png">PNG</a> · <a href="{FIGURE}.pdf">PDF</a>.</figcaption></figure>',
        '<h2>Measured validation comparison</h2>',table(headers,main_rows),f'<p class="note">{html.escape(stats)}</p>',
        '<h2>System-level paired evidence</h2>',table(['Ensemble contrast','ΔT RMSE [95% CI]','ΔS RMSE [95% CI]'],ci_rows),f'<p class="note">{html.escape(ci_note)}</p>',
        '<h2>Which contributions are supported?</h2>',table(['Candidate','Inputs','Required contrast','Current evidence'],mechanism_rows),f'<p class="note">{html.escape(mechanism_note)}</p>',
        '<details><summary>Uncertainty and computation</summary>'+table(cost_headers,cost_rows)+f'<p class="note">{html.escape(cost_note)}</p></details>',
        '<details><summary>Pending families and audit status</summary>'+table(['Campaign','Family','Inputs','Audited seeds','Status'],pending_rows)+'</details>',
        '<h2>Source robustness diagnostics</h2>'+f'<p class="note">{html.escape(stress_note)} <a href="{html.escape(robustness.get("script",""))}">Method / stress script</a> · <a href="{html.escape(robustness.get("status_source",""))}">Live source_robustness schedule status</a>.</p>'+stress_html,
        f'<h2>Development status</h2><p class="note">{html.escape(dev["status"])}</p>'+dev_table,
        f'<footer>{link_html} · <a href="{html.escape(log_dir)}">Run logs</a> · <a href="{STEM}.csv">CSV</a> · <a href="{STEM}.json">Audit JSON</a> · <a href="{STEM}.md">Markdown</a><p>目前可用：已有系统与 Local Transformer 的验证趋势。新增机制只有完成匹配对照后，才能作为实验支持的 contribution。</p></footer>']
    atomic_text(report_dir / (STEM+'.html'), '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CESM2 contribution experiments</title><style>'+css+'</style></head><body><main>'+ '\n'.join(sections)+'</main></body></html>\n')
    md = ['# CESM2 ocean reconstruction: contribution experiments','',status,'',takeaway,'',protocol,'',execution_note,'',engineering_note,'',f'![Candidate architecture]({FIGURE}.png)','',f'[Editable SVG]({FIGURE}.svg) · [PDF]({FIGURE}.pdf)','',
        '## Measured validation comparison','',md_table(headers,main_rows),'',stats,'',
        '## System-level paired evidence','',md_table(['Ensemble contrast','ΔT RMSE [95% CI]','ΔS RMSE [95% CI]'],ci_rows),'',ci_note,'',
        '## Contribution tests','',md_table(['Candidate','Inputs','Required contrast','Current evidence'],mechanism_rows),'',mechanism_note,'',
        '## Uncertainty and computation','',md_table(cost_headers,cost_rows),'',cost_note,'',
        '## Pending families','',md_table(['Campaign','Family','Inputs','Audited seeds','Status'],pending_rows),'',
        '## Source robustness diagnostics','',stress_note,'',f"[Method / stress script]({robustness.get('script','')}) · [Live source_robustness schedule status]({robustness.get('status_source','')})",'',md_table(stress_headers,stress_rows) if stress_rows else 'No completed source-robustness diagnostic is available yet.','',
        '## Development status','',dev['status'],'',dev_md,'',
        ' · '.join(f'[{name}]({path})' for name,path in links), '',f'[Run logs]({log_dir}) · [CSV]({STEM}.csv) · [Audit JSON]({STEM}.json)','']
    atomic_text(report_dir / (STEM+'.md'),'\n'.join(md))
    S.atomic_json(report_dir / (STEM+'.json'),data)
    csv_path = report_dir / (STEM+'.csv')
    temporary = csv_path.with_name(csv_path.name+'.tmp')
    with temporary.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['campaign','family','inputs','channel','aggregation','rmse','mae','bias','crps','coverage95','n'])
        for row in complete:
            for ch in S.CHANNELS:
                metrics=row.get('ensemble',row)['physical_metrics'][ch]
                writer.writerow([row['origin'],row['family'],row['mode'],ch,'ensemble' if row.get('ensemble') else 'fixed',*[metrics.get(k) for k in ('rmse','mae','mean_bias','crps','coverage_95','n')]])
                if row.get('seed_mean_sd'):
                    metric=row['seed_mean_sd'][ch]
                    writer.writerow([row['origin'],row['family'],row['mode'],ch,'seed_mean',*[metric[k]['mean'] if k in metric and isinstance(metric[k],dict) else metric.get(k) for k in ('rmse','mae','mean_bias','crps','coverage_95','n')]])
    temporary.replace(csv_path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,default=ROOT/'outputs/innovation_20261007/campaign.json')
    p.add_argument('--original-campaign',type=Path,default=ROOT/'outputs/synthetic_matched_20261007/campaign.json')
    p.add_argument('--baseline-dir',type=Path)
    p.add_argument('--selection',type=Path)
    p.add_argument('--report-dir',type=Path,default=FOLDER)
    p.add_argument('--bootstrap-draws',type=int,default=4000)
    p.add_argument('--refresh-status-only',action='store_true',help='Refresh operational/verification metadata while preserving the previously audited metrics and 4,000-draw contrasts')
    a=p.parse_args()
    old=read_json(a.original_campaign)
    baseline=a.baseline_dir or Path(old['baseline_dir'])
    selection=a.selection or a.campaign.parent/'selection.json'
    if a.refresh_status_only:
        data=read_json(a.report_dir/(STEM+'.json'))
        data.setdefault('scientific_audit_as_of_utc',data['as_of_utc'])
        data['as_of_utc']=datetime.now(timezone.utc).isoformat()
        data.update(operational_snapshot(a.campaign))
        data['sources']['reporter_sha256']=S.sha256(Path(__file__))
    else:
        data=create_snapshot(a.campaign,a.original_campaign,baseline,selection,a.bootstrap_draws)
    write_outputs(data,a.report_dir)
    print(json.dumps({'report':str(a.report_dir/(STEM+'.html')),'counts':data['counts'],'development':data['development']['status']}))


if __name__=='__main__':
    main()
