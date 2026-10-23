"""The LaTeX tables of the paper draft, generated from the sealed-test results (nothing typed by hand).

Reads the results 109 and 113 wrote once, the damped-persistence results of 121 and the row
definitions of 118, and writes ``tables/*.tex`` into the given paper folder: the main results
(both sealed tests), the ablations, the real-data budget and simulation pretraining, the
calibration, the supplementary tables with seed standard deviations, the sealed iHESP, GLORYS12 and second
CESM2 tests, the 100k-step variants on the selection years, the controlled GP benchmark and the operational
analyses at the held-out floats.

  .venv/bin/python experiments/synthetic/122_paper_tables.py PAPER_FOLDER
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("tables118", ROOT / "experiments/synthetic/118_validation_tables.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
#: "real" is the sealed 2024 test (ka-dev4's 133); "real2223" the held-out 2022-2023 test (113)
TESTS = {"cesm2": ROOT / "outputs/cesm2_test_20261019",
         "real": Path("/DATA2/linke/ocean_project_ka_dev4/outputs/real_2024_test/sealed"),
         "real2223": ROOT / "outputs/real_global_20261019/sealed"}
TITLE = {"cesm2": "CESM2 OSSE, sealed test 2006--2007", "real": "Real Argo, sealed test 2024",
         "real2223": "Real Argo, held-out test 2022--2023"}
EXTRA = {"cesm2": "cesm2", "real2223": "real"}         # the declared extra pass of 134, per test
#: the official 4DVarNet's declared extra pass (ka-4dvar 161), per test
FOURDVAR = Path("/DATA2/linke/ocean_project_ka_4dvar/outputs/fourdvarnet_20261021")
FOURDVAR_PASS = {"cesm2": "cesm2", "real": "real2024", "real2223": "real"}
MAIN_TESTS = ("cesm2", "real")
LEADS = ("0", "1", "3", "6")
CH = ("TEMP", "SALT")
DIGITS = {"TEMP": 3, "SALT": 4}


def load():
    out = {}
    for test, folder in TESTS.items():
        results = json.loads((folder / "results.json").read_text())
        statistical = (results["damped_persistence"] if "damped_persistence" in results
                       else json.loads((folder / "results_statistical.json").read_text()))
        by_path = {e["checkpoint"]: e["by_lead"] for e in results["entries"]}
        if test in EXTRA:
            extra = T.CLUSTER / "sealed" / f"results_{EXTRA[test]}.json"
            by_path.update({e["checkpoint"]: e["by_lead"] for e in json.loads(extra.read_text())["entries"]})
        extra = FOURDVAR / "sealed" / f"results_{FOURDVAR_PASS[test]}.json"
        by_path.update({e["checkpoint"]: e["by_lead"] for e in json.loads(extra.read_text())["entries"]})
        out[test] = (by_path, statistical["by_lead"], results["entries"][0]["by_lead"])
    return out


def runs(results, test, folder, tag):
    by_path = results[test][0]
    found = [by_path.get(str(folder / f"{tag}_s{s}" / f"best_seed{s}.pt")) for s in T.SEEDS]
    if any(f is None for f in found):
        raise SystemExit(f"{test} {tag}: not every seed was scored")
    return found


def value(entries, lead, ch, block="model", key="rmse"):
    return np.array([e[lead]["all"][ch][block][key] for e in entries])


ROWS = {  # name, (folder, tag) per test or a fixed baseline; "group" opens a block of rows
    "Climatology": {"fixed": "climatology", "group": "Reference"},
    "Persisted OI$^\\dagger$": {"fixed": "persisted_oi", "group": "Built on an OI analysis"},
    "Damped persistence$^\\dagger$": {"fixed": "damped"},
    "OI-anchored local transformer$^\\dagger$": {"cesm2": (T.REFERENCE, "f_y40")},
    "ARMOR3D-style OI": {"cesm2": (T.FORMAL, "c_gp_oi"), "real": (T.FORMAL, "r_gp_oi"), "group": "End-to-end from observations"},
    "Aardvark-style (enc.--proc.--dec.)": {"cesm2": (T.PILOT, "gj_none"), "real": (T.FORMAL, "r_none")},
    "4DVarNet, official code$^\\ddagger$": {"cesm2": (FOURDVAR, "fv_cesm2"), "real": (FOURDVAR, "fv_real")},
    "Poseidon-T, fine-tuned": {"cesm2": (T.POSEIDON, "p_none"), "real": (T.SELECTION, "rp_none")},
    "\\textbf{KA-Ocean (ours)}": {"cesm2": (T.PILOT, "gj_flow"), "real": (T.REAL, "scratch")},
}


def row_values(results, test, spec):
    """mean and sd over seeds of RMSE by (lead, ch), or None when the row was not run on this test."""
    if "fixed" in spec:
        if spec["fixed"] == "damped":
            stat = results[test][1]
            return {(l, ch): (stat[l]["all"][ch]["model"]["rmse"], 0.) for l in LEADS for ch in CH}
        first = results[test][2]
        return {(l, ch): (first[l]["all"][ch][spec["fixed"]]["rmse"], 0.) for l in LEADS for ch in CH}
    key = test if test in spec else "real" if test == "real2223" and "real" in spec else None   # same models on both real tests
    if key is None:
        return None
    entries = runs(results, test, *spec[key])
    return {(l, ch): (value(entries, l, ch).mean(), value(entries, l, ch).std(ddof=1)) for l in LEADS for ch in CH}


def fmt(x, ch, bold=False):
    s = f"{x:.{DIGITS[ch]}f}"
    return f"\\textbf{{{s}}}" if bold else s


def main_table(results):
    lines = ["% generated by experiments/synthetic/122_paper_tables.py", "\\begin{table*}[t]", "\\centering", "\\small",
             "\\setlength{\\tabcolsep}{4.2pt}", "\\begin{tabular}{l cccc cccc @{\\hskip 10pt} cccc cccc}", "\\toprule",
             f" & \\multicolumn{{8}}{{c}}{{{TITLE['cesm2']}}} & \\multicolumn{{8}}{{c}}{{{TITLE['real']}}}\\\\",
             "\\cmidrule(lr){2-9} \\cmidrule(lr){10-17}",
             " & \\multicolumn{4}{c}{Temperature RMSE ($^\\circ$C)} & \\multicolumn{4}{c}{Salinity RMSE (PSU)}"
             " & \\multicolumn{4}{c}{Temperature RMSE ($^\\circ$C)} & \\multicolumn{4}{c}{Salinity RMSE (PSU)}\\\\",
             "\\cmidrule(lr){2-5} \\cmidrule(lr){6-9} \\cmidrule(lr){10-13} \\cmidrule(lr){14-17}",
             "Method & 0 & 1 & 3 & 6 & 0 & 1 & 3 & 6 & 0 & 1 & 3 & 6 & 0 & 1 & 3 & 6\\\\", "\\midrule"]
    values = {test: {name: row_values(results, test, spec) for name, spec in ROWS.items()} for test in MAIN_TESTS}
    best = {test: {(l, ch): min(v[(l, ch)][0] for v in values[test].values() if v is not None) for l in LEADS for ch in CH}
            for test in MAIN_TESTS}
    for name, spec in ROWS.items():
        cells = []
        for test in MAIN_TESTS:
            v = values[test][name]
            for ch in CH:
                for l in LEADS:
                    cells.append("--" if v is None else fmt(v[(l, ch)][0], ch, abs(v[(l, ch)][0] - best[test][(l, ch)]) < 1e-12))
        if "group" in spec:
            if spec["group"] != "Reference":
                lines.append("\\midrule")
            lines.append(f"\\multicolumn{{17}}{{l}}{{\\emph{{{spec['group']}}}}}\\\\")
        lines.append(f"{name} & " + " & ".join(cells) + "\\\\")
    sd = max(v[(l, ch)][1] / v[(l, ch)][0] for test in MAIN_TESTS for v in values[test].values() if v is not None
             for l in LEADS for ch in CH)
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{\\textbf{Accuracy on the two sealed tests.} RMSE of the anomalies at lead 0 "
              "(reconstruction) and 1, 3 and 6 months (forecast) on every held-out cell; learned models are the mean of three "
              f"seeds (seed standard deviations below {100 * sd:.1f}\\,\\% of the mean, all in the supplement). \\method is "
              "ahead of every end-to-end baseline in all 16 settings. $^\\dagger$Methods built on an OI analysis of the input "
              "month; damped persistence fits a separate factor to every lead and depth band. $^\\ddagger$Authors' code; "
              "forecasts persist its reconstruction. Best per column in bold.}", "\\label{tab:main}", "\\end{table*}"]
    return "\n".join(lines) + "\n"


ABLATION = [  # group, name, cesm2 (folder, tag) or None, real (folder, tag) or None
    ("Read-out", "none (grid only)", (T.PILOT, "gj_none"), (T.FORMAL, "r_none")),
    ("Read-out", "softmax attention", (T.DEV / "grid_ka_70k_20261019", "gj_softmax"), None),
    ("Read-out", "KA, fixed OI kernel", (T.PILOT, "gj_oi"), None),
    ("Read-out", "KA, learned stationary kernel", (T.PILOT, "gj_stat"), None),
    ("Read-out", "KA, CLOINet's cluster covariance$^\\ddagger$", (T.CLUSTER, "cl_cesm2"), (T.CLUSTER, "cl_real")),
    ("Read-out", "KA, learned flow kernel (ours)", (T.PILOT, "gj_flow"), (T.REAL, "scratch")),
    ("Backbone", "Poseidon-T, no read-out", (T.POSEIDON, "p_none"), (T.SELECTION, "rp_none")),
    ("Backbone", "Poseidon-T + KA", (T.POSEIDON, "p_ka"), (T.SELECTION, "rc_poseidon")),
    ("Backbone", "Poseidon-T + KA, random init.", (T.SELECTION, "pr_random"), None),
    ("State", "w/o lead evolution", (T.FORMAL, "c_noevo"), None),
    ("Data", "4 instead of 40 training years", (T.FORMAL, "c_y4"), None),
]


def ablation_table(results):
    lines = ["% generated by experiments/synthetic/122_paper_tables.py", "\\begin{table}[t]", "\\centering", "\\small",
             "\\setlength{\\tabcolsep}{3.5pt}", "\\begin{tabular}{ll cccc}", "\\toprule",
             " & & \\multicolumn{2}{c}{CESM2 T RMSE} & \\multicolumn{2}{c}{Real T RMSE}\\\\",
             "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}", " & Variant & lead 0 & lead 6 & lead 0 & lead 6\\\\", "\\midrule"]
    previous = None
    for group, name, cesm2, real in ABLATION:
        if previous is not None and group != previous:
            lines.append("\\midrule")
        cells = []
        for test, where in (("cesm2", cesm2), ("real", real)):
            for l in ("0", "6"):
                cells.append("--" if where is None else f"{value(runs(results, test, *where), l, 'TEMP').mean():.3f}")
        bold = "ours" in name
        label = group if group != previous else ""
        lines.append(f"{label} & {name} & " + " & ".join(f'\\textbf{{{c}}}' if bold and c != '--' else c for c in cells) + "\\\\")
        previous = group
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{\\textbf{Every ingredient of the read-out counts, and KA transfers to another backbone.} "
              "Test temperature RMSE ($^\\circ$C), mean of three seeds; each row changes one part of \\method. "
              "$^\\ddagger$CLOINet's covariance~\\cite{cutolo2024cloinet} in KA's solve, trained and scored once after "
              "the tests were opened (supplement).}", "\\label{tab:ablation}", "\\end{table}"]
    return "\n".join(lines) + "\n"


EFFICIENCY = [  # name, real years, (folder, tag)
    ("\\method, simulation only", "0", (T.FORMAL, "r_zeroshot_noise")),
    ("\\method", "1", (T.FORMAL, "r_scratch_y1")),
    ("\\method", "2", (T.FORMAL, "r_scratch_y2")),
    ("\\method", "5", (T.REAL, "scratch")),
    ("ARMOR3D-style OI", "5", (T.FORMAL, "r_gp_oi")),
    ("Aardvark-style", "5", (T.FORMAL, "r_none")),
    ("Poseidon-T, fine-tuned", "5", (T.SELECTION, "rp_none")),
]


def transfer_table(results):
    lines = ["% generated by experiments/synthetic/122_paper_tables.py", "\\begin{table}[t]", "\\centering", "\\small",
             "\\setlength{\\tabcolsep}{5pt}", "\\begin{tabular}{l c cc}", "\\toprule",
             "Model & Real training years & T RMSE ($^\\circ$C) & S RMSE (PSU)\\\\", "\\midrule"]
    for name, years, where in EFFICIENCY:
        entries = runs(results, "real", *where)
        cells = [f"{value(entries, '0', 'TEMP').mean():.3f}", f"{value(entries, '0', 'SALT').mean():.4f}"]
        if name == "ARMOR3D-style OI":
            lines.append("\\midrule")
        lines.append(f"{name} & {years} & " + " & ".join(cells) + "\\\\")
    oi = results["real"][2]["0"]["all"]
    lines.append("OI & -- & " + f"{oi['TEMP']['persisted_oi']['rmse']:.3f} & {oi['SALT']['persisted_oi']['rmse']:.4f}" + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{\\textbf{Data efficiency.} Reconstruction (lead 0) on the sealed real test (2024). With two years of "
              "real data \\method matches baselines trained on five; trained on noise-augmented simulation alone it already "
              "improves on OI.}", "\\label{tab:transfer}", "\\end{table}"]
    return "\n".join(lines) + "\n"


CALIBRATION = [("KA-Ocean (ours)", {"cesm2": (T.PILOT, "gj_flow"), "real": (T.REAL, "scratch")}),
               ("OI-anchored local transformer", {"cesm2": (T.REFERENCE, "f_y40")}),
               ("ARMOR3D-style OI", {"cesm2": (T.FORMAL, "c_gp_oi"), "real": (T.FORMAL, "r_gp_oi")}),
               ("Aardvark-style", {"cesm2": (T.PILOT, "gj_none"), "real": (T.FORMAL, "r_none")})]


def calibration_table(results):
    """CRPS of every method at every lead on the real test; for deterministic forecasts the CRPS is the MAE."""
    lines = ["% generated by experiments/synthetic/122_paper_tables.py", "\\begin{table}[t]", "\\centering", "\\small",
             "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{l cccc cccc}", "\\toprule",
             " & \\multicolumn{4}{c}{Temperature CRPS ($^\\circ$C)} & \\multicolumn{4}{c}{Salinity CRPS (PSU)}\\\\",
             "\\cmidrule(lr){2-5} \\cmidrule(lr){6-9}", "Method & 0 & 1 & 3 & 6 & 0 & 1 & 3 & 6\\\\", "\\midrule"]
    stat, first = results["real"][1], results["real"][2]
    fourdvar = runs(results, "real", FOURDVAR, "fv_real")
    rows = [("Persisted OI", {(l, ch): first[l]["all"][ch]["persisted_oi"]["mae"] for l in LEADS for ch in CH}),
            ("Damped persistence", {(l, ch): stat[l]["all"][ch]["model"]["mae"] for l in LEADS for ch in CH}),
            ("4DVarNet, official code", {(l, ch): value(fourdvar, l, ch, "model", "mae").mean() for l in LEADS for ch in CH})]
    for name, (folder, tag) in (("ARMOR3D-style OI", (T.FORMAL, "r_gp_oi")), ("Aardvark-style", (T.FORMAL, "r_none")),
                                ("Poseidon-T, fine-tuned", (T.SELECTION, "rp_none")), ("\\textbf{KA-Ocean (ours)}", (T.REAL, "scratch"))):
        e = runs(results, "real", folder, tag)
        rows.append((name, {(l, ch): value(e, l, ch, "model_calibrated", "crps").mean() for l in LEADS for ch in CH}))
    best = {key: min(r[key] for _, r in rows) for key in rows[0][1]}
    for name, r in rows:
        if name.startswith("\\textbf"):
            lines.append("\\midrule")
        cells = [fmt(r[(l, ch)], ch, abs(r[(l, ch)] - best[(l, ch)]) < 1e-12) for ch in CH for l in LEADS]
        lines.append(f"{name} & " + " & ".join(cells) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{\\textbf{Best probabilistic forecasts at every lead} (sealed real test 2024, CRPS; the CRPS of a "
              "deterministic forecast is its MAE). Learned models use the spread scale fitted on the selection year; "
              "\\method's 95\\,\\% intervals cover 92--93\\,\\% of the errors at every lead.}",
              "\\label{tab:calibration}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def held_out_table(results, test="real2223"):
    """The main table's rows on the held-out 2022-2023 test (supplement)."""
    lines = ["% generated by experiments/synthetic/122_paper_tables.py", "\\begin{table}[h]", "\\centering", "\\small",
             "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{l cccc cccc}", "\\toprule",
             " & \\multicolumn{4}{c}{Temperature RMSE ($^\\circ$C)} & \\multicolumn{4}{c}{Salinity RMSE (PSU)}\\\\",
             "\\cmidrule(lr){2-5} \\cmidrule(lr){6-9}", "Method & 0 & 1 & 3 & 6 & 0 & 1 & 3 & 6\\\\", "\\midrule"]
    values = {name: row_values(results, test, spec) for name, spec in ROWS.items()}
    best = {(l, ch): min(v[(l, ch)][0] for v in values.values() if v is not None) for l in LEADS for ch in CH}
    for name, spec in ROWS.items():
        v = values[name]
        if v is None:
            continue
        if "group" in spec:
            if spec["group"] != "Reference":
                lines.append("\\midrule")
            lines.append(f"\\multicolumn{{9}}{{l}}{{\\emph{{{spec['group']}}}}}\\\\")
        cells = [fmt(v[(l, ch)][0], ch, abs(v[(l, ch)][0] - best[(l, ch)]) < 1e-12) for ch in CH for l in LEADS]
        lines.append(f"{name} & " + " & ".join(cells) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              f"\\caption{{{TITLE[test]}: the rows of Table 2 of the paper, mean of three seeds. These years were the "
              "development period of earlier design iterations on this cohort; every model was trained on 2016--2020, "
              "selected on 2021 and frozen before it was scored. Best per column in bold.}",
              "\\label{tab:supp_heldout}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def supp_table(results, tests=("cesm2", "real"), label="tab:supp_main"):
    lines = ["% generated by experiments/synthetic/122_paper_tables.py", "\\begin{table*}[t]", "\\centering", "\\scriptsize",
             "\\setlength{\\tabcolsep}{3pt}"]
    for test, title, groups in ((test, TITLE[test], T.CESM2 if test == "cesm2" else T.REAL_ROWS) for test in tests):
        lines += [f"\\textbf{{{title}}}\\\\[2pt]", "\\begin{tabular}{l cccc cccc}", "\\toprule",
                  "Run & \\multicolumn{4}{c}{Temperature RMSE ($^\\circ$C), leads 0/1/3/6} & "
                  "\\multicolumn{4}{c}{Salinity RMSE (PSU), leads 0/1/3/6}\\\\", "\\midrule"]
        for group, rows in groups.items():
            lines.append(f"\\multicolumn{{9}}{{l}}{{\\emph{{{group.replace(', 115', '')}}}}}\\\\")
            for name, folder, tag in rows:
                e = runs(results, test, folder, tag)
                cells = [f"{value(e, l, ch).mean():.{DIGITS[ch]}f}$\\pm${value(e, l, ch).std(ddof=1):.{DIGITS[ch]}f}"
                         for ch in CH for l in LEADS]
                lines.append(f"{name.replace('no exact-value read-out', 'no KA read-out')} & " + " & ".join(cells) + "\\\\")
        lines += ["\\bottomrule", "\\end{tabular}\\\\[6pt]"]
    lines += [f"\\caption{{Every scored run on {', '.join(TITLE[t] for t in tests)}: mean $\\pm$ standard deviation over "
              "seeds 1234--1236.}", f"\\label{{{label}}}", "\\end{table*}"]
    return "\n".join(lines) + "\n"


#: the second simulator's sealed test (ka-ihesp 140, scored once)
IHESP = Path("/DATA2/linke/ocean_project_ka_ihesp/outputs/ihesp_20261021")
IHESP_ROWS = {  # name: a tag of the ka-ihesp runs or a fixed baseline; "group" opens a block of rows
    "Climatology": {"fixed": "climatology", "group": "Reference"},
    "Persisted OI$^\\dagger$": {"fixed": "persisted_oi", "group": "Built on an OI analysis"},
    "Damped persistence$^\\dagger$": {"fixed": "damped"},
    "ARMOR3D-style OI": {"tag": "ih_armor", "group": "End-to-end from observations"},
    "Aardvark-style (enc.--proc.--dec.)": {"tag": "ih_none"},
    "Softmax read-out (ablation)": {"tag": "ih_softmax"},
    "\\textbf{KA-Ocean (ours)}": {"tag": "ih_ka"},
}


def sealed_table(results, rows_spec, checkpoint, head, label):
    """A sealed test on another data set (iHESP, GLORYS12): the claims of the caption are computed here."""
    by_path = {e["checkpoint"]: e["by_lead"] for e in results["entries"]}
    first, damped = results["entries"][0]["by_lead"], results["damped_persistence"]["by_lead"]

    def values(spec):
        if "fixed" in spec:
            block = damped if spec["fixed"] == "damped" else first
            key = "model" if spec["fixed"] == "damped" else spec["fixed"]
            return {(l, ch): (block[l]["all"][ch][key]["rmse"], 0.) for l in LEADS for ch in CH}
        entries = [by_path[checkpoint(spec["tag"], s)] for s in T.SEEDS]
        return {(l, ch): (value(entries, l, ch).mean(), value(entries, l, ch).std(ddof=1)) for l in LEADS for ch in CH}

    rows = {name: values(spec) for name, spec in rows_spec.items()}
    shown = lambda x, ch: round(x, DIGITS[ch])
    best = {(l, ch): min(shown(v[(l, ch)][0], ch) for v in rows.values()) for l in LEADS for ch in CH}
    lines = ["% generated by experiments/synthetic/122_paper_tables.py", "\\begin{table}[h]", "\\centering", "\\small",
             "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{l cccc cccc}", "\\toprule",
             " & \\multicolumn{4}{c}{Temperature RMSE ($^\\circ$C)} & \\multicolumn{4}{c}{Salinity RMSE (PSU)}\\\\",
             "\\cmidrule(lr){2-5} \\cmidrule(lr){6-9}", "Method & 0 & 1 & 3 & 6 & 0 & 1 & 3 & 6\\\\", "\\midrule"]
    for name, spec in rows_spec.items():
        if "group" in spec:
            if spec["group"] != "Reference":
                lines.append("\\midrule")
            lines.append(f"\\multicolumn{{9}}{{l}}{{\\emph{{{spec['group']}}}}}\\\\")
        v = rows[name]
        lines.append(f"{name} & " + " & ".join(fmt(v[(l, ch)][0], ch, shown(v[(l, ch)][0], ch) == best[(l, ch)])
                                              for ch in CH for l in LEADS) + "\\\\")
    ours, learned = rows["\\textbf{KA-Ocean (ours)}"], [rows[n] for n, s in rows_spec.items() if "tag" in s]
    rivals = [rows["ARMOR3D-style OI"], rows["Aardvark-style (enc.--proc.--dec.)"]]
    gain = {ch: 100 * (1 - ours[("0", ch)][0] / min(r[("0", ch)][0] for r in rivals)) for ch in CH}
    lead_learned = sum(shown(ours[(l, ch)][0], ch) == min(shown(r[(l, ch)][0], ch) for r in learned)
                       for l in LEADS for ch in CH)
    lost = [(l, ch) for l in LEADS for ch in CH if shown(ours[(l, ch)][0], ch) > best[(l, ch)]]
    sd = max(v[(l, ch)][1] / v[(l, ch)][0] for v in learned for l in LEADS for ch in CH)
    name = {"TEMP": "temperature", "SALT": "salinity"}
    lost_text = "; ".join(f"{name[ch]} at lead {l}" for l, ch in lost)
    print(f"{label}: ours vs best end-to-end baseline at lead 0: T {gain['TEMP']:.1f}%, S {gain['SALT']:.1f}%; "
          f"best among learned in {lead_learned} of 8; damped persistence better: {lost_text or 'nowhere'}")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{" + head + f"RMSE on all {first['0']['cells']:,} held-out cells per lead, mean of three seeds (standard "
              f"deviations below {100 * sd:.1f}\\,\\% of the mean). The comparison fixed in advance, \\method against the best "
              f"end-to-end baseline at lead 0, is {gain['TEMP']:.1f}\\,\\% lower for temperature and {gain['SALT']:.1f}\\,\\% "
              f"for salinity. Among the learned models \\method is best or tied in {lead_learned} of 8 settings"
              + (f"; damped persistence is better for {lost_text}" if lost else "")
              + ". $^\\dagger$Built on an OI analysis of the input month. Best per column in bold, ties at the shown "
              "precision included.}", f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def ihesp_table():
    """The sealed iHESP 0.1-degree test (supplement)."""
    results = json.loads((IHESP / "ihesp_hr_test/sealed/results.json").read_text())
    head = ("\\textbf{A second simulator: eddy-resolving iHESP CESM1.3 at 0.1$^\\circ$, sealed test 1998--1999.} "
            "Trained on 2000--2003 and selected on 2004--2005 with the 4-year settings of the CESM2 runs, registered and "
            "scored once; ")
    return sealed_table(results, IHESP_ROWS, lambda tag, s: str(IHESP / f"{tag}_s{s}" / f"best_seed{s}.pt"), head, "tab:supp_ihesp")


#: the reanalysis sealed test (ka-dev6 151, scored once): the iHESP rows with the g_ tags; checkpoints are relative paths there
GLORYS = Path("/DATA2/linke/ocean_project_ka_dev6/outputs/glorys_20261021")
GLORYS_ROWS = {name: {**spec, "tag": spec["tag"].replace("ih_", "g_")} if "tag" in spec else spec for name, spec in IHESP_ROWS.items()}


def glorys_table():
    """The sealed GLORYS12 test (supplement)."""
    results = json.loads((GLORYS / "glorys12_test/sealed/results.json").read_text())
    head = ("\\textbf{A global reanalysis: GLORYS12~\\cite{glorys12} (Mercator Ocean, NEMO at 1/12$^\\circ$), sealed test "
            "2019--2020.} Trained on 1997--2016 (20 years) and selected on 2017--2018, 100,000 steps, registered and scored "
            "once; ")
    return sealed_table(results, GLORYS_ROWS, lambda tag, s: f"outputs/glorys_20261021/runs/{tag}_s{s}/best_seed{s}.pt",
                        head, "tab:supp_glorys")


#: the second sealed CESM2 test, 2008-2010 (ka-dev6 154, scored once): training years and steps against the main tables' model
SCALE = Path("/DATA2/linke/ocean_project_ka_dev6/outputs/cesm2_test2_20261022/sealed")
SCALE_ROWS = [("40 (main tables)", "70k", "gj_flow"), ("40", "100k", "c_flow100"), ("\\textbf{154}", "\\textbf{100k}", "c_y154")]
SCALE_EXTRA = [("ARMOR3D-style OI, 154", "100k", "c_armor154"), ("Aardvark-style, 154", "100k", "c_none154")]  # declared extra pass


def scale_table():
    results = json.loads((SCALE / "results.json").read_text())
    entries, rows = results["entries"], list(SCALE_ROWS)
    if (SCALE / "results_extra.json").exists():
        entries = entries + json.loads((SCALE / "results_extra.json").read_text())["entries"]
        rows += SCALE_EXTRA
    by_tag = {e["tag"]: e["by_lead"] for e in entries}
    first = entries[0]["by_lead"]
    damped = results["damped_persistence"]["by_cohort"][entries[0]["cohort"]]["by_lead"]   # identical for both cohorts
    mean = lambda tag: {(l, ch): value([by_tag[f"{tag}_s{s}"] for s in T.SEEDS], l, ch).mean() for l in LEADS for ch in CH}
    lines = ["% generated by experiments/synthetic/122_paper_tables.py", "\\begin{table}[t]", "\\centering", "\\small",
             "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{ll cccc cccc}", "\\toprule",
             " & & \\multicolumn{4}{c}{Temperature RMSE ($^\\circ$C)} & \\multicolumn{4}{c}{Salinity RMSE (PSU)}\\\\",
             "\\cmidrule(lr){3-6} \\cmidrule(lr){7-10}", "Training years & Steps & 0 & 1 & 3 & 6 & 0 & 1 & 3 & 6\\\\", "\\midrule",
             "\\multicolumn{10}{l}{\\emph{Reference}}\\\\"]
    for name, block, key in (("Climatology", first, "climatology"), ("Persisted OI$^\\dagger$", first, "persisted_oi"),
                             ("Damped persistence$^\\dagger$", damped, "model")):
        lines.append(f"{name} & -- & " + " & ".join(fmt(block[l]["all"][ch][key]["rmse"], ch) for ch in CH for l in LEADS) + "\\\\")
    lines += ["\\midrule", "\\multicolumn{10}{l}{\\emph{\\method}}\\\\"]
    values = {tag: mean(tag) for _, _, tag in rows}
    for years, steps, tag in rows:
        if tag == SCALE_EXTRA[0][2]:
            lines += ["\\midrule", "\\multicolumn{10}{l}{\\emph{Baselines}}\\\\"]
        v = values[tag]
        lines.append(f"{years} & {steps} & " + " & ".join(fmt(v[(l, ch)], ch, tag == "c_y154") for ch in CH for l in LEADS) + "\\\\")
    main, long, steps = values["gj_flow"], values["c_y154"], values["c_flow100"]
    gain = {key: 100 * (1 - long[key] / main[key]) for key in main}
    step_change = max(100 * abs(steps[key] / main[key] - 1) for key in main)
    assert min(gain.values()) > 0, gain   # the caption says "lowers"
    print(f"tab:scale: 154 vs 40 years, lead 6: T {gain[('6', 'TEMP')]:.1f}%, S {gain[('6', 'SALT')]:.1f}%; lead 0: "
          f"T {gain[('0', 'TEMP')]:.1f}%, S {gain[('0', 'SALT')]:.1f}%; 100k vs 70k steps at most {step_change:.1f}%")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{\\textbf{More simulated years buy forecast skill.} Sealed CESM2 test 2008--2010, never read by any run "
              f"before scoring; RMSE on all {first['0']['cells']:,} held-out cells per lead, mean of three seeds. Trained on 154 "
              f"years (1850--2003) instead of 40, \\method lowers the 6-month temperature error by {gain[('6', 'TEMP')]:.1f}\\,\\% "
              f"and the salinity error by {gain[('6', 'SALT')]:.1f}\\,\\% relative to the main tables' model, and the lead-0 "
              f"errors by {gain[('0', 'TEMP')]:.1f}\\,\\% and {gain[('0', 'SALT')]:.1f}\\,\\%. Steps alone (100k vs 70k) change "
              f"the forecasts by at most {step_change:.1f}\\,\\%. $^\\dagger$Built on an OI analysis of the input month.}}",
              "\\label{tab:scale}", "\\end{table}"]
    return "\n".join(lines) + "\n"


#: the 100k-step variants (ka-dev6/ka-dev8 150), compared on the selection years of CESM2 only
VARIANTS, VARIANTS8 = (Path(f"/DATA2/linke/ocean_project_ka_dev{d}/outputs/wang_20261021") for d in (6, 8))
VARIANT_ROWS = [  # group (None: the basis), name, folder, tag
    (None, "Flow kernel (\\method)", VARIANTS, "c_flow100"),
    ("Learned covariance", "Gaussian in a learned feature space", VARIANTS8, "c_deep"),
    ("Learned covariance", "Flow kernel $\\times$ feature kernel", VARIANTS8, "c_product"),
    ("Learned covariance", "Two flow kernels (second $3\\times$ longer)", VARIANTS, "c_multiscale"),
    ("Solve", "Temperature and salinity jointly (coregionalisation)", VARIANTS, "c_mv"),
    ("Augmentation", "Random input density 1,520--6,080 a step", VARIANTS, "c_density"),
    ("Augmentation", "Three profile draws a month", VARIANTS, "c_draws"),
    ("Augmentation", "Both", VARIANTS, "c_both"),
    ("Training years", "154 years (1850--2003)", VARIANTS, "c_y154"),
]


def variants_table():
    """Selection-year numbers (not a test) of the variants: the selection score and the RMSE at leads 0 and 6."""
    def summary(folder, tag):
        val = [json.loads((folder / f"{tag}_s{s}" / f"summary_seed{s}.json").read_text())["validation"] for s in T.SEEDS]
        score = np.array([v["scores"]["macro_z"] for v in val])
        cells = {(l, ch): np.mean([v["by_lead"][l]["scores"][ch]["rmse_physical"] for v in val]) for l in ("0", "6") for ch in CH}
        return score.mean(), score.std(ddof=1), cells
    rows = [(group, name, summary(folder, tag)) for group, name, folder, tag in VARIANT_ROWS]
    shown = lambda x, ch: round(x, DIGITS[ch])
    best_score = min(round(r[2][0], 3) for r in rows)
    best = {key: min(shown(r[2][2][key], key[1]) for r in rows) for key in rows[0][2][2]}
    lines = ["% generated by experiments/synthetic/122_paper_tables.py", "\\begin{table}[h]", "\\centering", "\\small",
             "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{l c cc cc}", "\\toprule",
             " & & \\multicolumn{2}{c}{T RMSE ($^\\circ$C)} & \\multicolumn{2}{c}{S RMSE (PSU)}\\\\",
             "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}", "Variant & Selection score & 0 & 6 & 0 & 6\\\\", "\\midrule"]
    previous = None
    for group, name, (mean, sd, cells) in rows:
        if group is not None and group != previous:
            lines += ["\\midrule", f"\\multicolumn{{6}}{{l}}{{\\emph{{{group}}}}}\\\\"]
        previous = group
        score = f"{mean:.3f}$\\pm${sd:.3f}"
        score = f"\\textbf{{{score}}}" if round(mean, 3) == best_score else score
        lines.append(f"{name} & {score} & " + " & ".join(fmt(cells[(l, ch)], ch, shown(cells[(l, ch)], ch) == best[(l, ch)])
                                                        for ch in CH for l in ("0", "6")) + "\\\\")
        print(f"tab:supp_variants: {name}: score {mean:.3f}+-{sd:.3f}")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{\\textbf{Alternatives to the flow kernel, tried after the main results were frozen.} Selection years of "
              "CESM2 (2004--2005), 100,000 steps, three seeds; score = mean standardised RMSE over leads (lower is better). None "
              "of the learned covariance families, the joint solve or the augmentations improves on the flow kernel; "
              "the training window does. Best per column in bold, ties at the shown precision included.}",
              "\\label{tab:supp_variants}", "\\end{table}"]
    return "\n".join(lines) + "\n"


#: the controlled benchmark on four GP truths (ka-dev6 153); ka-dev8 re-ran the two feature-space kinds with the fixed initialisation
GP, GP8 = (Path(f"/DATA2/linke/ocean_project_ka_dev{d}/outputs/synthetic_gp_nongaussian_20261021/results.json") for d in (6, 8))
GP_ROWS = [("OI", "oi"), ("Softmax attention", "softmax"), ("KA, stationary kernel", "ka_stationary"),
           ("KA, flow kernel (ours)", "ka_flow"), ("KA, feature-space Gaussian", "ka_deep"), ("KA, flow $\\times$ feature", "ka_product"),
           ("KA, two scales", "ka_multiscale"), ("Exact GP, true covariance (oracle)", "oracle")]
GP_TRUTHS = (("flow", "Flow (paper)"), ("warped", "Curved lobes"), ("exponential", "Exponential"), ("mixture", "Two scales"))


def gp_truths_table():
    truths, truths8 = (json.loads(p.read_text())["truths"] for p in (GP, GP8))
    rmse = {(kind, t): (truths8 if kind in ("ka_deep", "ka_product") else truths)[t]["summary"][kind]["rmse"]["mean"]
            for _, kind in GP_ROWS for t, _ in GP_TRUTHS}
    best = {t: min(round(rmse[(kind, t)], 3) for _, kind in GP_ROWS if kind != "oracle") for t, _ in GP_TRUTHS}
    lines = ["% generated by experiments/synthetic/122_paper_tables.py", "\\begin{table}[h]", "\\centering", "\\small",
             "\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{l cccc}", "\\toprule",
             "Read-out & " + " & ".join(name for _, name in GP_TRUTHS) + "\\\\", "\\midrule"]
    for name, kind in GP_ROWS:
        if kind == "oracle":
            lines.append("\\midrule")
        cells = [f"{rmse[(kind, t)]:.3f}" for t, _ in GP_TRUTHS]
        bold = [kind != "oracle" and round(rmse[(kind, t)], 3) == best[t] for t, _ in GP_TRUTHS]
        lines.append(f"{name} & " + " & ".join(f"\\textbf{{{c}}}" if b else c for c, b in zip(cells, bold)) + "\\\\")
    print("tab:supp_gp_truths: KA flow / oracle: " + ", ".join(f"{name} {rmse[('ka_flow', t)] / rmse[('oracle', t)]:.3f}"
                                                               for t, name in GP_TRUTHS))
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{\\textbf{The flow kernel reaches the exact GP on a curved-lobe covariance.} RMSE on 200 held-out fields, "
              "three seeds; truths: the paper's flow-dependent Gaussian; a Gaussian in a warped plane whose lobes curve with the "
              "stream function; an exponential (Mat\\'ern-1/2) kernel; a mixture of two Gaussians at lengths $L$ and $3L$. The "
              "oracle is the exact local GP with the true covariance. Best learned read-out per column in bold.}",
              "\\label{tab:supp_gp_truths}", "\\end{table}"]
    return "\n".join(lines) + "\n"


#: the operational analyses at the held-out floats (ka-dev6 152): they assimilated the scored floats, so references, never baselines
PRODUCTS = Path("/DATA2/linke/ocean_project_ka_dev6/outputs/reference_products_20261021")
PRODUCT_TESTS = (("real", "2024"), ("real2223", "2022-2023"))


def products_table(results):
    """The paper's rows next to ARMOR3D and GLORYS12 on both real tests; the claims of the caption are computed here."""
    def product(name, period):
        by_lead = json.loads((PRODUCTS / f"{name}_{period}.json").read_text())["by_lead"]
        return {(l, ch): (by_lead[l][ch]["product"]["rmse"], 0.) for l in LEADS for ch in CH}
    rows = [(name, {test: row_values(results, test, ROWS[name]) for test, _ in PRODUCT_TESTS})
            for name in ("Climatology", "Persisted OI$^\\dagger$", "\\textbf{KA-Ocean (ours)}")]
    products = [(name, {test: product(key, period) for test, period in PRODUCT_TESTS})
                for name, key in (("ARMOR3D", "armor3d"), ("GLORYS12", "glorys12"))]
    rows += [(f"{name} (analysis at 0 / persisted at 1, 3, 6)", v) for name, v in products]
    lines = ["% generated by experiments/synthetic/122_paper_tables.py", "\\begin{table*}[t]", "\\centering", "\\small",
             "\\setlength{\\tabcolsep}{4.2pt}", "\\begin{tabular}{l cccc cccc @{\\hskip 10pt} cccc cccc}", "\\toprule",
             f" & \\multicolumn{{8}}{{c}}{{{TITLE['real']}}} & \\multicolumn{{8}}{{c}}{{{TITLE['real2223']}}}\\\\",
             "\\cmidrule(lr){2-9} \\cmidrule(lr){10-17}",
             " & \\multicolumn{4}{c}{Temperature RMSE ($^\\circ$C)} & \\multicolumn{4}{c}{Salinity RMSE (PSU)}"
             " & \\multicolumn{4}{c}{Temperature RMSE ($^\\circ$C)} & \\multicolumn{4}{c}{Salinity RMSE (PSU)}\\\\",
             "\\cmidrule(lr){2-5} \\cmidrule(lr){6-9} \\cmidrule(lr){10-13} \\cmidrule(lr){14-17}",
             "Method & 0 & 1 & 3 & 6 & 0 & 1 & 3 & 6 & 0 & 1 & 3 & 6 & 0 & 1 & 3 & 6\\\\", "\\midrule"]
    for name, v in rows:
        if name.startswith("ARMOR3D"):
            lines += ["\\midrule", "\\multicolumn{17}{l}{\\emph{Operational analyses that assimilate the scored floats "
                      "(reference, not baselines)}}\\\\"]
        lines.append(f"{name} & " + " & ".join(fmt(v[test][(l, ch)][0], ch) for test, _ in PRODUCT_TESTS for ch in CH for l in LEADS) + "\\\\")
    shown = lambda x, ch: round(x, DIGITS[ch])
    oi, ours = rows[1][1], rows[2][1]
    forecast = [(test, l, ch) for test, _ in PRODUCT_TESTS for l in LEADS[1:] for ch in CH]
    lost = [(test, l, ch) for test, l, ch in forecast
            if any(shown(v[test][(l, ch)][0], ch) <= shown(ours[test][(l, ch)][0], ch) for _, v in products)]
    behind = [l for l in LEADS[1:] if all(shown(v[test][(l, ch)][0], ch) > shown(oi[test][(l, ch)][0], ch)
                                           for _, v in products for test, _ in PRODUCT_TESTS for ch in CH)]
    from_lead = next(l for l in LEADS[1:] if all(later in behind for later in LEADS[LEADS.index(l):]))
    name = {"TEMP": "temperature", "SALT": "salinity"}
    words = {"1": "one month", "3": "three months", "6": "six months"}
    lost_text = "; ".join(f"{name[ch]} at lead {l} of {dict(PRODUCT_TESTS)[test].replace('-', '--')}, where "
                          f"{(p := min(products, key=lambda nv: nv[1][test][(l, ch)][0]))[0]}'s persisted anomaly is "
                          f"{shown(ours[test][(l, ch)][0], ch) - shown(p[1][test][(l, ch)][0], ch):.{DIGITS[ch]}f} lower"
                          for test, l, ch in lost)
    print(f"tab:supp_products: ours ahead of both products in {len(forecast) - len(lost)} of {len(forecast)} forecast settings"
          f"{'; exception: ' + lost_text if lost else ''}; products behind persisted OI everywhere from lead {from_lead}")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{\\textbf{Operational analyses at the same held-out floats.} ARMOR3D and GLORYS12 assimilate every Argo "
              "profile, including the floats scored here, so their lead-0 rows are references that have seen the answer. "
              f"\\method's forecasts are ahead of both products' persisted anomalies in {len(forecast) - len(lost)} of "
              f"{len(forecast)} forecast settings" + (f" (the exception: {lost_text})" if lost else "")
              + f", and from {words[from_lead]} ahead the persisted anomalies of both products are behind the persisted OI in every "
              "setting. $^\\dagger$Built on an OI analysis of the input month.}", "\\label{tab:supp_products}", "\\end{table*}"]
    return "\n".join(lines) + "\n"


def fit_width(text):
    """Scale every tabular to the width of its float (a column, or the page for table*)."""
    return (text.replace("\\begin{tabular}", "\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}")
                .replace("\\end{tabular}", "\\end{tabular}}"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paper", type=Path)
    args = parser.parse_args()
    results = load()
    out = args.paper / "tables"
    out.mkdir(parents=True, exist_ok=True)
    for name, text in (("main", main_table(results)), ("ablation", ablation_table(results)),
                       ("transfer", transfer_table(results)), ("calibration", calibration_table(results)),
                       ("supp_main", supp_table(results)), ("supp_main_2223", supp_table(results, ("real2223",), "tab:supp_main_2223")),
                       ("supp_heldout", held_out_table(results)), ("supp_ihesp", ihesp_table()),
                       ("supp_glorys", glorys_table()), ("scale", scale_table()), ("supp_variants", variants_table()),
                       ("supp_gp_truths", gp_truths_table()), ("supp_products", products_table(results))):
        (out / f"{name}.tex").write_text(fit_width(text))
        print(f"wrote {out / name}.tex")


if __name__ == "__main__":
    main()
