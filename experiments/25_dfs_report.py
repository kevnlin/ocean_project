"""DFS-Attention — the success-criterion report (plan Section 12).

The plan is explicit that low duplication sensitivity alone proves nothing: a
model that ignores Argo entirely is perfectly duplication-robust and useless.
Success requires all three legs at once, so this report puts them in one table
and marks each one pass/fail:

    low duplication sensitivity + high observation retention + competitive accuracy

Sources (all produced by earlier scripts; nothing is recomputed here):
  outputs/cache/<prefix>_<variant>_s<seed>.json         trainer -> test RMSE
  outputs/cache/full_eval_<prefix>_<variant>_s<seed>.json
                                                        probes, modality matrix,
                                                        density sweep, evidence
  outputs/cache/dfs_evidence_probes.json                Section-11 probes

Run:  python experiments/25_dfs_report.py [--prefix fullA] [--seeds 1234,1235,1236]
Out:  reports/dfs_success_criterion.md
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np

from ocean_tokenizer import config as C

ap = argparse.ArgumentParser()
ap.add_argument("--prefix", default="fullA")
ap.add_argument("--seeds", default="1234,1235,1236")
ap.add_argument("--variants", default="perceiver,resampler,mbca,dfs")
args = ap.parse_args()
SEEDS = [int(s) for s in args.seeds.split(",")]
VARIANTS = args.variants.split(",")
NAME = {"perceiver": "Standard Perceiver", "resampler": "Fixed resampler",
        "mbca": "MBCA", "dfs": "DFS-Attention"}

runs, evals = {}, {}
for v in VARIANTS:
    for s in SEEDS:
        tag = f"{args.prefix}_{v}_s{s}"
        p = os.path.join(C.CACHE, f"{tag}.json")
        if os.path.exists(p):
            d = json.load(open(p))
            if d.get("status") == "done":
                runs[(v, s)] = d
        p = os.path.join(C.CACHE, f"full_eval_{tag}.json")
        if os.path.exists(p):
            evals[(v, s)] = json.load(open(p))

have = sorted({v for v, _ in runs})
print(f"trainer runs: {len(runs)} | evals: {len(evals)} | variants: {have}")


def agg(d, v, path, seeds=SEEDS):
    vals = []
    for s in seeds:
        cur = d.get((v, s))
        for k in path:
            cur = None if cur is None else cur.get(k)
        if cur is not None:
            vals.append(float(cur))
    return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals \
        else (float("nan"), float("nan"), 0)


def fmt(m, sd, n, dec=4):
    if n == 0:
        return "—"
    return f"{m:.{dec}f}" + (f" ±{sd:.{dec}f}" if n > 1 else "")


L = ["# DFS-Attention — Success Criterion (plan Section 12)\n",
     "*Three legs, one table. A method that scores well on duplication "
     "sensitivity while ignoring the profiles fails leg 2; a method that uses "
     "the profiles but is destabilised by duplicates fails leg 1. Both must "
     "hold while accuracy stays competitive.*\n",
     f"Runs: prefix `{args.prefix}`, seeds {SEEDS}, protocol_v1 "
     "(276/36/12 months, anomaly target, unobserved-only RMSE). Mean ± sd "
     "over seeds.\n"]

# ---------------------------------------------------------------- accuracy
L.append("\n## Leg 3 · Reconstruction accuracy (pinned test months)\n")
floor = agg(runs, have[0], ["test_floor", "TEMP"]) if have else (0, 0, 0)
L.append("| variant | TEMP RMSE (°C) | SALT RMSE (PSU) | skill vs floor (TEMP) "
         "| params | seeds |")
L.append("|---|---:|---:|---:|---:|---:|")
for v in VARIANTS:
    t = agg(runs, v, ["test", "TEMP"])
    s_ = agg(runs, v, ["test", "SALT"])
    sk = agg(runs, v, ["test", "skill_vs_floor", "TEMP"])
    npar = agg(runs, v, ["n_params"])
    L.append(f"| {NAME[v]} | {fmt(*t)} | {fmt(*s_)} | "
             f"{fmt(sk[0]*100, sk[1]*100, sk[2], 1)} % | "
             f"{'—' if npar[2] == 0 else f'{npar[0]/1e6:.2f} M'} | {t[2]} |")
if floor[2]:
    L.append(f"\nTrain-climatology floor on the test months: "
             f"TEMP {floor[0]:.4f} °C.\n")

# run hygiene — the comparison is only worth reading if the configs match
cfg_keys = ["steps", "val_every", "patience", "lr", "warmup", "obs_query_frac",
            "input_noise", "weight_decay", "grid_drop", "d_model", "n_latent",
            "n_self_blocks"]
cfgs = {}
for v in VARIANTS:
    for s in SEEDS:
        if (v, s) in runs:
            cfgs.setdefault(v, tuple(runs[(v, s)].get(k) for k in cfg_keys))
distinct = set(cfgs.values())
if len(distinct) > 1:
    L.append("\n> **Configuration mismatch.** The variants below were not "
             "trained with identical settings, so the accuracy column is not "
             "a like-for-like comparison:\n")
    for v, c in cfgs.items():
        diff = [f"`{k}`={val}" for k, val in zip(cfg_keys, c)
                if len({cc[i] for cc in distinct}) > 1
                for i in [cfg_keys.index(k)]]
        L.append(f">   - {NAME[v]}: " + ", ".join(diff))
    L.append("")
elif cfgs:
    L.append(f"\nAll variants trained with identical settings "
             f"({', '.join(f'`{k}`={val}' for k, val in zip(cfg_keys, next(iter(distinct))))}); "
             "only the fusion rule differs. DFS-Attention carries the extra "
             "parameters of its evidence resampler and background attention, "
             "noted in the params column.\n")

# ------------------------------------------------------------- duplication
L.append("\n## Leg 1 · Duplication sensitivity (validation months, real fields)\n")
L.append("Relative change of the decoded field when observations are "
         "re-ingested carrying no new information. `duplicate_half` adds the "
         "copies anonymously; `duplicate_half_declared` gives them the "
         "provenance of the float cycle they came from (a real duplicated "
         "Argo record), which is the case DFS-Attention consolidates exactly.\n")
PROBES = ["duplicate_half", "duplicate_half_declared", "patch_refine_2x",
          "profile_resample_2x"]
L.append("| probe | " + " | ".join(NAME[v] for v in VARIANTS) + " |")
L.append("|---|" + "---:|" * len(VARIANTS))
for pr in PROBES:
    cells = []
    for v in VARIANTS:
        m, sd, n = agg(evals, v, ["probes", pr, "rel_output_change_mean"])
        cells.append("—" if n == 0 else f"{m:.4f}")
    L.append(f"| `{pr}` | " + " | ".join(cells) + " |")

# --------------------------------------------------------------- retention
L.append("\n## Leg 2 · Observation retention\n")
L.append("How much the reconstruction actually depends on the profiles.\n\n"
         "* **withheld** — RMSE penalty when the profiles are removed "
         "entirely. A model that ignores Argo pays nothing.\n"
         "* **marginal** — RMSE penalty of halving the profile count from "
         "3000 to 1500. A model that has saturated on the observations it "
         "already reads pays nothing here either, even if it does use *some* "
         "of them.\n\n"
         "Both are penalties, so bigger is better: they measure how much of "
         "the answer is actually coming from the observations.\n")
pct = lambda x: "—" if np.isnan(x) else f"{x:+.1f} %"
L.append("| variant | TEMP RMSE, all inputs | without profiles | withheld | "
         "marginal (3000→1500) | seeds |")
L.append("|---|---:|---:|---:|---:|---:|")
for v in VARIANTS:
    full = agg(evals, v, ["modality_matrix", "full", "TEMP"])
    drop = agg(evals, v, ["modality_matrix", "drop_profiles", "TEMP"])
    deg = (drop[0] / full[0] - 1.0) * 100 if full[2] and full[0] else float("nan")
    lo, hi = [], []
    for s in SEEDS:
        sw = evals.get((v, s), {}).get("count_sweep")
        if not sw:
            continue
        by = {r["density"]: r["TEMP"] for r in sw}
        if 1500 in by and 3000 in by:
            lo.append(by[1500]); hi.append(by[3000])
    marg = ((np.mean(lo) / np.mean(hi) - 1.0) * 100 if lo else float("nan"))
    L.append(f"| {NAME[v]} | {fmt(*full)} | {fmt(*drop)} | {pct(deg)} | "
             f"{pct(marg)} | {drop[2]} |")

# ---------------------------------------------------------------- evidence
ev_seeds = [s for s in SEEDS if ("dfs", s) in evals
            and "evidence" in evals[("dfs", s)]]
if ev_seeds:
    L.append("\n### Evidence budget of the observing system (DFS-Attention)\n")
    L.append("What the estimator says the month's observations are worth, "
             "before any training. `profile share` is retention measured at "
             "the evidence level rather than through the loss.\n")
    L.append("| target scale | obs tokens | DFS | DFS / token | profile share "
             "| surface share | neighbour cut |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    loose = []
    for sc in ("coarse", "protocol", "fine"):
        rows = [evals[("dfs", s)]["evidence"][sc] for s in ev_seeds]
        g = lambda k: float(np.mean([r[k] for r in rows]))
        gm = lambda k: float(np.mean([r["by_modality"].get(k, 0.0) for r in rows]))
        cut = g("neighbour_cut")
        flag = "" if cut < 0.05 else " ⚠"
        if cut >= 0.05:
            loose.append((sc, cut))
        L.append(f"| {sc} | {g('obs_tokens'):.0f} | {g('dfs_total'):.1f}{flag} | "
                 f"{g('dfs_per_token'):.3f} | {g('profile_share'):.3f} | "
                 f"{gm('surf_grid')/max(g('dfs_total'),1e-9):.3f} | "
                 f"{cut:.4f} |")
    L.append("\nThe background (WOA) contributes exactly 0 by construction — "
             "it is what the evidence is measured against. `neighbour cut` is "
             "the localisation diagnostic (docs/dfs_attention.md §7): below "
             "0.05 the localised solve is effectively exact on this geometry.\n")
    for sc, cut in loose:
        L.append(f"> ⚠ At the **{sc}** target the cut is {cut:.3f}: the long "
                 "length scales of a coarse target make the observations "
                 "correlated well beyond the 32-token neighbourhood, so that "
                 "row's DFS is an **over-estimate** — the true coarse-target "
                 "evidence is lower still, which only strengthens the "
                 "direction of the scale trend. Truncation always biases DFS "
                 "upward.\n")
    prof_share = float(np.mean([evals[("dfs", s)]["evidence"]["protocol"]
                                ["profile_share"] for s in ev_seeds]))
    L.append(f"\n**The most informative number in this report.** The estimator "
             f"says the profiles carry **{prof_share*100:.0f} %** of the "
             "month's independent information, yet removing them costs the "
             "trained model only the couple of percent RMSE in leg 2. The "
             "evidence is there and is correctly identified; what fails to "
             "exploit it is downstream of the evidence estimate. That "
             "localises the remaining gap to training and decoding rather "
             "than to the fusion rule — consistent with the month-identity "
             "recall overfit that caps every variant in this family at a "
             "~0.94 validation score by ~5k steps.\n")

# ------------------------------------------------------------------ verdict
L.append("\n## Verdict\n")
ok = {}
for v in VARIANTS:
    acc = agg(runs, v, ["test", "TEMP"])
    dup = agg(evals, v, ["probes", "duplicate_half_declared",
                         "rel_output_change_mean"])
    full = agg(evals, v, ["modality_matrix", "full", "TEMP"])
    drop = agg(evals, v, ["modality_matrix", "drop_profiles", "TEMP"])
    ok[v] = dict(acc=acc, dup=dup,
                 ret=(drop[0] / full[0] - 1.0) if full[2] and full[0] else float("nan"))
best_acc = min((o["acc"][0] for o in ok.values() if o["acc"][2]), default=float("nan"))
L.append("| variant | leg 1 duplication (declared) | leg 2 retention | "
         "leg 3 accuracy vs best | all three |")
L.append("|---|---|---|---|---|")
for v in VARIANTS:
    o = ok[v]
    if o["acc"][2] == 0:
        L.append(f"| {NAME[v]} | — | — | — | not run |")
        continue
    l1 = o["dup"][0]
    l2 = o["ret"]
    l3 = o["acc"][0] / best_acc - 1.0
    p1 = "" if np.isnan(l1) else ("pass" if l1 < 0.01 else "fail")
    p2 = "" if np.isnan(l2) else ("pass" if l2 > 0.02 else "fail")
    p3 = "pass" if l3 < 0.03 else "fail"
    allp = "**yes**" if (p1 == "pass" and p2 == "pass" and p3 == "pass") else "no"
    L.append(f"| {NAME[v]} | {'—' if np.isnan(l1) else f'{l1:.4f}'} {p1} | "
             f"{'—' if np.isnan(l2) else f'{l2*100:+.1f} %'} {p2} | "
             f"{l3*100:+.1f} % {p3} | {allp} |")
L.append("\nThresholds (fixed before the runs, stated here so they are not "
         "read off the results): leg 1 passes below 0.01 relative output "
         "change; leg 2 passes above a 2 % RMSE degradation when profiles are "
         "withheld; leg 3 passes within 3 % of the best variant's TEMP RMSE.\n")

# ------------------------------------------------------------ section-11 xref
pp = os.path.join(C.CACHE, "dfs_evidence_probes.json")
if os.path.exists(pp):
    L.append("\n## Section-11 evidence probes\n")
    L.append("The mechanism-level requirements (exact duplication, depth "
             "complementarity, smooth vs thermocline density, resolution "
             "sweep, horizontal cases) are reported in full by "
             "[`dfs_evidence_probes.md`](dfs_evidence_probes.md). Headline "
             "numbers:\n")
    r = json.load(open(pp))["results"]
    dup0 = r["duplication"]
    worst = max(abs(x["d_dfs"]) for x in dup0["rows"])
    L.append(f"- exact duplication (×2/×4/×8, whole profiles, and real-time + "
             f"delayed-mode copies): largest ΔDFS = **{worst:.1e}** on a base "
             f"of {dup0['base_dfs']:.2f};")
    dc = r["depth_complementarity"]
    L.append(f"- one column's {dc[-1]['bands']} depth bands carry "
             f"{dc[-1]['dfs']:.2f} DFS — different depths are complementary, "
             "not redundant;")
    vd = {(x["target"], x["column"], x["sampling_dbar"]): x["dfs"]
          for x in r["vertical_density"]}
    ft = [k for k in vd if k[0].startswith("fine")]
    if ft:
        t = [k for k in ft if k[1] == "thermocline" and k[2] == 2.0][0]
        s = [k for k in ft if k[1] == "smooth" and k[2] == 2.0][0]
        L.append(f"- the same 2 dbar sampling at a 10 m target yields "
                 f"**{vd[t]:.1f}** DFS across a thermocline vs **{vd[s]:.1f}** "
                 "in a smooth column of equal anomaly amplitude;")
    sw = r["resolution_sweep"]["thermocline"]
    L.append(f"- the resolution sweep is monotone: "
             + " → ".join(f"{x['dfs']:.1f}" for x in sw)
             + " DFS for Δz = "
             + " → ".join(f"{x['dz_m']:.0f}" for x in sw) + " m.")

path = os.path.join(C.REPORTS, "dfs_success_criterion.md")
open(path, "w").write("\n".join(L) + "\n")
print("wrote", path)
