"""The final method, chosen once by a rule fixed on 2026-10-20 before its deciding runs finished.

Candidates (``114_method_selection_campaign.py``), simplest first:

  A  U-Net + kriging attention (flow kernel), joint background
  B  U-Net + kriging attention, prior background
  C  Poseidon-T (public weights) + kriging attention, joint background

Rule (``docs/superpowers/specs/2026-10-20-method-selection.md``): only the selection years are read
(CESM2 2004-2005, real 2021). A run's score is the one it chose its checkpoint by: the mean over leads
0, 1, 3, 6 of the per-variable standardized RMSE on the fixed 8000-cell validation draw
(``summary["validation"]["scores"]["macro_z"]``). A candidate's score on a dataset is the mean over
seeds 1234-1236; its overall score is the mean over the two datasets of that score divided by A's.
The chosen method is the simplest candidate within 0.5 % (0.005) of the lowest overall score.

  .venv/bin/python experiments/real_data/115_method_selection.py            # decide; refuses if a run is missing
  .venv/bin/python experiments/real_data/115_method_selection.py --status   # list the finished runs, decide nothing
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/selection_20261020"
DECISION = OUTPUT / "decision.json"
SEEDS = (1234, 1235, 1236)
CANDIDATES = ("A", "B", "C")
TIE = 0.005
DESCRIPTION = {"A": "U-Net + kriging attention (flow kernel), joint background",
               "B": "U-Net + kriging attention (flow kernel), prior background",
               "C": "Poseidon-T (public weights) + kriging attention (flow kernel), joint background"}
#: candidate -> dataset -> (run folder pattern, steps, background, backbone)
RUNS = {
    "A": {"cesm2": ("/DATA2/linke/ocean_project_ka/outputs/grid_ka_70k_20261019/gj_flow_s{seed}", 70000, "joint", "unet"),
          "real": ("/DATA2/linke/ocean_project_ka_dev/outputs/real_global_20261019/scratch_s{seed}", 30000, "joint", "unet")},
    "B": {"cesm2": ("/DATA2/linke/ocean_project_ka_dev/outputs/formal_20261019/c_gp_flow_s{seed}", 70000, "prior", "unet"),
          "real": (str(OUTPUT / "rb_prior_s{seed}"), 30000, "prior", "unet")},
    "C": {"cesm2": ("/DATA2/linke/ocean_project_ka_dev2/outputs/poseidon_20261019/p_ka_s{seed}", 70000, "joint", "poseidon"),
          "real": (str(OUTPUT / "rc_poseidon_s{seed}"), 30000, "joint", "poseidon")}}
VALIDATION_YEARS = {"cesm2": [2004, 2005], "real": [2021]}


def score(folder, seed, steps, background, backbone, dataset):
    """A finished run's selection score, or None if its summary is not written yet."""
    path = Path(folder) / f"summary_seed{seed}.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text())
    config = summary["config"]
    if (summary["completed_steps"] != steps or summary["seed"] != seed or config["kriging"] != "ka_flow"
            or config["background"] != background or config.get("backbone", "unet") != backbone
            or config.get("poseidon_init", "pretrained") != "pretrained"
            or summary["val_years"] != VALIDATION_YEARS[dataset]):
        raise ValueError(f"{path} is not the candidate's run")
    return float(summary["validation"]["scores"]["macro_z"])


def collect(runs=RUNS):
    """candidate -> dataset -> per-seed scores (None where a run has not finished)."""
    return {c: {d: [score(pattern.format(seed=s), s, steps, background, backbone, d) for s in SEEDS]
                for d, (pattern, steps, background, backbone) in datasets.items()}
            for c, datasets in runs.items()}


def decide(scores):
    """The rule: mean over seeds, divided by A's, mean over datasets; the simplest within TIE of the best."""
    if any(v is None for datasets in scores.values() for values in datasets.values() for v in values):
        raise ValueError("a candidate run has not finished; the rule needs every run")
    mean = {c: {d: sum(v) / len(v) for d, v in datasets.items()} for c, datasets in scores.items()}
    ratio = {c: {d: mean[c][d] / mean["A"][d] for d in mean[c]} for c in CANDIDATES}
    overall = {c: sum(ratio[c].values()) / len(ratio[c]) for c in CANDIDATES}
    best = min(overall.values())
    chosen = next(c for c in CANDIDATES if overall[c] - best < TIE)
    return {"mean": mean, "ratio_to_A": ratio, "overall": overall, "best": min(overall, key=overall.get),
            "chosen": chosen}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--status", action="store_true", help="list the finished runs; decide nothing")
    args = parser.parse_args()
    scores = collect()
    if args.status:
        for c, datasets in scores.items():
            for d, values in datasets.items():
                print(f"{c} {d:5s} " + "  ".join("running" if v is None else f"{v:.4f}" for v in values))
        return
    if DECISION.exists():
        raise SystemExit(f"{DECISION} exists: the method is chosen once")
    outcome = decide(scores)
    record = {"decided_unix": time.time(), "rule": __doc__, "tie": TIE, "candidates": DESCRIPTION,
              "runs": RUNS, "scores": scores, **outcome}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    temporary = DECISION.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=1) + "\n")
    os.replace(temporary, DECISION)
    for c in CANDIDATES:
        print(f"{c}: CESM2 {outcome['mean'][c]['cesm2']:.4f} ({outcome['ratio_to_A'][c]['cesm2']:.4f} of A), "
              f"real {outcome['mean'][c]['real']:.4f} ({outcome['ratio_to_A'][c]['real']:.4f} of A), "
              f"overall {outcome['overall'][c]:.4f}  {DESCRIPTION[c]}")
    print(f"lowest overall: {outcome['best']}; chosen (simplest within {TIE:.3f}): {outcome['chosen']} -> {DECISION}")


if __name__ == "__main__":
    main()
