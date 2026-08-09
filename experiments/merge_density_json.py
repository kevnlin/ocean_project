"""Phase-2 / Task 2.1 — merge an extended density sweep into the week-2 curve.

`08_density_ablation.py --densities 4000,6000 --out-suffix _ext` writes its
results to `density_ablation_seed<seed>_ext.{json,npz}`, leaving the committed
week-2 files untouched.  This script folds the extension into the main curve.

It is deliberately defensive: the week-2 files are committed results, and a bad
merge silently corrupts every downstream figure.  So it

  * refuses to run unless both parts exist;
  * writes a `.bak_premerge` copy of the main file the first time it touches it;
  * refuses to merge two runs with different `run_config` essentials (config,
    split seed, train/test months) -- densities from incomparable runs must
    never land in one table;
  * skips densities already present rather than duplicating them, and reports
    what it skipped;
  * is idempotent: running it twice changes nothing the second time.

Run:
    python experiments/merge_density_json.py                      # all 3 seeds
    python experiments/merge_density_json.py --seeds 1234 --dry-run
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np

from ocean_tokenizer import config as C

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="1234,1235,1236")
ap.add_argument("--suffix", default="_ext", help="suffix of the extension run")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

# run_config keys that must agree before two sweeps may share a table
MUST_MATCH = ("config", "split_seed", "sweep_seed", "train_months", "test_months")


def density_of(row):
    for key in ("density", "n_profiles", "profiles"):
        if key in row:
            return row[key]
    raise KeyError(f"no density field in result row: {sorted(row)[:8]}")


for seed in [int(s) for s in args.seeds.split(",")]:
    main_j = os.path.join(C.CACHE, f"density_ablation_seed{seed}.json")
    ext_j = os.path.join(C.CACHE, f"density_ablation_seed{seed}{args.suffix}.json")
    main_n = os.path.join(C.CACHE, f"density_ablation_seed{seed}_depth.npz")
    ext_n = os.path.join(C.CACHE,
                         f"density_ablation_seed{seed}{args.suffix}_depth.npz")

    if not os.path.exists(ext_j):
        print(f"seed {seed}: no extension file ({os.path.basename(ext_j)}) — skip")
        continue
    if not os.path.exists(main_j):
        print(f"seed {seed}: MAIN FILE MISSING ({os.path.basename(main_j)}) — skip")
        continue

    main, ext = json.load(open(main_j)), json.load(open(ext_j))
    mc, ec = main.get("run_config", {}), ext.get("run_config", {})
    bad = [k for k in MUST_MATCH if k in mc and k in ec and mc[k] != ec[k]]
    if bad:
        print(f"seed {seed}: REFUSING to merge — run_config differs on {bad}")
        continue

    have = {density_of(r) for r in main["results"]}
    new = [r for r in ext["results"] if density_of(r) not in have]
    dup = sorted({density_of(r) for r in ext["results"]} & have)
    if dup:
        print(f"seed {seed}: densities already present, not re-added: {dup}")
    if not new:
        print(f"seed {seed}: nothing new to merge (idempotent no-op)")
        continue

    merged_densities = sorted(have | {density_of(r) for r in new})
    print(f"seed {seed}: adding {len(new)} rows at densities "
          f"{sorted({density_of(r) for r in new})} -> curve {merged_densities}")
    if args.dry_run:
        continue

    if not os.path.exists(main_j + ".bak_premerge"):
        shutil.copy2(main_j, main_j + ".bak_premerge")
    run_cfg = dict(mc)
    run_cfg["densities"] = merged_densities
    run_cfg["merged_from"] = run_cfg.get("merged_from", []) + [
        {"file": os.path.basename(ext_j), "git_commit": ec.get("git_commit")}]
    json.dump({"results": main["results"] + new, "run_config": run_cfg},
              open(main_j, "w"), indent=1)

    if os.path.exists(ext_n) and os.path.exists(main_n):
        if not os.path.exists(main_n + ".bak_premerge"):
            shutil.copy2(main_n, main_n + ".bak_premerge")
        d = dict(np.load(main_n))
        for k, v in np.load(ext_n).items():
            if k == "depths":
                continue
            d.setdefault(k, v)
        np.savez(main_n, **d)
        print(f"         depth tables merged -> {os.path.basename(main_n)}")

print("\ndone.  Backups: *.bak_premerge (only written on the first merge).")
