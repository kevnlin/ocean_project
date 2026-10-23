"""Every checkpoint the paper reports, for the one-shot registrations of the sealed tests.

The rows are those of ``118_validation_tables.py`` (one source for the tables and the tests): the
chosen method A, the other candidates, the baselines and every ablation, all seeds. A run of the
reference folder (the OI-anchored local transformer, runner 94) is family "reference", every other
run "grid". Each checkpoint's spread calibration (``116``) is registered with it by 109 and 113.

  .venv/bin/python experiments/synthetic/119_sealed_registration.py              # list; registers nothing
  .venv/bin/python experiments/synthetic/119_sealed_registration.py --register   # once, after Zihao agrees
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("tables118", ROOT / "experiments/synthetic/118_validation_tables.py")
tables = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tables)

SCORERS = {"cesm2": ROOT / "experiments/synthetic/109_sealed_cesm2_test.py",
           "real": ROOT / "experiments/real_data/113_sealed_real_test.py"}


def checkpoints(groups):
    """(family, checkpoint, calibrated) of every row and seed, and the rows with a missing run."""
    out, missing = [], []
    for rows in groups.values():
        for name, folder, tag in rows:
            for seed in tables.SEEDS:
                run = folder / f"{tag}_s{seed}"
                if not (run / f"summary_seed{seed}.json").exists():
                    missing.append(f"{name} seed {seed}")
                    continue
                family = "reference" if folder == tables.REFERENCE else "grid"
                calibrated = (tables.CALIBRATIONS / f"{run.resolve().parent.name}__{run.resolve().name}.json").exists()
                out.append((family, run / f"best_seed{seed}.pt", calibrated))
    return out, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--register", action="store_true", help="write both registrations (once)")
    args = parser.parse_args()
    plans = {"cesm2": checkpoints(tables.CESM2), "real": checkpoints(tables.REAL_ROWS)}
    for test, (entries, missing) in plans.items():
        print(f"{test}: {len(entries)} checkpoints, {sum(c for _, _, c in entries)} with a calibration"
              + (f"; missing: {', '.join(missing)}" if missing else ""))
        for family, path, calibrated in entries:
            print(f"  {family:9s} {'cal' if calibrated else '   '} {path}")
    if not args.register:
        return
    if any(missing for _, missing in plans.values()):
        raise SystemExit("a reported run has not finished; register when every row is complete")
    for test, (entries, _) in plans.items():
        items = [f"{family}:{path}" if test == "cesm2" else str(path) for family, path, _ in entries]
        subprocess.run([sys.executable, str(SCORERS[test]), "--register", *items], check=True,
                       env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})


if __name__ == "__main__":
    main()
