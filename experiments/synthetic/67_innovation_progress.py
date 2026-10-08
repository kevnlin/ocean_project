"""Refresh the report's queue status without recomputing scientific results."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
STATUS = ROOT / "outputs/innovation_20261007/status.json"
REPORT = ROOT / "experiments/synthetic/62_innovation_report.py"

while True:
    subprocess.run([sys.executable, str(REPORT), "--refresh-status-only"], cwd=ROOT, check=True)
    state = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    if state.get("phase") == "complete" or state.get("failure"):
        break
    time.sleep(60)
