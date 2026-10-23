"""Queue of the DKNN-style read-out runs (2026-10-21): method A with an attention-score covariance.

KA-Ocean (A: U-Net state, joint background) with ``--kriging ka_scores``: the same local solve and
noise, but the covariance of DKNN (Chen et al. 2024), scaled dot products of learned query and key
projections of every point's state vector with the neighbours' own scores removed. Everything else
is the queue of ``114`` (A's setups), seeds 1234-1236. No run reads a test year.

  .venv/bin/python experiments/real_data/135_scores_campaign.py --smoke --arms sc_real sc_cesm2 --seeds 1234
  .venv/bin/python experiments/real_data/135_scores_campaign.py --gpus 6 --per-gpu 3
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("campaign114", ROOT / "experiments/real_data/114_method_selection_campaign.py")
Q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Q)

Q.OUTPUT = ROOT / "outputs/scores_20261021"
#: arm -> (dataset, background, backbone, Poseidon-T initialisation (unused by the U-Net), kriging)
Q.ARMS = {"sc_cesm2": ("cesm2", "joint", "unet", "pretrained", "ka_scores"),
          "sc_real": ("real", "joint", "unet", "pretrained", "ka_scores")}
Q.DEFAULT_ARMS = tuple(Q.ARMS)

if __name__ == "__main__":
    Q.main()
