"""Queue of the CLOINet-style read-out runs (2026-10-21): method A with the cluster covariance.

KA-Ocean (A: U-Net state, joint background) with ``--kriging ka_cluster``: the same local solve,
noise and amplitude heads, but the covariance of CLOINet (Cutolo et al. 2024), the similarity of
20 soft cluster memberships read from the state, with no distance term. Everything else is the
queue of ``114`` (A's setups: CESM2 70,000 steps chosen on 2004-2005; real 2016-2020, 30,000 steps
chosen on 2021), seeds 1234-1236. No run reads a test year.

  .venv/bin/python experiments/real_data/131_cluster_campaign.py --smoke --arms cl_real --seeds 1234
  .venv/bin/python experiments/real_data/131_cluster_campaign.py --gpus 0,1,5,6 --per-gpu 2
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("campaign114", ROOT / "experiments/real_data/114_method_selection_campaign.py")
Q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Q)

Q.OUTPUT = ROOT / "outputs/cluster_20261021"
#: arm -> (dataset, background, backbone, Poseidon-T initialisation (unused by the U-Net), kriging)
Q.ARMS = {"cl_cesm2": ("cesm2", "joint", "unet", "pretrained", "ka_cluster"),
          "cl_real": ("real", "joint", "unet", "pretrained", "ka_cluster")}
Q.DEFAULT_ARMS = tuple(Q.ARMS)

if __name__ == "__main__":
    Q.main()
