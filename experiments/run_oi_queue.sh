#!/usr/bin/env bash
# Phase-1 (M1) queue: tune OI on validation months, then score the headline
# comparison on the 12 pinned test months, then re-sweep on training months as
# a stability check.  CPU only — no GPU needed for any step.
#
#   bash experiments/run_oi_queue.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/nvidia/.venv/bin/python}
mkdir -p outputs/logs

echo "[$(date +%T)] stage 1/3: OI tuning on validation months"
$PY -u experiments/26_oi_tuning.py --split val > outputs/logs/oi_tuning_val.log 2>&1
echo "[$(date +%T)] stage 1 exit=$?"

echo "[$(date +%T)] stage 2/3: OI vs U-Net on the 12 pinned test months"
$PY -u experiments/27_oi_vs_unet.py --verify-unet > outputs/logs/oi_vs_unet.log 2>&1
echo "[$(date +%T)] stage 2 exit=$?"

echo "[$(date +%T)] stage 3/3: tuning stability check on training months"
$PY -u experiments/26_oi_tuning.py --split train > outputs/logs/oi_tuning_train.log 2>&1
echo "[$(date +%T)] stage 3 exit=$?"

echo "[$(date +%T)] QUEUE COMPLETE"
