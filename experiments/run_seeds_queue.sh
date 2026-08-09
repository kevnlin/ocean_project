#!/usr/bin/env bash
# Remaining seeds for the two GPU experiments, so every headline number reaches
# the repo's 3-seed convention.
#
# This box is permanently contended (~73 GB of 80 GB held by other people's
# jobs on every card), so both queues use the low-memory path: training stack in
# host RAM, capped inference batch, hard per-process ceiling.  Measured peak
# 3.9 GB.  Pick cards that are idle *in compute* — check `nvidia-smi` first;
# GPUs 4/5 have been at 0% utilization with memory reserved by idle VLLM
# servers, which is the ideal case (their memory is spoken for, their SMs are not).
#
#   bash experiments/run_seeds_queue.sh ssh     4     # arg: which queue, which GPU
#   bash experiments/run_seeds_queue.sh density 5
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/nvidia/.venv/bin/python}
QUEUE=${1:?usage: run_seeds_queue.sh ssh|density GPU_INDEX [seeds]}
GPU=${2:?usage: run_seeds_queue.sh ssh|density GPU_INDEX [seeds]}
SEEDS=${3:-"1235 1236"}
mkdir -p outputs/logs
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for s in $SEEDS; do
  echo "[$(date +%T)] $QUEUE seed $s on GPU $GPU"
  if [ "$QUEUE" = "ssh" ]; then
    $PY -u experiments/29_ssh_ablation.py --seed "$s" \
        --ssh-cache outputs/cache/ssh_dyn.npz \
        --cpu-tensors --fwd-batch 16 --mem-cap-gb 5.5 \
        --arms control_pws,treat_pws_ssh,profiles_only \
        > "outputs/logs/ssh_ablation_s${s}.log" 2>&1
  else
    $PY -u experiments/08_density_ablation.py --seed "$s" \
        --densities 4000,6000 --methods clim_floor,unet_depthwise \
        --out-suffix _ext --cpu-tensors \
        > "outputs/logs/density_ext_${s}.log" 2>&1
  fi
  echo "[$(date +%T)] $QUEUE seed $s exit=$?"
done
echo "[$(date +%T)] $QUEUE QUEUE COMPLETE"
