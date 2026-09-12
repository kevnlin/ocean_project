#!/usr/bin/env bash
# Track A real-Argo run queue.
#
# One seed per GPU so the three headline seeds run concurrently; within a shard
# the packages run in sequence because P2 consumes the checkpoints P0 writes.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
ROWS=${ROWS:-"dfs_expertlocal_cbottle,uniform_expertlocal_cbottle,count_expertlocal_cbottle,thin_expertlocal_cbottle,superob_expertlocal_cbottle"}
LOG=${LOG_DIR:-/tmp/argo_runs}
STEPS=${STEPS:-4000}
mkdir -p "$LOG"

# Resume-aware: a (package, region, seed, protocol) whose signed artifact is
# already on disk is skipped.  The queue is long enough that it will be
# interrupted, and re-running a finished package would overwrite a good artifact
# with an identical one at best, and waste an hour at worst.
run () {  # gpu seed package region [split-protocol]
  local gpu=$1 seed=$2 pkg=$3 region=$4 proto=${5:-main}
  local out="outputs/argo_${pkg}_${region}"
  local tag=""
  if [ "$proto" != "main" ]; then out="outputs/argo_${pkg}_${region}_${proto}"; tag="_${proto}"; fi
  if [ -f "$out/artifact_${pkg}_seed${seed}.json" ]; then
    echo "[$(date +%H:%M:%S)] skip  $pkg $region$tag seed=$seed (artifact exists)"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] start $pkg $region$tag seed=$seed gpu=$gpu"
  CUDA_VISIBLE_DEVICES=$gpu $PY experiments/real_data/45_argo_real_data.py \
      --package "$pkg" --region "$region" --rows "$ROWS" --seed "$seed" \
      --steps "$STEPS" --split-protocol "$proto" --output "$out" \
      > "$LOG/${pkg}_${region}${tag}_s${seed}.log" 2>&1
  echo "[$(date +%H:%M:%S)] done  $pkg $region$tag seed=$seed rc=$?"
}

redundancy () {  # gpu seed region — reuses the P0 checkpoints
  local gpu=$1 seed=$2 region=$3
  local dir="outputs/argo_P0_${region}"
  local ck=""
  for r in dfs_expertlocal_cbottle uniform_expertlocal_cbottle count_expertlocal_cbottle; do
    [ -f "$dir/${r}_s${seed}.pt" ] && ck="${ck}${ck:+,}${r}=$dir/${r}_s${seed}.pt"
  done
  if [ -f "outputs/argo_P2_${region}/artifact_P2_seed${seed}.json" ]; then
    echo "[$(date +%H:%M:%S)] skip  P2 $region seed=$seed (artifact exists)"; return 0
  fi
  echo "[$(date +%H:%M:%S)] start P2 $region seed=$seed gpu=$gpu ckpts=${ck:-none}"
  CUDA_VISIBLE_DEVICES=$gpu $PY experiments/real_data/46_argo_redundancy.py \
      --region "$region" --seed "$seed" --checkpoints "$ck" \
      > "$LOG/P2_${region}_s${seed}.log" 2>&1
  echo "[$(date +%H:%M:%S)] done  P2 $region seed=$seed rc=$?"
}

shard () {  # gpu seed
  local gpu=$1 seed=$2
  for region in gulfstream npac_gyre; do
    run "$gpu" "$seed" P0 "$region"     # also writes the checkpoints P2 needs
    redundancy "$gpu" "$seed" "$region"
    run "$gpu" "$seed" P1 "$region"
    run "$gpu" "$seed" P5 "$region"
    # SECONDARY protocol: eras shifted inside ECCO V4r4's coverage so ECCO can
    # be scored at all.  Never merged into the main headline table.
    run "$gpu" "$seed" P0 "$region" ecco_overlap
  done
}

shard 0 1234 &
shard 1 1235 &
shard 2 1236 &
wait
echo "queue complete"
