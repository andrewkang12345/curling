#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
INVERSE_GLOB="${INVERSE_GLOB:-$ROOT/inverse_rescued/stones_with_estimates.chunk*.csv}"
VAL_END_FRAC="${VAL_END_FRAC:-0.10}"
SPLIT_SEED="${SPLIT_SEED:-123}"

COMP_IDS=(0 22230015 23240026 24250026)
GPU_IDS=(0 1 2 3)

backup_one() {
  local comp_id="$1"
  local run_dir="$ROOT/holdouts/$comp_id"
  local backup_dir="$run_dir/backups/pre_fixed_split_${RUN_TS}"
  mkdir -p "$backup_dir"
  for name in model scoring reports logs; do
    if [[ -e "$run_dir/$name" ]]; then
      cp -a "$run_dir/$name" "$backup_dir/"
    fi
  done
  mkdir -p "$run_dir/logs"
  : > "$run_dir/logs/pipeline.log"
  : > "$run_dir/logs/train.log"
}

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] backing up existing holdout artifacts"
for comp_id in "${COMP_IDS[@]}"; do
  backup_one "$comp_id"
done

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting fixed-split holdout regeneration"
pids=()
for idx in "${!COMP_IDS[@]}"; do
  comp_id="${COMP_IDS[$idx]}"
  gpu_id="${GPU_IDS[$idx]}"
  echo "  launching comp=$comp_id on gpu=$gpu_id"
  (
    export INVERSE_GLOB
    export VAL_END_FRAC
    export SPLIT_SEED
    "$ROOT/run_holdout_comp.sh" "$comp_id" "$gpu_id"
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] holdout regeneration failed"
  exit "$status"
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] holdout regeneration finished"
