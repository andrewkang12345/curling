#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 PID [PID ...]" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")" && pwd)"
TRUE_SPAN_ROOT="$ROOT/valueModel/ablation"
INVERSE_GLOB="${INVERSE_GLOB:-$ROOT/inverse_rescued/stones_with_estimates.chunk*.csv}"
VAL_END_FRAC="${VAL_END_FRAC:-0.10}"
SPLIT_SEED="${SPLIT_SEED:-123}"

COMP_IDS=(0 22230015 23240026 24250026)
GPU_IDS=(0 1 2 3)
GNN_PIDS=("$@")

if [[ "${#GNN_PIDS[@]}" -ne 4 ]]; then
  echo "expected 4 GNN PIDs in gpu order 0..3; got ${#GNN_PIDS[@]}" >&2
  exit 2
fi

backup_one() {
  local comp_id="$1"
  local run_dir="$ROOT/holdouts/$comp_id"
  local backup_dir="$run_dir/backups/pre_fixed_split_$(date -u +%Y%m%d_%H%M%S)"
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

wait_and_regen() {
  local gnn_pid="$1"
  local comp_id="$2"
  local gpu_id="$3"

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] gpu=$gpu_id comp=$comp_id waiting for GNN pid=$gnn_pid"
  while kill -0 "$gnn_pid" 2>/dev/null; do
    sleep 60
  done

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] gpu=$gpu_id freed; starting holdout regeneration for comp=$comp_id"
  (
    export INVERSE_GLOB
    export VAL_END_FRAC
    export SPLIT_SEED
    "$ROOT/run_holdout_comp.sh" "$comp_id" "$gpu_id"
  )
}

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] backing up existing holdout artifacts before regeneration"
for comp_id in "${COMP_IDS[@]}"; do
  backup_one "$comp_id"
done

watcher_pids=()
for idx in "${!COMP_IDS[@]}"; do
  wait_and_regen "${GNN_PIDS[$idx]}" "${COMP_IDS[$idx]}" "${GPU_IDS[$idx]}" &
  watcher_pids+=("$!")
done

status=0
for pid in "${watcher_pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] one or more holdout regenerations failed"
  exit "$status"
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] launching deferred true-span GNN sweep"
"$TRUE_SPAN_ROOT/launch_true_span_runs.sh"
