#!/usr/bin/env bash
# EXP-075: paired root-action audit of EXP-074's v26 response oracle.
set -euo pipefail
cd /mnt/data/curling2/csas_world

OUT=eval_out/exp075_oracle_audit
LOG="$OUT/run.log"
LOCK="$OUT/launcher.pid"
mkdir -p "$OUT"
if [[ -f "$LOCK" ]]; then
  old_pid=$(<"$LOCK")
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "REFUSING: EXP-075 launcher $old_pid is already alive" | tee -a "$LOG"
    exit 1
  fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

export PYTHONUNBUFFERED=1
ENVV="WORLD_BOUNDARY_REMOVAL=1 PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu \
OMP_NUM_THREADS=2 POLICY_BATCH_CAP=32 VALUE_EVAL_BATCH=48 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing \
GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none"

say() { echo "[exp075] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }
run() { env -u LD_LIBRARY_PATH $ENVV "$@"; }
run_gpu() {
  local gpu=$1
  shift
  env -u LD_LIBRARY_PATH $ENVV CUDA_VISIBLE_DEVICES="$gpu" "$@"
}

if [[ ! -f "$OUT/states.npz" ]]; then
  say "fresh v25-v26 state generation start"
  run_gpu 1 python3 scripts/exp075_oracle_audit.py --phase states >> "$LOG" 2>&1
fi
say "state generation complete"

say "16k action audit start (3 shards)"
pids=()
for shard in 0 1 2; do
  run_gpu $((shard + 1)) timeout 43200 \
    python3 scripts/exp075_oracle_audit.py --phase actions \
      --num-shards 3 --shard-id "$shard" \
      > "$OUT/actions_shard${shard}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    say "FATAL action worker $pid failed"
    exit 1
  fi
done
say "action audit complete"

say "exact 48x8 paired continuation evaluation start (3 shards)"
pids=()
for shard in 0 1 2; do
  run_gpu $((shard + 1)) timeout 86400 \
    python3 scripts/exp075_oracle_audit.py --phase eval \
      --num-shards 3 --shard-id "$shard" \
      > "$OUT/eval_shard${shard}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    say "FATAL evaluation worker $pid failed"
    exit 1
  fi
done
say "paired evaluation complete"

run python3 scripts/exp075_oracle_audit.py --phase aggregate \
  | tee -a "$LOG" | tee -a experiments_log.md
say "EXP075_DONE"
