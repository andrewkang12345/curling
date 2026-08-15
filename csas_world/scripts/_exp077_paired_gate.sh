#!/usr/bin/env bash
# EXP-077: fresh independent confirmation of the paired exact response-oracle gate.
set -euo pipefail
cd /mnt/data/curling2/csas_world

OUT=eval_out/exp077_paired_gate
LOG="$OUT/run.log"
LOCK="$OUT/launcher.pid"
mkdir -p "$OUT"
if [[ -f "$LOCK" ]]; then
  old_pid=$(<"$LOCK")
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "REFUSING: EXP-077 launcher $old_pid is already alive" | tee -a "$LOG"
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
GPUS=(0 1 2 3)
NSHARDS=${#GPUS[@]}
STATE_SEED=770077

say() { echo "[exp077] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }
run() { env -u LD_LIBRARY_PATH $ENVV "$@"; }
run_gpu() {
  local gpu=$1
  shift
  env -u LD_LIBRARY_PATH $ENVV CUDA_VISIBLE_DEVICES="$gpu" "$@"
}

if [[ ! -f "$OUT/states.npz" ]]; then
  say "fresh v25-v26 balanced state generation start"
  run_gpu 1 python3 scripts/exp075_oracle_audit.py --phase states \
    --out-dir "$OUT" --seed "$STATE_SEED" >> "$LOG" 2>&1
fi
say "state generation complete"

say "16k frozen-action generation start ($NSHARDS shards on GPUs ${GPUS[*]})"
pids=()
for shard in "${!GPUS[@]}"; do
  run_gpu "${GPUS[$shard]}" timeout 43200 \
    python3 scripts/exp075_oracle_audit.py --phase actions \
      --out-dir "$OUT" --seed "$STATE_SEED" \
      --num-shards "$NSHARDS" --shard-id "$shard" \
      > "$OUT/actions_shard${shard}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then say "FATAL action worker $pid failed"; exit 1; fi
done
say "frozen-action generation complete"

say "8-repeat paired exact baseline screen start"
pids=()
for shard in "${!GPUS[@]}"; do
  run_gpu "${GPUS[$shard]}" timeout 43200 \
    python3 scripts/exp077_paired_gate.py --phase gate --out-dir "$OUT" \
      --num-shards "$NSHARDS" --shard-id "$shard" \
      > "$OUT/gate_shard${shard}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then say "FATAL gate worker $pid failed"; exit 1; fi
done
say "paired exact baseline screen complete"

say "16-repeat independent exact readout start"
pids=()
for shard in "${!GPUS[@]}"; do
  run_gpu "${GPUS[$shard]}" timeout 86400 \
    python3 scripts/exp077_paired_gate.py --phase eval --out-dir "$OUT" \
      --num-shards "$NSHARDS" --shard-id "$shard" \
      > "$OUT/eval_shard${shard}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then say "FATAL readout worker $pid failed"; exit 1; fi
done
say "independent exact readout complete"

run python3 scripts/exp077_paired_gate.py --phase aggregate --out-dir "$OUT" \
  | tee -a "$LOG" | tee -a experiments_log.md
say "EXP077_DONE"
