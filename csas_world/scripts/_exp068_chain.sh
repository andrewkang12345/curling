#!/usr/bin/env bash
# EXP-068: vectorised 4-ply search — h=2 back-check, then h=4 and h=10.
set -uo pipefail
cd /mnt/data/curling2/csas_world
OUT=eval_out/exp068_deep
LOG="$OUT/chain.log"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1 WORLD_BOUNDARY_REMOVAL=1
CPU_ENV="PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu VALUE_EVAL_BATCH=128 \
POLICY_BATCH_CAP=256 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing GNN_NODE_FEATURE_MODE=none \
GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none"
say() { echo "[exp068] $* $(date -u +%H:%M)" | tee -a "$LOG"; }

say "start"
# ---- 0. vectorisation back-check at h=2 (scored on EXP-066's expectimax table) ----
pids=()
for k in $(seq 0 11); do
  env -u LD_LIBRARY_PATH $CPU_ENV CUDA_VISIBLE_DEVICES=$((k % 4)) \
    python3 scripts/exp068_deep_search.py --phase validate_h2 --n-states 30 \
    --shard-id $k --num-shards 12 --out-dir "$OUT" >> "$OUT/valh2_shard$k.log" 2>&1 &
  pids+=($!); sleep 3
done
wait "${pids[@]}" || true
say "validate_h2 done"

for H in 4 10; do
  D="$OUT/h$(printf %02d $H)"
  mkdir -p "$D"
  if [ ! -f "$D/states.npz" ]; then
    env -u LD_LIBRARY_PATH $CPU_ENV CUDA_VISIBLE_DEVICES=0 \
      python3 scripts/exp068_deep_search.py --phase states --horizon $H --n-states 30 \
      --out-dir "$OUT" >> "$D/states.log" 2>&1
  fi
  say "h$H states done"
  pids=()
  for k in $(seq 0 11); do
    env -u LD_LIBRARY_PATH $CPU_ENV CUDA_VISIBLE_DEVICES=$((k % 4)) \
      python3 scripts/exp068_deep_search.py --phase search --horizon $H \
      --shard-id $k --num-shards 12 --out-dir "$OUT" >> "$D/search_shard$k.log" 2>&1 &
    pids+=($!); sleep 3
  done
  wait "${pids[@]}" || true
  say "h$H search done"
  pids=()
  for k in $(seq 0 7); do
    env -u LD_LIBRARY_PATH $CPU_ENV CUDA_VISIBLE_DEVICES=$((k % 4)) \
      python3 scripts/exp068_deep_search.py --phase adjudicate --horizon $H \
      --shard-id $k --num-shards 8 --out-dir "$OUT" >> "$D/adj_shard$k.log" 2>&1 &
    pids+=($!); sleep 3
  done
  wait "${pids[@]}" || true
  say "h$H adjudicate done"
  env -u LD_LIBRARY_PATH $CPU_ENV python3 scripts/exp068_deep_search.py --phase aggregate \
    --horizon $H --out-dir "$OUT" | tee -a "$LOG" | tee -a experiments_log.md
done
say "EXP068_DONE"
