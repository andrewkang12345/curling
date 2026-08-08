#!/usr/bin/env bash
# EXP-068b: STRONG-PLAY adjudication (user directive 2026-08-08) — forced root action,
# then deep minimax stochastic search by BOTH sides; regret measured on that table.
set -uo pipefail
cd /mnt/data/curling2/csas_world
OUT=eval_out/exp068_deep
LOG="$OUT/strong.log"
export PYTHONUNBUFFERED=1 WORLD_BOUNDARY_REMOVAL=1
ENV="PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu VALUE_EVAL_BATCH=128 \
POLICY_BATCH_CAP=256 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing GNN_NODE_FEATURE_MODE=none \
GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none"
say() { echo "[exp068b] $* $(date -u +%H:%M)" | tee -a "$LOG"; }

say "h4 strong adjudication start"
pids=()
for k in $(seq 0 11); do
  env -u LD_LIBRARY_PATH $ENV CUDA_VISIBLE_DEVICES=$((k % 4)) \
    python3 scripts/exp068_deep_search.py --phase adj_strong --horizon 4 \
    --shard-id $k --num-shards 12 --out-dir "$OUT" >> "$OUT/h04/adjs_shard$k.log" 2>&1 &
  pids+=($!); sleep 3
done
wait "${pids[@]}" || true
say "h4 strong done"
env -u LD_LIBRARY_PATH $ENV python3 scripts/exp068_deep_search.py --phase aggregate \
  --horizon 4 --out-dir "$OUT" | tee -a "$LOG" | tee -a experiments_log.md

# h=10: wait for the main chain's search+champion-adjudication to drain
while pgrep -f "_exp068_chai[n]" > /dev/null; do sleep 300; done
say "h10 strong adjudication start"
pids=()
for k in $(seq 0 11); do
  env -u LD_LIBRARY_PATH $ENV CUDA_VISIBLE_DEVICES=$((k % 4)) \
    python3 scripts/exp068_deep_search.py --phase adj_strong --horizon 10 \
    --shard-id $k --num-shards 12 --out-dir "$OUT" >> "$OUT/h10/adjs_shard$k.log" 2>&1 &
  pids+=($!); sleep 3
done
wait "${pids[@]}" || true
say "h10 strong done"
env -u LD_LIBRARY_PATH $ENV python3 scripts/exp068_deep_search.py --phase aggregate \
  --horizon 10 --out-dir "$OUT" | tee -a "$LOG" | tee -a experiments_log.md
say "EXP068B_DONE"
