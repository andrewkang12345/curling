#!/usr/bin/env bash
# EXP-072: root-integration sweep — root_out_cap in {8,32,64,256} x 2 seeds, on the SAME
# 30 h=10 states, everything else identical to EXP-069's collection config (4-ply, raw
# tail). Tests the EXP-071 hypothesis that a fixed root outcome cap (8) bounded the root
# expectation's accuracy regardless of budget, making regret rise with more search.
set -uo pipefail
cd /mnt/data/curling2/csas_world
OUT=eval_out/exp072_rootsweep
LOG="$OUT/chain.log"
mkdir -p "$OUT/h10"
export PYTHONUNBUFFERED=1 WORLD_BOUNDARY_REMOVAL=1
ENV="PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu VALUE_EVAL_BATCH=128 \
POLICY_BATCH_CAP=256 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing GNN_NODE_FEATURE_MODE=none \
GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none"
say() { echo "[exp072] $* $(date -u +%H:%M)" | tee -a "$LOG"; }

say "sweep arms start (4 caps x 2 seeds x 30 states)"
pids=()
for k in $(seq 0 11); do
  env -u LD_LIBRARY_PATH $ENV CUDA_VISIBLE_DEVICES=$((k % 4)) \
    python3 scripts/exp068_deep_search.py --phase rootsweep --horizon 10 \
    --shard-id $k --num-shards 12 --out-dir "$OUT" >> "$OUT/h10/sweep_shard$k.log" 2>&1 &
  pids+=($!)
  sleep 3
done
wait "${pids[@]}" || true
say "sweep arms done"

say "strong (game-value) adjudication of the new action union"
pids=()
for k in $(seq 0 11); do
  env -u LD_LIBRARY_PATH $ENV CUDA_VISIBLE_DEVICES=$((k % 4)) \
    python3 scripts/exp068_deep_search.py --phase adj_strong --horizon 10 --adj-only-budget 16000 \
    --shard-id $k --num-shards 12 --out-dir "$OUT" >> "$OUT/h10/adjs_shard$k.log" 2>&1 &
  pids+=($!)
  sleep 3
done
wait "${pids[@]}" || true
say "adjudication done"

env -u LD_LIBRARY_PATH $ENV python3 scripts/exp068_deep_search.py --phase aggregate \
  --horizon 10 --out-dir "$OUT" | tee -a "$LOG" | tee -a experiments_log.md
say "EXP072_DONE"
