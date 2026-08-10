#!/usr/bin/env bash
# EXP-071: TAIL ABLATION at fixed budget (h=10, 30 states; arms differ ONLY in how a
# node at the search-depth cap is evaluated). Motivation: EXP-068's deployment-value
# gap was negligible where the tree has no tail (h=4: 0.040 +- 0.086) and large where
# it does (h=10: 0.110 +- 0.065), so the raw-policy leaf continuation is the leading
# suspect for why EXP-069's vectree-trained champion failed its gate.
#   arms: tail_raw (what EXP-069 used) | tail_vgreedy | tail_vleaf | tail_vleaf_d6
#   baselines carried over from EXP-068: flat_width, ref(64k)
# Then re-adjudicate the new action union under BOTH estimands (game value, deployment).
set -uo pipefail
cd /mnt/data/curling2/csas_world
OUT=eval_out/exp071_tails
LOG="$OUT/chain.log"
mkdir -p "$OUT/h10"
export PYTHONUNBUFFERED=1 WORLD_BOUNDARY_REMOVAL=1
ENV="PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu VALUE_EVAL_BATCH=128 \
POLICY_BATCH_CAP=256 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing GNN_NODE_FEATURE_MODE=none \
GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none"
say() { echo "[exp071] $* $(date -u +%H:%M)" | tee -a "$LOG"; }

say "tail arms start"
pids=()
for k in $(seq 0 11); do
  env -u LD_LIBRARY_PATH $ENV CUDA_VISIBLE_DEVICES=$((k % 4)) \
    python3 scripts/exp068_deep_search.py --phase tails --horizon 10 \
    --shard-id $k --num-shards 12 --out-dir "$OUT" >> "$OUT/h10/tails_shard$k.log" 2>&1 &
  pids+=($!)
  sleep 3
done
wait "${pids[@]}" || true
say "tail arms done"

say "strong (game-value) adjudication start"
pids=()
for k in $(seq 0 11); do
  env -u LD_LIBRARY_PATH $ENV CUDA_VISIBLE_DEVICES=$((k % 4)) \
    python3 scripts/exp068_deep_search.py --phase adj_strong --horizon 10 --adj-only-budget 16000 \
    --shard-id $k --num-shards 12 --out-dir "$OUT" >> "$OUT/h10/adjs_shard$k.log" 2>&1 &
  pids+=($!)
  sleep 3
done
wait "${pids[@]}" || true
say "strong adjudication done"

say "deployment adjudication start"
pids=()
for k in $(seq 0 7); do
  env -u LD_LIBRARY_PATH $ENV CUDA_VISIBLE_DEVICES=$((k % 4)) \
    python3 scripts/exp068_deep_search.py --phase adjudicate --horizon 10 --adj-only-budget 16000 \
    --shard-id $k --num-shards 8 --out-dir "$OUT" >> "$OUT/h10/adj_shard$k.log" 2>&1 &
  pids+=($!)
  sleep 3
done
wait "${pids[@]}" || true
say "deployment adjudication done"

env -u LD_LIBRARY_PATH $ENV python3 scripts/exp068_deep_search.py --phase aggregate \
  --horizon 10 --out-dir "$OUT" | tee -a "$LOG" | tee -a experiments_log.md
say "EXP071_DONE"
