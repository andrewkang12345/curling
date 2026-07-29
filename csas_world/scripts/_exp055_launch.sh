#!/usr/bin/env bash
# EXP-055 chain: wait for the old-rules regression draw to finish, then run
# mode=weak (4 shards) and mode=sb (4 shards) sequentially.
set -uo pipefail
cd /mnt/data/curling2/csas_world
while pgrep -f "_eval_hig[h]N" > /dev/null; do sleep 300; done
ENVV="PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=0 \
POLICY_BATCH_CAP=96 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing \
GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none"
for MODE in weak sb; do
  pids=()
  for k in 0 1 2 3; do
    [ "$k" -gt 0 ] && sleep 45
    unset LD_LIBRARY_PATH
    env $ENVV nohup python3 scripts/exp055_depth_diagnosis.py --mode "$MODE" \
      --shard-id "$k" --num-shards 4 --device cuda:0 \
      > "eval_out/exp055_${MODE}/shard$k.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}" || true
  echo "EXP055_${MODE}_ALL_DONE $(date -u +%FT%TZ)"
done
echo "EXP055_CHAIN_DONE $(date -u +%FT%TZ)"
