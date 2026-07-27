#!/usr/bin/env bash
# EXP-053 launcher: 4 resumable shards of the depth-certification study.
set -uo pipefail
cd /mnt/data/curling2/csas_world
mkdir -p eval_out/exp053_depth
for k in 0 1 2 3; do
  [ "$k" -gt 0 ] && sleep 45
  unset LD_LIBRARY_PATH
  PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu \
  CUDA_VISIBLE_DEVICES=0 POLICY_BATCH_CAP=96 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing \
  GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none \
  nohup python3 scripts/exp053_depth_cert.py --shard-id "$k" --num-shards 4 --device cuda:0 \
    > "eval_out/exp053_depth/shard$k.log" 2>&1 &
  echo "shard $k pid $!"
done
