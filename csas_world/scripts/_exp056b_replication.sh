#!/usr/bin/env bash
# EXP-056b: fresh-roots replication of the StT-vs-record primary (queued behind EXP-057).
set -uo pipefail
cd /mnt/data/curling2/csas_world
mkdir -p eval_out/exp056b
while ! grep -aq "EXP057_DONE" eval_out/exp057_k8/chain.log 2>/dev/null; do sleep 600; done
echo "[exp056b] start $(date -u +%FT%TZ)"
for k in 0 1 2 3; do
  [ "$k" -gt 0 ] && sleep 40
  unset LD_LIBRARY_PATH
  PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu VALUE_EVAL_BATCH=64 \
  CUDA_VISIBLE_DEVICES=0 POLICY_BATCH_CAP=96 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing \
  GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none \
  nohup python3 scripts/exp056_rollout_estimator.py --shard-id "$k" --num-shards 4 --device cuda:0 \
    --arms StT --seed 57 --out-dir eval_out/exp056b \
    > "eval_out/exp056b/shard$k.log" 2>&1 &
done
wait || true
echo "EXP056B_DONE $(date -u +%FT%TZ)"
