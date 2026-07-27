#!/usr/bin/env bash
# 4 additional independent draws of az_v9/iter2/best.pt (adds to its existing 3) to
# confirm the deploy champion's numbers before the paper. Serial, ~50min each.
set -uo pipefail
cd /mnt/data/curling2/csas_world
source scripts/setup_gpu.sh
for d in 4 5 6 7; do
  OUT=eval_out/az_v9_selfplay/iter1... ; OUT=eval_out/az_v9_selfplay/iter2_run$d
  [ -f "$OUT/summary.json" ] && continue
  mkdir -p "$OUT"
  unset LD_LIBRARY_PATH
  PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu \
  GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing \
  GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none \
  python3 scripts/_eval_parallel.py --champion checkpoints/csas_world/az_v9_selfplay/iter2/best.pt \
    --vs prior --N 400 --horizons 1,2,3,4,5,6,7,8,9,10 --gpus 0,1,2,3 --shards 4 --noisy \
    --out-dir "$OUT" >> eval_out/az_v9_selfplay/iter2_confirm.log 2>&1
done
echo "ITER2_CONFIRM_DONE" >> eval_out/az_v9_selfplay/iter2_confirm.log
