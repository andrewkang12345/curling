#!/usr/bin/env bash
# After the VH run: 4 more draws of v13it1-vs-v14d — the converged-L4 vs converged-L8
# edge IS the scaling-study headline, currently unresolved (-0.032 +/- 0.046, 3 draws).
set -uo pipefail
cd /mnt/data/curling2/csas_world
while ! grep -q "AZ_V15_VH_DONE" checkpoints/csas_world/az_v15_vh/run.log 2>/dev/null; do sleep 600; done
sleep 20
source scripts/setup_gpu.sh
export PYTHONUNBUFFERED=1
V13=checkpoints/csas_world/az_v13_ratchet/iter1/best.pt
V14=checkpoints/csas_world/az_v14d/best.pt
for RUN in 4 5 6 7; do
  OUT="eval_out/metagame/v13it1_vs_v14d_run$RUN"
  [ -f "$OUT/done" ] && continue
  mkdir -p "$OUT"
  unset LD_LIBRARY_PATH
  PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu \
  GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing \
  GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none \
  python3 scripts/_eval_parallel.py --champion "$V13" --vs "$V14" \
    --N 400 --horizons 1,2,3,4,5,6,7,8,9,10 --gpus 0,1,2,3 --shards 4 --noisy \
    --out-dir "$OUT" >> eval_out/metagame/scaling_edge.log 2>&1
  touch "$OUT/done"; echo "run$RUN done" >> eval_out/metagame/scaling_edge.log
done
echo "SCALING_EDGE_DONE" >> eval_out/metagame/scaling_edge.log
