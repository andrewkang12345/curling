#!/usr/bin/env bash
set -uo pipefail
cd /mnt/data/curling2/csas_world
source scripts/setup_gpu.sh
export PYTHONUNBUFFERED=1
CH=checkpoints/csas_world/az_v9_selfplay/iter2/best.pt
V13=checkpoints/csas_world/az_v13_ratchet/iter1/best.pt
V14=checkpoints/csas_world/az_v14d/best.pt
for RUN in 2 3; do
  for P in "champ_vs_v14d:$CH:$V14" "v13it1_vs_v14d:$V13:$V14"; do
    NAME="${P%%:*}"; REST="${P#*:}"; A="${REST%%:*}"; B="${REST##*:}"
    OUT="eval_out/metagame/${NAME}_run$RUN"
    [ -f "$OUT/done" ] && continue
    mkdir -p "$OUT"
    unset LD_LIBRARY_PATH
    PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu \
    GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing \
    GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none \
    python3 scripts/_eval_parallel.py --champion "$A" --vs "$B" \
      --N 400 --horizons 1,2,3,4,5,6,7,8,9,10 --gpus 0,1,2,3 --shards 4 --noisy \
      --out-dir "$OUT" >> eval_out/metagame/confirm.log 2>&1
    touch "$OUT/done"
    echo "$OUT done" >> eval_out/metagame/confirm.log
  done
done
echo "CONFIRM_DONE" >> eval_out/metagame/confirm.log
