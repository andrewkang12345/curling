#!/usr/bin/env bash
# PSRO step 0: meta-game payoff matrix. World-vs-world noisy h2h for all new pairs
# among {exp_021, champion, az_v13-it1, az_v14d}. (vs-prior column already measured.)
set -uo pipefail
cd /mnt/data/curling2/csas_world
source scripts/setup_gpu.sh
export PYTHONUNBUFFERED=1
LOG=eval_out/metagame/run.log
mkdir -p eval_out/metagame

declare -A CK=(
  [exp021]=checkpoints/csas_world/exp_021_valuemcts_earlystop/best.pt
  [champ]=checkpoints/csas_world/az_v9_selfplay/iter2/best.pt
  [v13it1]=checkpoints/csas_world/az_v13_ratchet/iter1/best.pt
  [v14d]=checkpoints/csas_world/az_v14d/best.pt
)
PAIRS="exp021:champ exp021:v13it1 exp021:v14d champ:v13it1 champ:v14d v13it1:v14d"
for P in $PAIRS; do
  A="${P%%:*}"; B="${P##*:}"
  OUT="eval_out/metagame/${A}_vs_${B}"
  [ -f "$OUT/done" ] && continue
  mkdir -p "$OUT"
  echo ">>> $A vs $B" | tee -a "$LOG"
  unset LD_LIBRARY_PATH
  PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu \
  GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing \
  GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none \
  python3 scripts/_eval_parallel.py --champion "${CK[$A]}" --vs "${CK[$B]}" \
    --N 400 --horizons 1,2,3,4,5,6,7,8,9,10 --gpus 0,1,2,3 --shards 4 --noisy \
    --out-dir "$OUT" >> "$LOG" 2>&1
  touch "$OUT/done"
done
echo "METAGAME_DONE" | tee -a "$LOG"
