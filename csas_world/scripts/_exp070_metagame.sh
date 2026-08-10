#!/usr/bin/env bash
# EXP-070: POPULATION META-GAME MATRIX + robustness (maximin) certification.
#
# Motivation (user, 2026-08-09): our promotion rule is head-to-head dScore vs the
# incumbent, which REWARDS EXPLOITATION and is blind to robustness — a perfect Nash
# policy would read as "parity" against everyone. EXP-069's az_v27 (minimax/game-value
# targets) lost the head-to-head by -0.057; that is consistent with it being worse OR
# with it being MORE Nash-like. The distinguishing measurement is the population
# cross-play matrix: compare candidates on MAXIMIN (worst-case dScore over opponents)
# and look for intransitive cycles.
#
# All pairs are NEW-RULES, k=4 selection, N=150 x 10 horizons x 2 orders (3,000 ends,
# SE ~0.03/end) — reduced N because we need ~20 cells, not one certified promotion.
set -uo pipefail
cd /mnt/data/curling2/csas_world
OUT=eval_out/exp070_meta
LOG="$OUT/chain.log"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1 WORLD_BOUNDARY_REMOVAL=1
ENV="PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu VALUE_EVAL_BATCH=64 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing GNN_NODE_FEATURE_MODE=none \
GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none"
C=checkpoints/csas_world
declare -A CK=(
  [v14d]="$C/az_v14d/best.pt"
  [v19]="$C/az_v19_newrules/best.pt"
  [v21]="$C/az_v21_stt2x/best.pt"
  [v25]="$C/az_v25_br/best.pt"
  [v26]="$C/az_v26_br2/best.pt"
  [v27]="$C/az_v27_vectree/best.pt"
)
NAMES=(v14d v19 v21 v25 v26 v27)
echo "[exp070] start $(date -u +%FT%TZ): ${#NAMES[@]} models, $(( ${#NAMES[@]} * (${#NAMES[@]} - 1) / 2 )) pairs" | tee -a "$LOG"

gpu=0
for i in "${!NAMES[@]}"; do
  for j in "${!NAMES[@]}"; do
    [ "$j" -le "$i" ] && continue
    A=${NAMES[$i]}; B=${NAMES[$j]}
    O="$OUT/${A}_vs_${B}"
    [ -f "$O/summary.json" ] && continue
    mkdir -p "$O"
    env -u LD_LIBRARY_PATH $ENV python3 scripts/_eval_parallel.py \
      --champion "${CK[$A]}" --vs "${CK[$B]}" \
      --N 150 --horizons 1,2,3,4,5,6,7,8,9,10 --gpus $gpu,$gpu --shards 2 --noisy --sel-noise 4 \
      --out-dir "$O" >> "$O/run.log" 2>&1 &
    gpu=$(( (gpu + 1) % 4 ))
    # keep 4 pair-jobs in flight (one per GPU)
    while [ "$(jobs -rp | wc -l)" -ge 4 ]; do sleep 20; done
  done
done
wait || true
echo "[exp070] all pairs done $(date -u +%FT%TZ)" | tee -a "$LOG"

env -u LD_LIBRARY_PATH $ENV python3 scripts/exp070_analyze.py --out-dir "$OUT" \
  | tee -a "$LOG" | tee -a experiments_log.md
echo "EXP070_DONE $(date -u +%FT%TZ)" | tee -a "$LOG"
