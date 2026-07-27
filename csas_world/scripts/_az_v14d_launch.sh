#!/usr/bin/env bash
set -uo pipefail
cd /mnt/data/curling2/csas_world
WORK=checkpoints/csas_world/az_v14d
mkdir -p "$WORK"; LOG="$WORK/run.log"; LOCK="$WORK/launcher.pid"
if [ -f "$LOCK" ]; then p=$(cat "$LOCK"); [ -n "$p" ] && kill -0 "$p" 2>/dev/null && { echo "[launch] REFUSING: $p alive" | tee -a "$LOG"; exit 1; }; fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
source scripts/setup_gpu.sh
export PYTHONUNBUFFERED=1
ENVV="GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "[launch] az_v14d (depth-extended champion, 13.65M) at $(date -u +%H:%M)" | tee -a "$LOG"
unset LD_LIBRARY_PATH
env $ENVV python3 scripts/run_consolidate.py \
  --config configs/exp_041_depth_extend.yaml \
  --union artifacts/replay/az_v14_train --mcts-val artifacts/replay/az_v14_val \
  --init checkpoints/csas_world/az_v14d_seed.pt \
  --out "$WORK" >> "$WORK/train.log" 2>&1
echo "[v14d] train rc=$?" | tee -a "$LOG"
CK="$WORK/best.pt"; [ -f "$CK" ] || CK="$WORK/model.pt"
for d in 1 2 3; do
  OUT=eval_out/az_v14d/run$d
  [ -f "$OUT/summary.json" ] && continue
  mkdir -p "$OUT"
  unset LD_LIBRARY_PATH
  env $ENVV python3 scripts/_eval_parallel.py --champion "$CK" --vs prior \
    --N 400 --horizons 1,2,3,4,5,6,7,8,9,10 --gpus 0,1,2,3 --shards 4 --noisy \
    --out-dir "$OUT" >> "$WORK/eval.log" 2>&1
  echo "[v14d] draw $d done" | tee -a "$LOG"
done
echo "AZ_V14D_DONE $(date -u +%H:%M)" | tee -a "$LOG"
