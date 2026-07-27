#!/usr/bin/env bash
# az_v16 creation: L12 (19.99M) = v14d + 4 near-identity layers, fine-tuned on the
# az_v14 corpus (same data as v14d's own creation — clean +capacity analogy), then
# 3-draw h2h gate vs v14d.
set -uo pipefail
cd /mnt/data/curling2/csas_world
WORK=checkpoints/csas_world/az_v16_create
mkdir -p "$WORK"; LOG="$WORK/run.log"; LOCK="$WORK/launcher.pid"
if [ -f "$LOCK" ]; then p=$(cat "$LOCK"); [ -n "$p" ] && kill -0 "$p" 2>/dev/null && exit 1; fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
source scripts/setup_gpu.sh
export PYTHONUNBUFFERED=1
ENVV="GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "[v16] L12 creation fine-tune at $(date -u +%H:%M)" | tee -a "$LOG"
unset LD_LIBRARY_PATH
env $ENVV python3 scripts/run_consolidate.py \
  --config configs/exp_047_L12_create.yaml \
  --union artifacts/replay/az_v14_train --mcts-val artifacts/replay/az_v14_val \
  --init checkpoints/csas_world/az_v16_seed.pt --out "$WORK" >> "$WORK/train.log" 2>&1
echo "[v16] train rc=$?" | tee -a "$LOG"
CK="$WORK/best.pt"; [ -f "$CK" ] || CK="$WORK/model.pt"
for d in 1 2 3; do
  OUT=eval_out/az_v16_create/run$d
  [ -f "$OUT/summary.json" ] && continue
  mkdir -p "$OUT"
  unset LD_LIBRARY_PATH
  env $ENVV python3 scripts/_eval_parallel.py --champion "$CK" --vs checkpoints/csas_world/az_v14d/best.pt \
    --N 400 --horizons 1,2,3,4,5,6,7,8,9,10 --gpus 0,1,2,3 --shards 4 --noisy \
    --out-dir "$OUT" >> "$WORK/eval.log" 2>&1
  echo "[v16] draw $d done" | tee -a "$LOG"
done
echo "AZ_V16_CREATE_DONE $(date -u +%H:%M)" | tee -a "$LOG"
