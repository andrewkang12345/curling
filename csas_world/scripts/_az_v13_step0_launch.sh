#!/usr/bin/env bash
# az_v13 Step 0 — the degradation-channel-fixed fine-tune of the champion, on EXISTING
# champion-generation data (zero recollection):
#   fix 1: significance-filtered distillation (az_v13_conf_{train,val} buffers)
#   fix 2: val-driven early stopping (patience=4) + value-drift selection guard (2.70)
#   fix 3: checkpoint selection by confident-distill val metric
# Success criteria: retrain lands AT the champion (channels fixed; the ~0.01-0.02 universal
# retrain-churn eliminated) or ABOVE the gate (residual signal existed -> loop extends).
set -uo pipefail
cd /mnt/data/curling2/csas_world

WORK=checkpoints/csas_world/az_v13_step0
mkdir -p "$WORK"
LOG="$WORK/run.log"
LOCK="$WORK/launcher.pid"

if [ -f "$LOCK" ]; then
  prev=$(cat "$LOCK")
  if [ -n "$prev" ] && kill -0 "$prev" 2>/dev/null; then
    echo "[launch] REFUSING: pid $prev alive" | tee -a "$LOG"; exit 1
  fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

source scripts/setup_gpu.sh
export PYTHONUNBUFFERED=1
echo "[launch] az_v13 Step 0 at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"

unset LD_LIBRARY_PATH
PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing \
GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none \
python3 scripts/run_consolidate.py \
  --config configs/exp_036_confident_finetune.yaml \
  --union artifacts/replay/az_v13_conf_train \
  --mcts-val artifacts/replay/az_v13_conf_val \
  --init checkpoints/csas_world/az_v9_selfplay/iter2/best.pt \
  --out "$WORK" >> "$WORK/train.log" 2>&1
echo "[step0] train rc=$?" | tee -a "$LOG"
grep -aE "early-stop|\[select\]" "$WORK/train.log" | tail -5 | tee -a "$LOG"

CK="$WORK/best.pt"; [ -f "$CK" ] || CK="$WORK/model.pt"
echo "[step0] 3-draw eval of $CK" | tee -a "$LOG"
for d in 1 2 3; do
  OUT=eval_out/az_v13_step0/run$d
  [ -f "$OUT/summary.json" ] && continue
  mkdir -p "$OUT"
  unset LD_LIBRARY_PATH
  PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu \
  GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing \
  GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none \
  python3 scripts/_eval_parallel.py --champion "$CK" --vs prior \
    --N 400 --horizons 1,2,3,4,5,6,7,8,9,10 --gpus 0,1,2,3 --shards 4 --noisy \
    --out-dir "$OUT" >> "$WORK/eval.log" 2>&1
  echo "[step0] draw $d done" | tee -a "$LOG"
done
echo "AZ_V13_STEP0_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
