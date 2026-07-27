#!/usr/bin/env bash
# Fresh-window retrain (az_v9 follow-up probe): warm-start from the champion
# (az_v9/iter2/best.pt) and train ONLY on the newest two self-play generations
# (it5+it6, both collected by iter-2 itself) — no stale exp_021-era targets.
# Volume-matched to iter-2's own training set (9,600 train / 3,200 val).
#
# Decides between the two readings of the post-iter-2 plateau/decline:
#   * result ≈ or > iter-2 (0.5657±0.0048 / +0.234±0.023): stale-generation drag
#     caused the decline; windowed buffers unlock further ratcheting.
#   * result ~0.55 like iters 3-6: the flat new-generation targets themselves
#     can't sustain training — operator signal exhausted at 2-ply/120sims.
set -uo pipefail
cd /mnt/data/curling2/csas_world

WORK=checkpoints/csas_world/az_v9_freshwin
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

# --- build the fresh-window train/val dirs (it5+it6 only) ---
TRAIN=artifacts/replay/az_v9_freshwin_train
VAL=artifacts/replay/az_v9_freshwin_val
rm -rf "$TRAIN" "$VAL"; mkdir -p "$TRAIN" "$VAL"
for it in 5 6; do
  for k in 0 1 2; do
    ln -s "$(readlink -f artifacts/replay/mcts/az_v9_selfplay_iter$it/shard$k.npz)" "$TRAIN/it${it}_shard$k.npz"
  done
  ln -s "$(readlink -f artifacts/replay/mcts/az_v9_selfplay_iter$it/shard3.npz)" "$VAL/it${it}_shard3.npz"
done
echo "[freshwin] window: $(ls $TRAIN | wc -l) train files, $(ls $VAL | wc -l) val files (it5+it6 only)" | tee -a "$LOG"

source scripts/setup_gpu.sh
export PYTHONUNBUFFERED=1

echo "[freshwin] TRAIN from iter2/best.pt on fresh window at $(date -u +%H:%M)" | tee -a "$LOG"
unset LD_LIBRARY_PATH
PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing \
GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none \
python3 scripts/run_consolidate.py \
  --config configs/exp_021_valuemcts_earlystop.yaml \
  --union "$TRAIN" --mcts-val "$VAL" \
  --init checkpoints/csas_world/az_v9_selfplay/iter2/best.pt \
  --out "$WORK" >> "$WORK/train.log" 2>&1
echo "[freshwin] train rc=$?" | tee -a "$LOG"

CK="$WORK/best.pt"; [ -f "$CK" ] || CK="$WORK/model.pt"
echo "[freshwin] 3-draw eval of $CK" | tee -a "$LOG"
for d in 1 2 3; do
  OUT=eval_out/az_v9_freshwin/run$d
  [ -f "$OUT/summary.json" ] && continue
  mkdir -p "$OUT"
  unset LD_LIBRARY_PATH
  PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu \
  GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing \
  GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none \
  python3 scripts/_eval_parallel.py --champion "$CK" --vs prior \
    --N 400 --horizons 1,2,3,4,5,6,7,8,9,10 --gpus 0,1,2,3 --shards 4 --noisy \
    --out-dir "$OUT" >> "$WORK/eval.log" 2>&1
  echo "[freshwin] draw $d done" | tee -a "$LOG"
done
echo "FRESHWIN_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
