#!/usr/bin/env bash
# az_v14 — capacity scaling: 23.2M-param model (3.2x champion) FROM SCRATCH on the full
# clean corpus, az_v13-fixed selection recipe, 3-draw eval vs the champion.
set -uo pipefail
cd /mnt/data/curling2/csas_world

WORK=checkpoints/csas_world/az_v14_big
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

# ---- assemble the training corpus (symlinks; provenance per COLLECTIONS.md) ----
T=artifacts/replay/az_v14_train
V=artifacts/replay/az_v14_val
rm -rf "$T" "$V"; mkdir -p "$T" "$V"
for it in 1 2; do
  for k in 0 1 2; do
    ln -s "$(readlink -f artifacts/replay/mcts/az_v9_selfplay_iter$it/shard$k.npz)" "$T/v9it${it}_s$k.npz"
    ln -s "$(readlink -f artifacts/replay/mcts/az_v13_ratchet_iter$it/shard$k.npz)" "$T/v13it${it}_s$k.npz"
  done
  ln -s "$(readlink -f artifacts/replay/mcts/az_v9_selfplay_iter$it/shard3.npz)" "$V/v9it${it}_s3.npz"
  ln -s "$(readlink -f artifacts/replay/mcts/az_v13_ratchet_iter$it/shard3.npz)" "$V/v13it${it}_s3.npz"
done
for f in artifacts/replay/az_v13_conf_train/*.npz; do
  ln -s "$(readlink -f "$f")" "$T/conf_$(basename "$f")"
done
for f in artifacts/replay/az_v13_conf_val/*.npz; do
  ln -s "$(readlink -f "$f")" "$V/conf_$(basename "$f")"
done
echo "[launch] corpus: $(ls $T | wc -l) train files, $(ls $V | wc -l) val files" | tee -a "$LOG"

source scripts/setup_gpu.sh
export PYTHONUNBUFFERED=1
echo "[launch] az_v14 (23.2M from scratch) at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"

unset LD_LIBRARY_PATH
PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing \
GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none \
python3 scripts/run_consolidate.py \
  --config configs/exp_038_big_model.yaml \
  --union "$T" --mcts-val "$V" \
  --out "$WORK" >> "$WORK/train.log" 2>&1
echo "[v14] train rc=$?" | tee -a "$LOG"
grep -aE "early-stop|\[select\]" "$WORK/train.log" | tail -4 | tee -a "$LOG"

CK="$WORK/best.pt"; [ -f "$CK" ] || CK="$WORK/model.pt"
echo "[v14] 3-draw eval of $CK" | tee -a "$LOG"
for d in 1 2 3; do
  OUT=eval_out/az_v14_big/run$d
  [ -f "$OUT/summary.json" ] && continue
  mkdir -p "$OUT"
  unset LD_LIBRARY_PATH
  PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu \
  GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing \
  GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none \
  python3 scripts/_eval_parallel.py --champion "$CK" --vs prior \
    --N 400 --horizons 1,2,3,4,5,6,7,8,9,10 --gpus 0,1,2,3 --shards 4 --noisy \
    --out-dir "$OUT" >> "$WORK/eval.log" 2>&1
  echo "[v14] draw $d done" | tee -a "$LOG"
done
echo "AZ_V14_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
