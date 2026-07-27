#!/usr/bin/env bash
# az_v18: the paused-collection big-corpus training tests (single-GPU box).
#   Run 1 (L8 / data-wall): fine-tune az_v14d on the fresh corpus, 3-draw h2h gate vs v14d.
#   Run 2 (L12 / wake-up):  fine-tune az_v16_create on the same corpus, same gate.
# Eval: 4 shard-workers share GPU 0 (--gpus 0,0,0,0) — eval is JAX-CPU-sim dominated.
set -uo pipefail
cd /mnt/data/curling2/csas_world
WORK_BASE=checkpoints/csas_world
LOG=checkpoints/csas_world/az_v18_tests.log
LOCK=checkpoints/csas_world/az_v18.pid
if [ -f "$LOCK" ]; then p=$(cat "$LOCK"); [ -n "$p" ] && kill -0 "$p" 2>/dev/null && { echo REFUSING | tee -a "$LOG"; exit 1; }; fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT

# wait for collection drain, then REBUILD the split to catch the final round
while pgrep -f "world.search.selfplay" > /dev/null; do sleep 120; done
T=artifacts/replay/az_v18_train; V=artifacts/replay/az_v18_val
rm -rf "$T" "$V"; mkdir -p "$T" "$V"
for f in artifacts/replay/mcts/az_v17_bigcorpus/r*_shard*.npz; do
  b=$(basename "$f"); k=${b##*shard}; k=${k%.npz}
  if [ "$k" = "4" ]; then ln -s "$(readlink -f "$f")" "$V/$b"; else ln -s "$(readlink -f "$f")" "$T/$b"; fi
done
echo "[v18] split: $(ls $T | wc -l) train / $(ls $V | wc -l) val shards" | tee -a "$LOG"

source scripts/setup_gpu.sh
export PYTHONUNBUFFERED=1
ENVV="GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
INC=checkpoints/csas_world/az_v14d/best.pt

run_test() {
  local NAME="$1" CFG="$2" INIT="$3"
  local WORK="$WORK_BASE/$NAME"
  mkdir -p "$WORK"
  echo "[v18] TRAIN $NAME at $(date -u +%H:%M)" | tee -a "$LOG"
  unset LD_LIBRARY_PATH
  env $ENVV python3 scripts/run_consolidate.py \
    --config "$CFG" --union "$T" --mcts-val "$V" \
    --init "$INIT" --out "$WORK" >> "$WORK/train.log" 2>&1
  echo "[v18] $NAME train rc=$?" | tee -a "$LOG"
  grep -aE "early-stop" "$WORK/train.log" | tail -1 | tee -a "$LOG"
  local CK="$WORK/best.pt"; [ -f "$CK" ] || CK="$WORK/model.pt"
  for d in 1 2 3; do
    local OUT="eval_out/$NAME/run$d"
    [ -f "$OUT/summary.json" ] && continue
    mkdir -p "$OUT"
    unset LD_LIBRARY_PATH
    env $ENVV python3 scripts/_eval_parallel.py --champion "$CK" --vs "$INC" \
      --N 400 --horizons 1,2,3,4,5,6,7,8,9,10 --gpus 0,0,0,0 --shards 4 --noisy \
      --out-dir "$OUT" >> "$WORK/eval.log" 2>&1
    echo "[v18] $NAME draw $d done $(date -u +%H:%M)" | tee -a "$LOG"
  done
}

run_test az_v18_L8  configs/exp_048_L8_bigcorpus.yaml  "$INC"
run_test az_v18_L12 configs/exp_049_L12_bigcorpus.yaml checkpoints/csas_world/az_v16_create/best.pt

echo "AZ_V18_TESTS_DONE $(date -u +%FT%TZ)" | tee -a "$LOG"
