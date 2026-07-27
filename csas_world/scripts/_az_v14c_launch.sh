#!/usr/bin/env bash
# az_v14b — capacity scaling, take 2 (distill-then-finetune):
#   phase 0: build the champion-distillation set (24-sample policy targets + V targets
#            on the 44k corpus states) — _build_champion_distill_set.py
#   phase 1: distill champion -> 23.2M model (exp_039)
#   phase 2: az_v13-recipe fine-tune on the real corpus (exp_040, --init phase-1 best)
#   then 3-draw eval vs prior.
set -uo pipefail
cd /mnt/data/curling2/csas_world

WORK=checkpoints/csas_world/az_v14c
mkdir -p "$WORK/phase1" "$WORK/phase2"
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
ENVV="GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

echo "[launch] az_v14b at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"

if [ ! -f artifacts/replay/az_v14b_distill_train/shard000.npz ]; then
  echo "[v14b] phase 0: building distillation set" | tee -a "$LOG"
  unset LD_LIBRARY_PATH
  env $ENVV python3 scripts/_build_champion_distill_set.py >> "$WORK/phase0.log" 2>&1
  grep -a "DISTILL_SET_DONE" "$WORK/phase0.log" >/dev/null || { echo "[v14b] PHASE 0 FAILED" | tee -a "$LOG"; exit 1; }
  tail -2 "$WORK/phase0.log" | tee -a "$LOG"
fi

echo "[v14b] phase 1: distill champion -> 23.2M" | tee -a "$LOG"
unset LD_LIBRARY_PATH
env $ENVV python3 scripts/run_consolidate.py \
  --config configs/exp_039_distill_champion.yaml \
  --union artifacts/replay/az_v14b_distill_train \
  --mcts-val artifacts/replay/az_v14b_distill_val \
  --init checkpoints/csas_world/az_v14c_seed.pt \
  --out "$WORK/phase1" >> "$WORK/phase1.log" 2>&1
echo "[v14b] phase 1 rc=$?" | tee -a "$LOG"
CK1="$WORK/phase1/best.pt"; [ -f "$CK1" ] || CK1="$WORK/phase1/model.pt"
[ -f "$CK1" ] || { echo "[v14b] PHASE 1 produced no ckpt" | tee -a "$LOG"; exit 1; }

echo "[v14b] phase 2: fine-tune on the real corpus" | tee -a "$LOG"
unset LD_LIBRARY_PATH
env $ENVV python3 scripts/run_consolidate.py \
  --config configs/exp_040_big_finetune.yaml \
  --union artifacts/replay/az_v14_train \
  --mcts-val artifacts/replay/az_v14_val \
  --init "$CK1" \
  --out "$WORK/phase2" >> "$WORK/phase2.log" 2>&1
echo "[v14b] phase 2 rc=$?" | tee -a "$LOG"
CK2="$WORK/phase2/best.pt"; [ -f "$CK2" ] || CK2="$WORK/phase2/model.pt"

echo "[v14b] 3-draw eval of $CK2" | tee -a "$LOG"
for d in 1 2 3; do
  OUT=eval_out/az_v14c/run$d
  [ -f "$OUT/summary.json" ] && continue
  mkdir -p "$OUT"
  unset LD_LIBRARY_PATH
  env $ENVV python3 scripts/_eval_parallel.py --champion "$CK2" --vs prior \
    --N 400 --horizons 1,2,3,4,5,6,7,8,9,10 --gpus 0,1,2,3 --shards 4 --noisy \
    --out-dir "$OUT" >> "$WORK/eval.log" 2>&1
  echo "[v14b] draw $d done" | tee -a "$LOG"
done
echo "AZ_V14C_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
