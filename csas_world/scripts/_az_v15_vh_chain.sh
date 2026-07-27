#!/usr/bin/env bash
# Chained after the L8 leg: value-head-only continuation of v13it1 on the L4-leg data,
# early-stopped on its own (aligned) val loss, then 3-draw h2h gate vs v13it1.
set -uo pipefail
cd /mnt/data/curling2/csas_world
while ! grep -q "SCALING_EDGE_DONE" eval_out/metagame/scaling_edge.log 2>/dev/null; do sleep 600; done
sleep 20
WORK=checkpoints/csas_world/az_v15_vh
mkdir -p "$WORK"; LOG="$WORK/run.log"; LOCK="$WORK/launcher.pid"
if [ -f "$LOCK" ]; then p=$(cat "$LOCK"); [ -n "$p" ] && kill -0 "$p" 2>/dev/null && exit 1; fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
source scripts/setup_gpu.sh
export PYTHONUNBUFFERED=1
ENVV="GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
INC=checkpoints/csas_world/az_v13_ratchet/iter1/best.pt
echo "[vh] value-head-only train from v13it1 at $(date -u +%H:%M)" | tee -a "$LOG"
unset LD_LIBRARY_PATH
env $ENVV python3 scripts/run_consolidate.py \
  --config configs/exp_046_valuehead_only.yaml \
  --union artifacts/replay/az_v15_L4_accum_train --mcts-val artifacts/replay/az_v15_L4_accum_val \
  --init "$INC" --out "$WORK" >> "$WORK/train.log" 2>&1
echo "[vh] train rc=$?" | tee -a "$LOG"
grep -aE "early-stop|freeze" "$WORK/train.log" | tail -3 | tee -a "$LOG"
CK="$WORK/best.pt"; [ -f "$CK" ] || CK="$WORK/model.pt"
for d in 1 2 3; do
  OUT=eval_out/az_v15_vh/run$d
  [ -f "$OUT/summary.json" ] && continue
  mkdir -p "$OUT"
  unset LD_LIBRARY_PATH
  env $ENVV python3 scripts/_eval_parallel.py --champion "$CK" --vs "$INC" \
    --N 400 --horizons 1,2,3,4,5,6,7,8,9,10 --gpus 0,1,2,3 --shards 4 --noisy \
    --out-dir "$OUT" >> "$WORK/eval.log" 2>&1
  echo "[vh] draw $d done" | tee -a "$LOG"
done
echo "AZ_V15_VH_DONE $(date -u +%H:%M)" | tee -a "$LOG"
