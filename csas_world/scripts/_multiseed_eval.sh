#!/usr/bin/env bash
# Multi-seed eval: top up the deploy-shortlist candidates to >=3 independent N=400 eval
# draws each. az_v6/iter1 (3 draws) and exp_021 best (4 recorded draws) are already covered;
# this runs the 4 missing draws: 2x az_v5/iter1, 2x exp_019 last.pt.
# Each _eval_parallel.py run is one independent draw (sample_actions_z is unseeded).
set -uo pipefail
cd /mnt/data/curling2/csas_world

LOCK=eval_out/multiseed/launcher.pid
mkdir -p eval_out/multiseed
if [ -f "$LOCK" ]; then
  prev=$(cat "$LOCK")
  if [ -n "$prev" ] && kill -0 "$prev" 2>/dev/null; then
    echo "[multiseed] REFUSING: already running (pid $prev)"; exit 1
  fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

source scripts/setup_gpu.sh
export PYTHONUNBUFFERED=1
LOG=eval_out/multiseed/run.log
echo "[multiseed] starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"

run_eval() {
  local CKPT="$1"; local OUT="$2"
  if [ -f "$OUT/summary.json" ]; then
    echo "[multiseed] skip $OUT (already done)" | tee -a "$LOG"; return
  fi
  echo "[multiseed] eval $CKPT -> $OUT" | tee -a "$LOG"
  mkdir -p "$OUT"
  unset LD_LIBRARY_PATH
  PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu \
  GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing \
  GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none \
  python3 scripts/_eval_parallel.py --champion "$CKPT" --vs prior \
    --N 400 --horizons 1,2,3,4,5,6,7,8,9,10 --gpus 0,1,2,3 --shards 4 --noisy \
    --out-dir "$OUT" >> "$LOG" 2>&1
  echo "[multiseed] done $OUT rc=$?" | tee -a "$LOG"
}

run_eval checkpoints/csas_world/az_v5_novaluemcts/iter1/best.pt  eval_out/multiseed/az_v5_iter1_run2
run_eval checkpoints/csas_world/az_v5_novaluemcts/iter1/best.pt  eval_out/multiseed/az_v5_iter1_run3
run_eval checkpoints/csas_world/exp_019_consolidate/last.pt      eval_out/multiseed/exp019_run2
run_eval checkpoints/csas_world/exp_019_consolidate/last.pt      eval_out/multiseed/exp019_run3

echo "MULTISEED_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
