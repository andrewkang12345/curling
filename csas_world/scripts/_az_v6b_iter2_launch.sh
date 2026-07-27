#!/usr/bin/env bash
# az_v6b — CONTINUATION of az_v6 for one more iter. Warm-start = az_v6's iter-1 best.pt
# (the winning ckpt from the 2-ply + value_from_mcts=true run). Same collect config
# (2-ply, mcts_max_depth=2, value_leaf_bootstrap=true), same train config (value_from_mcts=true).
#
# In this new work dir, "iter-0" is a fresh re-eval of az_v6/iter1/best.pt, and "iter-1" is
# what az_v6 would have called "iter-2" (the loop declared convergence at iter-1 due to noise
# band, so this experiment overrides that and does one more compound-or-plateau iteration).
#
# Interpretation:
#   * If iter-1 wr AND dScore improve on baseline again -> AZ improvement is COMPOUNDING;
#     evidence to keep iterating.
#   * If iter-1 flatlines or regresses -> the 2-ply / VFM=true recipe has a fixed ceiling
#     at this data scale; time to focus elsewhere (more data, different operator, etc.).
set -uo pipefail
cd /mnt/data/curling2/csas_world

WORK=checkpoints/csas_world/az_v6b_iter2
mkdir -p "$WORK"
LOG="$WORK/run.log"
LOCK="$WORK/launcher.pid"

if [ -f "$LOCK" ]; then
  prev_pid=$(cat "$LOCK")
  if [ -n "$prev_pid" ] && kill -0 "$prev_pid" 2>/dev/null; then
    echo "[launch] REFUSING: another orchestrator alive (pid $prev_pid). Kill it first or rm $LOCK." | tee -a "$LOG"
    exit 1
  fi
  echo "[launch] stale lock from pid $prev_pid (not running) — clearing" | tee -a "$LOG"
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

source scripts/setup_gpu.sh
export PYTHONUNBUFFERED=1

echo "[launch] AZ v6b (continuation, iter-2) starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
echo "[launch] init-world  : checkpoints/csas_world/az_v6_2ply_unfrozen/iter1/best.pt  (az_v6's iter-1 model)" | tee -a "$LOG"
echo "[launch] collect cfg : configs/exp_026_2ply_valuemcts.yaml (same as az_v6)" | tee -a "$LOG"
echo "[launch] train cfg   : configs/exp_021_valuemcts_earlystop.yaml (same as az_v6, value_from_mcts=true)" | tee -a "$LOG"
echo "[launch] work dir    : $WORK" | tee -a "$LOG"

python3 scripts/az_converge_v2.py \
  --init-world checkpoints/csas_world/az_v6_2ply_unfrozen/iter1/best.pt \
  --collect-config configs/exp_026_2ply_valuemcts.yaml \
  --train-config configs/exp_021_valuemcts_earlystop.yaml \
  --max-iters 1 \
  --max-roots 160 \
  --horizons 1,2,3,4,5,6,7,8,9,10 \
  --eval-n 400 \
  --wr-band 0.03 \
  --ds-band 0.10 \
  --work "$WORK" \
  >> "$LOG" 2>&1

echo "AZ_V6B_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
