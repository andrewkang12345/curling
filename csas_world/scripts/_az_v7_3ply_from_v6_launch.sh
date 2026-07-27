#!/usr/bin/env bash
# az_v7 — 3-ply collect + value_from_mcts=true train, warm-started from az_v6's iter-1
# best.pt (the winning 2-ply ckpt: wr 0.543, dScore +0.225). This tests whether a
# DEEPER improvement operator (3-ply, one more ply of real lookahead than 2-ply)
# unlocks the compounding that 2-ply plateaued at (az_v6b: Δwr=-0.011, ΔdS=-0.079).
#
# In az_v7's naming, "iter-0" is a fresh re-eval of az_v6/iter1/best.pt (same baseline
# ckpt as az_v6b), and "iter-1" is the new 3-ply-collected+VFM-trained ckpt.
set -uo pipefail
cd /mnt/data/curling2/csas_world

WORK=checkpoints/csas_world/az_v7_3ply_from_v6
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

echo "[launch] AZ v7 (3-ply from az_v6/iter1) starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
echo "[launch] init-world  : checkpoints/csas_world/az_v6_2ply_unfrozen/iter1/best.pt" | tee -a "$LOG"
echo "[launch] collect cfg : configs/exp_029_3ply_valuemcts.yaml (mcts_max_depth=3)" | tee -a "$LOG"
echo "[launch] train cfg   : configs/exp_021_valuemcts_earlystop.yaml (value_from_mcts=true)" | tee -a "$LOG"
echo "[launch] work dir    : $WORK" | tee -a "$LOG"

python3 scripts/az_converge_v2.py \
  --init-world checkpoints/csas_world/az_v6_2ply_unfrozen/iter1/best.pt \
  --collect-config configs/exp_029_3ply_valuemcts.yaml \
  --train-config configs/exp_021_valuemcts_earlystop.yaml \
  --max-iters 1 \
  --max-roots 160 \
  --horizons 1,2,3,4,5,6,7,8,9,10 \
  --eval-n 400 \
  --wr-band 0.03 \
  --ds-band 0.10 \
  --work "$WORK" \
  >> "$LOG" 2>&1

echo "AZ_V7_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
