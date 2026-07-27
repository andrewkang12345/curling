#!/usr/bin/env bash
# az_v6 — TWO-PLY collect (configs/exp_026_2ply_valuemcts.yaml) + value_from_mcts=true
# train (configs/exp_021_valuemcts_earlystop.yaml). Same warm-start as az_v4/v5
# (exp_021/best.pt) so the only changes from az_v4 are:
#   (a) the collection operator: 1-ply EZ → 2-ply KR-UCT tree with value-head leaves
#   (b) the same value_from_mcts=true training as az_v4 ("unfrozen value head")
# This isolates the question: does a stronger improvement operator unlock real gains
# that the 1-ply path can't deliver?
set -uo pipefail
cd /mnt/data/curling2/csas_world

WORK=checkpoints/csas_world/az_v6_2ply_unfrozen
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

echo "[launch] AZ v6 (2-ply + unfrozen value head) starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
echo "[launch] init-world  : checkpoints/csas_world/exp_021_valuemcts_earlystop/best.pt" | tee -a "$LOG"
echo "[launch] collect cfg : configs/exp_026_2ply_valuemcts.yaml (use_mcts_tree=true, mcts_max_depth=2, value_leaf_bootstrap=true, mcts_sims=60)" | tee -a "$LOG"
echo "[launch] train cfg   : configs/exp_021_valuemcts_earlystop.yaml (value_from_mcts=true)" | tee -a "$LOG"
echo "[launch] work dir    : $WORK" | tee -a "$LOG"
echo "[launch] log         : $LOG" | tee -a "$LOG"

python3 scripts/az_converge_v2.py \
  --init-world checkpoints/csas_world/exp_021_valuemcts_earlystop/best.pt \
  --collect-config configs/exp_026_2ply_valuemcts.yaml \
  --train-config configs/exp_021_valuemcts_earlystop.yaml \
  --max-iters 5 \
  --max-roots 160 \
  --horizons 1,2,3,4,5,6,7,8,9,10 \
  --eval-n 400 \
  --wr-band 0.03 \
  --ds-band 0.10 \
  --work "$WORK" \
  >> "$LOG" 2>&1

echo "AZ_V6_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
