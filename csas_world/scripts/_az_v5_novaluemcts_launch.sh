#!/usr/bin/env bash
# az_v5 — exactly like az_v4_iter1 but with the EXP-022 (value_from_mcts=false) train recipe
# instead of EXP-021 (value_from_mcts=true), to isolate the question:
#   "is iter-1's regression caused by value-head drift (from value_from_mcts=true),
#    or by the degenerate self-distillation from a too-weak improvement operator?"
#
# Same warm-start (exp_021/best.pt), same collector (exp_021 policy), SAME collected data
# (symlinked from az_v4_iter1 — confirmed identical recipe at collect time), only the train
# config differs (configs/exp_022_exp019_earlystop.yaml: value_from_mcts=false, early-stop
# by val_total_mcts).
set -uo pipefail
cd /mnt/data/curling2/csas_world

WORK=checkpoints/csas_world/az_v5_novaluemcts
mkdir -p "$WORK"
LOG="$WORK/run.log"
LOCK="$WORK/launcher.pid"

# Single-instance guard (same as v4 — see _az_v4_launch.sh for context).
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

echo "[launch] AZ v5 (no-value-from-mcts) starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
echo "[launch] init-world : checkpoints/csas_world/exp_021_valuemcts_earlystop/best.pt" | tee -a "$LOG"
echo "[launch] train cfg  : configs/exp_022_exp019_earlystop.yaml  (value_from_mcts=false)" | tee -a "$LOG"
echo "[launch] work dir   : $WORK" | tee -a "$LOG"
echo "[launch] log        : $LOG" | tee -a "$LOG"

python3 scripts/az_converge_v2.py \
  --init-world checkpoints/csas_world/exp_021_valuemcts_earlystop/best.pt \
  --train-config configs/exp_022_exp019_earlystop.yaml \
  --max-iters 5 \
  --max-roots 160 \
  --horizons 1,2,3,4,5,6,7,8,9,10 \
  --eval-n 400 \
  --wr-band 0.03 \
  --ds-band 0.10 \
  --work "$WORK" \
  >> "$LOG" 2>&1

echo "AZ_V5_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
