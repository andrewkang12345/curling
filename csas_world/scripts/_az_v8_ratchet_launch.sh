#!/usr/bin/env bash
# az_v8 — the structurally-correct AZ ratchet loop (see scripts/az_ratchet.py).
#   * accumulating replay buffer across iterations (fixes the discard-data defect)
#   * 3-draw eval gate: promote only on dwr > 1x combined SE (fixes backward drift)
#   * 2-ply collect with concentrated sims (exp_031: mcts_sims=120, k_widen=1.5 —
#     fixes tree starvation at 60 sims)
#   * train: exp_021 (VFM=true) — value head keeps learning from search-grounded returns
# ~15h/iter (12h collect + 35m train + 2.6h 3-draw eval); 3 iters + baseline ≈ 2 days.
set -uo pipefail
cd /mnt/data/curling2/csas_world

WORK=checkpoints/csas_world/az_v8_ratchet
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

echo "[launch] AZ v8 ratchet starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
echo "[launch] init-world  : checkpoints/csas_world/exp_021_valuemcts_earlystop/best.pt" | tee -a "$LOG"
echo "[launch] collect cfg : configs/exp_031_2ply_sims120.yaml (2-ply, sims=120, k_widen=1.5)" | tee -a "$LOG"
echo "[launch] train cfg   : configs/exp_021_valuemcts_earlystop.yaml (VFM=true)" | tee -a "$LOG"
echo "[launch] iters=3 draws=3 gate=1.0x combined SE" | tee -a "$LOG"

python3 scripts/az_ratchet.py \
  --init-world checkpoints/csas_world/exp_021_valuemcts_earlystop/best.pt \
  --collect-config configs/exp_031_2ply_sims120.yaml \
  --train-config configs/exp_021_valuemcts_earlystop.yaml \
  --iters 3 \
  --draws 3 \
  --max-roots 160 \
  --horizons 1,2,3,4,5,6,7,8,9,10 \
  --eval-n 400 \
  --gate-k 1.0 \
  --work "$WORK" \
  >> "$LOG" 2>&1

echo "AZ_V8_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
