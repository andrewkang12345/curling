#!/usr/bin/env bash
# az_v15 L8 leg: corrected loop from the global champion az_v14d (13.65M).
set -uo pipefail
cd /mnt/data/curling2/csas_world
WORK=checkpoints/csas_world/az_v15_L8
mkdir -p "$WORK"; LOG="$WORK/run.log"; LOCK="$WORK/launcher.pid"
if [ -f "$LOCK" ]; then p=$(cat "$LOCK"); [ -n "$p" ] && kill -0 "$p" 2>/dev/null && { echo REFUSING | tee -a "$LOG"; exit 1; }; fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
source scripts/setup_gpu.sh
export PYTHONUNBUFFERED=1
echo "[launch] az_v15 L8 leg at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
python3 scripts/az_ratchet.py \
  --init-world checkpoints/csas_world/az_v14d/best.pt \
  --collect-config configs/exp_037_sig_screen_tree.yaml \
  --train-config configs/exp_045_L8_loop.yaml \
  --selfplay-games 64 \
  --selfplay-scorer screen_tree \
  --gate-metric ds \
  --gate-opponent incumbent \
  --stop-after-nonpromotions 2 \
  --iters 8 \
  --draws 3 \
  --eval-n 400 \
  --gate-k 1.0 \
  --work "$WORK" \
  >> "$LOG" 2>&1
echo "AZ_V15_L8_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
