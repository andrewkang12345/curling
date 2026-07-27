#!/usr/bin/env bash
# az_v15 L4 leg: the CORRECTED loop (incumbent-relative gate + H2 fixes) run to the
# gate-convergence signal at 7.3M, from the strongest L4 model (az_v13-it1).
# Waits for the metagame confirmation draws to free the GPUs first.
set -uo pipefail
cd /mnt/data/curling2/csas_world
while ! grep -q "CONFIRM_DONE" eval_out/metagame/confirm.log 2>/dev/null; do sleep 120; done
sleep 15

WORK=checkpoints/csas_world/az_v15_L4
mkdir -p "$WORK"; LOG="$WORK/run.log"; LOCK="$WORK/launcher.pid"
if [ -f "$LOCK" ]; then p=$(cat "$LOCK"); [ -n "$p" ] && kill -0 "$p" 2>/dev/null && { echo "[launch] REFUSING: $p alive" | tee -a "$LOG"; exit 1; }; fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
source scripts/setup_gpu.sh
export PYTHONUNBUFFERED=1
echo "[launch] az_v15 L4 leg at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"

python3 scripts/az_ratchet.py \
  --init-world checkpoints/csas_world/az_v13_ratchet/iter1/best.pt \
  --collect-config configs/exp_037_sig_screen_tree.yaml \
  --train-config configs/exp_043_corrected_loop.yaml \
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
echo "AZ_V15_L4_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
