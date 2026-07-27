#!/usr/bin/env bash
# Wait for the current az_v9 run (iters 1-3) to finish, then continue the SAME ratchet
# in the SAME work dir for 3 more iterations (4-6) via --resume. Accumulating buffer
# persists (it1..it3 symlinks stay; it4..it6 get added); incumbent + stats restored
# from history.json (currently iter2's promoted model unless iter3 promotes too).
set -uo pipefail
cd /mnt/data/curling2/csas_world

WORK=checkpoints/csas_world/az_v9_selfplay
LOG="$WORK/run.log"
LOCK="$WORK/launcher.pid"

echo "[chain] waiting for az_v9 (iters 1-3) to finish..." | tee -a "$LOG.chain"
while true; do
  if grep -q "AZ_V9_DONE" "$LOG" 2>/dev/null; then
    pid=$(cat "$LOCK" 2>/dev/null || echo "")
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
      echo "[chain] az_v9 done + lock released -> continuing" | tee -a "$LOG.chain"
      break
    fi
  fi
  sleep 60
done
sleep 15

# take over the lock for the continuation
if [ -f "$LOCK" ]; then
  prev=$(cat "$LOCK")
  if [ -n "$prev" ] && kill -0 "$prev" 2>/dev/null; then
    echo "[chain] REFUSING: pid $prev still alive" | tee -a "$LOG.chain"; exit 1
  fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

source scripts/setup_gpu.sh
export PYTHONUNBUFFERED=1
echo "[chain] AZ v9 CONTINUATION (iters 4-6) starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"

python3 scripts/az_ratchet.py \
  --resume \
  --collect-config configs/exp_031_2ply_sims120.yaml \
  --train-config configs/exp_021_valuemcts_earlystop.yaml \
  --selfplay-games 160 \
  --gate-metric ds \
  --iters 3 \
  --draws 3 \
  --eval-n 400 \
  --gate-k 1.0 \
  --work "$WORK" \
  >> "$LOG" 2>&1

echo "AZ_V9_CONT_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
