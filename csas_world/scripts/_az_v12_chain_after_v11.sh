#!/usr/bin/env bash
# Chain: wait for az_v11; launch az_v12 (screen-then-tree) ONLY IF az_v11 produced no
# promotion. If az_v11 promoted, stop and defer to the operator.
set -uo pipefail
cd /mnt/data/curling2/csas_world

V11_LOG=checkpoints/csas_world/az_v11_tree2term/run.log
V11_LOCK=checkpoints/csas_world/az_v11_tree2term/launcher.pid
WORK=checkpoints/csas_world/az_v12_screentree
mkdir -p "$WORK"
LOG="$WORK/run.log"

echo "[chain] waiting for az_v11..." | tee -a "$LOG"
while true; do
  if grep -q "AZ_V11_DONE" "$V11_LOG" 2>/dev/null; then
    pid=$(cat "$V11_LOCK" 2>/dev/null || echo "")
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then break; fi
  fi
  sleep 60
done
sleep 15

PROMOTED=$(python3 -c "
import json
h = json.load(open('checkpoints/csas_world/az_v11_tree2term/history.json'))
print(1 if any(e.get('promoted') for e in h if e['iter'] > 0) else 0)")
if [ "$PROMOTED" = "1" ]; then
  echo "[chain] az_v11 PROMOTED — NOT launching az_v12 (operator decision needed)" | tee -a "$LOG"
  exit 0
fi
echo "[chain] az_v11 finished without promotion -> launching az_v12" | tee -a "$LOG"

LOCK="$WORK/launcher.pid"
if [ -f "$LOCK" ]; then
  prev=$(cat "$LOCK")
  if [ -n "$prev" ] && kill -0 "$prev" 2>/dev/null; then
    echo "[chain] REFUSING: pid $prev alive" | tee -a "$LOG"; exit 1
  fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

# iter-0 baseline: champion confirmation draws 4,5,7 (unbiased; disjoint mix from v10's 4-6 / v11's 5-7)
for i in 1 2 3; do mkdir -p eval_out/az_v12_screentree/iter0_run$i; done
cp eval_out/az_v9_selfplay/iter2_run4/prior__*.json eval_out/az_v9_selfplay/iter2_run4/summary.json eval_out/az_v12_screentree/iter0_run1/ 2>/dev/null || true
cp eval_out/az_v9_selfplay/iter2_run5/prior__*.json eval_out/az_v9_selfplay/iter2_run5/summary.json eval_out/az_v12_screentree/iter0_run2/ 2>/dev/null || true
cp eval_out/az_v9_selfplay/iter2_run7/prior__*.json eval_out/az_v9_selfplay/iter2_run7/summary.json eval_out/az_v12_screentree/iter0_run3/ 2>/dev/null || true

source scripts/setup_gpu.sh
export PYTHONUNBUFFERED=1
echo "[launch] AZ v12 (screen-then-tree) at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"

python3 scripts/az_ratchet.py \
  --init-world checkpoints/csas_world/az_v9_selfplay/iter2/best.pt \
  --collect-config configs/exp_035_screen_tree.yaml \
  --train-config configs/exp_021_valuemcts_earlystop.yaml \
  --selfplay-games 64 \
  --selfplay-scorer screen_tree \
  --gate-metric ds \
  --iters 2 \
  --draws 3 \
  --eval-n 400 \
  --gate-k 1.0 \
  --work "$WORK" \
  >> "$LOG" 2>&1

echo "AZ_V12_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
