#!/usr/bin/env bash
# Chain: wait for az_v10 to finish; launch az_v11 ONLY IF az_v10 produced no promotion
# (per operator instruction: "try depth k=2 + terminal rollout once the current run
# lands without promotion"). If az_v10 DID promote, do nothing and leave a note —
# the operator will decide manually.
set -uo pipefail
cd /mnt/data/curling2/csas_world

V10_LOG=checkpoints/csas_world/az_v10_terminal/run.log
V10_LOCK=checkpoints/csas_world/az_v10_terminal/launcher.pid
WORK=checkpoints/csas_world/az_v11_tree2term
mkdir -p "$WORK"
LOG="$WORK/run.log"

echo "[chain] waiting for az_v10..." | tee -a "$LOG"
while true; do
  if grep -q "AZ_V10_DONE" "$V10_LOG" 2>/dev/null; then
    pid=$(cat "$V10_LOCK" 2>/dev/null || echo "")
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then break; fi
  fi
  sleep 60
done
sleep 15

PROMOTED=$(python3 -c "
import json
h = json.load(open('checkpoints/csas_world/az_v10_terminal/history.json'))
print(1 if any(e.get('promoted') for e in h if e['iter'] > 0) else 0)")
if [ "$PROMOTED" = "1" ]; then
  echo "[chain] az_v10 PROMOTED an iteration — NOT launching az_v11 (operator decision needed)" | tee -a "$LOG"
  exit 0
fi
echo "[chain] az_v10 finished without promotion -> launching az_v11" | tee -a "$LOG"

LOCK="$WORK/launcher.pid"
if [ -f "$LOCK" ]; then
  prev=$(cat "$LOCK")
  if [ -n "$prev" ] && kill -0 "$prev" 2>/dev/null; then
    echo "[chain] REFUSING: pid $prev alive" | tee -a "$LOG"; exit 1
  fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

# seed iter-0 baseline from the champion's unbiased confirmation draws (5,6,7 — the set
# NOT used by az_v10's seeding, keeping the two experiments' baselines independent)
for i in 1 2 3; do mkdir -p eval_out/az_v11_tree2term/iter0_run$i; done
cp eval_out/az_v9_selfplay/iter2_run5/prior__*.json eval_out/az_v9_selfplay/iter2_run5/summary.json eval_out/az_v11_tree2term/iter0_run1/ 2>/dev/null || true
cp eval_out/az_v9_selfplay/iter2_run6/prior__*.json eval_out/az_v9_selfplay/iter2_run6/summary.json eval_out/az_v11_tree2term/iter0_run2/ 2>/dev/null || true
cp eval_out/az_v9_selfplay/iter2_run7/prior__*.json eval_out/az_v9_selfplay/iter2_run7/summary.json eval_out/az_v11_tree2term/iter0_run3/ 2>/dev/null || true

source scripts/setup_gpu.sh
export PYTHONUNBUFFERED=1
echo "[launch] AZ v11 (dense-root KR-UCT k=2 + terminal rollouts) at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"

python3 scripts/az_ratchet.py \
  --init-world checkpoints/csas_world/az_v9_selfplay/iter2/best.pt \
  --collect-config configs/exp_034_tree2_terminal.yaml \
  --train-config configs/exp_021_valuemcts_earlystop.yaml \
  --selfplay-games 64 \
  --selfplay-scorer tree_terminal \
  --gate-metric ds \
  --iters 2 \
  --draws 3 \
  --eval-n 400 \
  --gate-k 1.0 \
  --work "$WORK" \
  >> "$LOG" 2>&1

echo "AZ_V11_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
