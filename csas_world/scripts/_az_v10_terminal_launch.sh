#!/usr/bin/env bash
# az_v10 — VALUE-FREE backward-induction-style ratchet, warm-started from the az_v9
# champion. Per-ply operator = dense candidates (policy+structured+diverse+local+global)
# scored by noise-averaged MC rollouts to TERMINAL + rule scoring (exp_033). No learned
# value model anywhere in the improvement operator, so:
#   * promotion => the value-free operator found real improvement the 2-ply/V operator
#     could not express (breaks the (π,V) co-adaptation fixed point);
#   * convergence => the champion is un-improvable against a dense proposal at MC
#     resolution — the strongest optimality certificate available in this domain.
# Per-shard .diag.json files record the certification margins (Qbest_all − Qbest_policy)
# per horizon.
set -uo pipefail
cd /mnt/data/curling2/csas_world

WORK=checkpoints/csas_world/az_v10_terminal
mkdir -p "$WORK"
LOG="$WORK/run.log"
LOCK="$WORK/launcher.pid"

if [ -f "$LOCK" ]; then
  prev=$(cat "$LOCK")
  if [ -n "$prev" ] && kill -0 "$prev" 2>/dev/null; then
    echo "[launch] REFUSING: pid $prev alive" | tee -a "$LOG"; exit 1
  fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

source scripts/setup_gpu.sh
export PYTHONUNBUFFERED=1

echo "[launch] AZ v10 (value-free terminal-dense) starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
echo "[launch] init-world : checkpoints/csas_world/az_v9_selfplay/iter2/best.pt (the champion)" | tee -a "$LOG"
echo "[launch] collect    : SELF-PLAY, scorer=terminal (exp_033: ~119 dense cands x 4 noisy execs x MC-to-terminal), 128 games/shard" | tee -a "$LOG"
echo "[launch] train cfg  : configs/exp_021_valuemcts_earlystop.yaml (VFM=true; grounded MC targets)" | tee -a "$LOG"
echo "[launch] gate       : dScore-primary, 1.0x combined SE" | tee -a "$LOG"

python3 scripts/az_ratchet.py \
  --init-world checkpoints/csas_world/az_v9_selfplay/iter2/best.pt \
  --collect-config configs/exp_033_terminal_dense.yaml \
  --train-config configs/exp_021_valuemcts_earlystop.yaml \
  --selfplay-games 128 \
  --selfplay-scorer terminal \
  --gate-metric ds \
  --iters 2 \
  --draws 3 \
  --eval-n 400 \
  --gate-k 1.0 \
  --work "$WORK" \
  >> "$LOG" 2>&1

echo "AZ_V10_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
