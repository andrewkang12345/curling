#!/usr/bin/env bash
# az_v13 Step 1 — the fully-fixed improvement loop, one ratchet run:
#   * collect: screen_tree + COLLECT-TIME significance gating (exp_037: dist_sig_t=2,
#     tie-break rollouts in the ambiguous band) — only statistically-real preferences
#     get distillation targets; every ply still contributes grounded value targets.
#   * train: exp_036 (aligned selection metric on significance-masked val, val-driven
#     early stopping, value-drift guard).
#   * accum pre-seeded with the Step-0 confident buffers (30.7k records) so iter-1
#     trains on Step-0's baseline + the fresh significant targets.
#   * gate: dScore-primary vs the champion, as always.
set -uo pipefail
cd /mnt/data/curling2/csas_world

WORK=checkpoints/csas_world/az_v13_ratchet
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

# pre-seed the accumulating buffers with the Step-0 confident buffers (symlinks)
AT=artifacts/replay/az_v13_ratchet_accum_train
AV=artifacts/replay/az_v13_ratchet_accum_val
mkdir -p "$AT" "$AV"
for f in artifacts/replay/az_v13_anchor_train/*.npz; do
  dst="$AT/step0_$(basename "$f")"; [ -e "$dst" ] || ln -s "$(readlink -f "$f")" "$dst"
done
for f in artifacts/replay/az_v13_anchor_val/*.npz; do
  dst="$AV/step0_$(basename "$f")"; [ -e "$dst" ] || ln -s "$(readlink -f "$f")" "$dst"
done
echo "[launch] accum pre-seeded: $(ls $AT | wc -l) train, $(ls $AV | wc -l) val files" | tee -a "$LOG"

# iter-0 baseline: champion confirmation draws 4-6
for i in 1 2 3; do mkdir -p eval_out/az_v13_ratchet/iter0_run$i; done
cp eval_out/az_v9_selfplay/iter2_run4/prior__*.json eval_out/az_v9_selfplay/iter2_run4/summary.json eval_out/az_v13_ratchet/iter0_run1/ 2>/dev/null || true
cp eval_out/az_v9_selfplay/iter2_run5/prior__*.json eval_out/az_v9_selfplay/iter2_run5/summary.json eval_out/az_v13_ratchet/iter0_run2/ 2>/dev/null || true
cp eval_out/az_v9_selfplay/iter2_run6/prior__*.json eval_out/az_v9_selfplay/iter2_run6/summary.json eval_out/az_v13_ratchet/iter0_run3/ 2>/dev/null || true

source scripts/setup_gpu.sh
export PYTHONUNBUFFERED=1
echo "[launch] az_v13 Step 1 (significance-gated loop) at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"

python3 scripts/az_ratchet.py \
  --init-world checkpoints/csas_world/az_v9_selfplay/iter2/best.pt \
  --collect-config configs/exp_037_sig_screen_tree.yaml \
  --train-config configs/exp_036_confident_finetune.yaml \
  --selfplay-games 64 \
  --selfplay-scorer screen_tree \
  --gate-metric ds \
  --iters 2 \
  --draws 3 \
  --eval-n 400 \
  --gate-k 1.0 \
  --work "$WORK" \
  >> "$LOG" 2>&1

echo "AZ_V13_STEP1_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
