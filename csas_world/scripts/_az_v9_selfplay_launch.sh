#!/usr/bin/env bash
# az_v9 — FULL-GAME SELF-PLAY ratchet loop. Same harness as az_v8 plus the three
# changes that make it the first structurally-complete AZ loop in this project:
#   1. Collection = full ends played from the pre-placed openings by the incumbent
#      policy, with a fresh 2-ply KR-UCT search at EVERY ply (world/search/selfplay.py).
#      Policy-improvement signal lands on policy-induced states that EVOLVE each
#      iteration — not on the frozen human root pool (the az_v4..v8 limitation).
#   2. Tree leaves are evaluated by the INCUMBENT's value head (--value-world), so
#      value-head improvements feed back into search. (az_v6..v8 used the frozen
#      csas_v3 value model at leaves — the search operator never changed across iters.)
#   3. Promotion gate is dScore-PRIMARY (winrate as a no-clear-loss guard), per the
#      project's metric convention.
# Volume matched to az_v8: 160 games/shard x 4 shards x 10 records/game = 6,400/iter
# (4,800 train + 1,600 val), accumulating across iters. ~15h/iter; 3 iters ≈ 2 days.
set -uo pipefail
cd /mnt/data/curling2/csas_world

WORK=checkpoints/csas_world/az_v9_selfplay
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

echo "[launch] AZ v9 self-play ratchet starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
echo "[launch] init-world  : checkpoints/csas_world/exp_021_valuemcts_earlystop/best.pt" | tee -a "$LOG"
echo "[launch] collect     : SELF-PLAY, 160 games/shard, cfg=exp_031 (2-ply, sims=120, k_widen=1.5), leaves=incumbent value head" | tee -a "$LOG"
echo "[launch] train cfg   : configs/exp_021_valuemcts_earlystop.yaml (VFM=true; MC value targets from realized games)" | tee -a "$LOG"
echo "[launch] gate        : dScore-primary, 1.0x combined SE (wr no-clear-loss guard)" | tee -a "$LOG"

python3 scripts/az_ratchet.py \
  --init-world checkpoints/csas_world/exp_021_valuemcts_earlystop/best.pt \
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

echo "AZ_V9_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
