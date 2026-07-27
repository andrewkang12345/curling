#!/usr/bin/env bash
# Background launcher for the v2 AZ-style convergence loop:
#   - iter-0 baseline = exp_021_valuemcts_earlystop/best.pt (re-evaluated deterministically)
#   - per iter: collect 1,200 train + 400 val records (4 shards × 40 rec × 10 horizons)
#              with KR-UCT n_sims=120 (anchor_mcts_v3.yaml); train via run_consolidate.py
#              with the exp_021 recipe (value_from_mcts=true, held-out val + early stop);
#              eval vs human prior (N=400 × 10 horizons × both orders); compare to prev iter.
#   - converged when MEAN-of-pairs winrate AND dScore both fail to improve over the
#     previous iter by more than the noise floor (Δwr ≤ 0.03 AND ΔdS ≤ 0.10).
#
# Cost: ~7h collect + ~30m train + ~1h eval ≈ 8.5h per iter on 4 GPUs. Capped at 5 iters
# (≈ 2 days wall-clock worst case; exits earlier on convergence).
set -uo pipefail
cd /mnt/data/curling2/csas_world

WORK=checkpoints/csas_world/az_v4
mkdir -p "$WORK"
LOG="$WORK/run.log"
LOCK="$WORK/launcher.pid"

# Single-instance guard. Prior runs raced (multiple orchestrators wrote to the same iter1/ dir,
# clobbered each other's best.pt, and interleaved eval shards). Refuse to start if another
# orchestrator is alive — operator must kill it first.
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

# az_converge_v2.py composes the env per subprocess (GPU/CPU JAX where appropriate, GNN flags,
# torch/jax separation). We only need a bare-minimum outer env here.
source scripts/setup_gpu.sh
export PYTHONUNBUFFERED=1

echo "[launch] AZ v2 starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
echo "[launch] init-world: checkpoints/csas_world/exp_021_valuemcts_earlystop/best.pt" | tee -a "$LOG"
echo "[launch] work dir : $WORK" | tee -a "$LOG"
echo "[launch] log      : $LOG" | tee -a "$LOG"

python3 scripts/az_converge_v2.py \
  --init-world checkpoints/csas_world/exp_021_valuemcts_earlystop/best.pt \
  --max-iters 5 \
  --max-roots 160 \
  --horizons 1,2,3,4,5,6,7,8,9,10 \
  --eval-n 400 \
  --wr-band 0.03 \
  --ds-band 0.10 \
  --work "$WORK" \
  >> "$LOG" 2>&1

echo "AZ_V4_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
