#!/bin/bash
# AlphaZero-style iteration 1: collect terminal-MC MCTS targets (rollout to terminal,
# noise=8) with the prior policy + frozen value for rollout guidance, then train the
# joint model with policy_bc=0 + value_from_mcts=true (configs/anchor_az.yaml).
set -uo pipefail
cd /mnt/data/curling2/csas_world
PRIOR=/mnt/data/curling2/csas_v3/checkpoints/policy/human_prior_fullcov/best.pt
VALUE=/mnt/data/curling2/csas_v3/checkpoints/value/holdout0/model.pt
OUT=artifacts/replay/mcts/anchor_az
CFG=configs/anchor_az.yaml
ROOTS=150
mkdir -p "$OUT"

echo "[$(date +%H:%M:%S)] === collect terminal-MC targets (rollout->terminal, noise=8) ==="
(
  source scripts/setup_gpu.sh
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  for h in 1 2 3 4 5 6 7 8 9 10; do
    hh=$(printf "h%02d" "$h"); t0=$(date +%s)
    for k in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$k XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
      python3 -m world.search.collect --config "$CFG" --horizon "$h" --max-roots "$ROOTS" \
        --kind mcts --policy "$PRIOR" --value "$VALUE" \
        --out "$OUT/$hh/shard$k.npz" --num-shards 4 --shard-id $k --device cuda:0 --seed $((200+h)) &
    done
    wait
    echo "[$(date +%H:%M:%S)] $hh done in $(( $(date +%s) - t0 ))s ($(find $OUT/$hh -name '*.npz'|wc -l) shards)"
  done
)
echo "[$(date +%H:%M:%S)] === train (policy_bc=0, value_from_mcts=true) ==="
(
  export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python3 scripts/train_world.py --config "$CFG" --ablation default \
    --mcts-dir "$OUT" --sim-dir artifacts/replay/sim \
    --out checkpoints/csas_world/anchor_az --run-name anchor_az
)
echo "[$(date +%H:%M:%S)] RETRAIN_AZ_DONE"
