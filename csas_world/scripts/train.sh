#!/bin/bash
# Train the joint csas_world model (4-GPU DDP) on the anchor replay buffers.
# Usage: scripts/train.sh [ablation] [extra args...]
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src
ABLATION="${1:-full}"; shift || true
python3 scripts/train_world.py \
  --config configs/base.yaml \
  --ablation "$ABLATION" \
  --mcts-dir artifacts/replay/mcts/anchor \
  --sim-dir artifacts/replay/sim \
  --out "checkpoints/csas_world/anchor_${ABLATION}" \
  --run-name "anchor_${ABLATION}" \
  --gpus 0,1,2,3 "$@"
