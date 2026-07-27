#!/bin/bash
cd /mnt/data/curling2/csas_world
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 scripts/train_world.py --config configs/anchor.yaml --ablation default \
  --mcts-dir artifacts/replay/mcts/anchor --sim-dir artifacts/replay/sim \
  --out checkpoints/csas_world/anchor_v3 --run-name anchor_v3
echo "ANCHOR_TRAIN_DONE"
