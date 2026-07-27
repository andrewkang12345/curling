#!/bin/bash
set -uo pipefail
cd /mnt/data/curling2/csas_world
source scripts/setup_gpu.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
python3 -m world.eval.baselines --world checkpoints/csas_world/anchor_noisy/model.pt \
  --config configs/anchor_noisy.yaml --horizons 2,6,10 --h2h-roots 40 --n-candidates 24 \
  --baselines checkpoints/policy/human_prior_fullcov/model.pt,checkpoints/policy/mcts_horizon/h10/model.pt \
  --device cuda:0 --out artifacts/metrics/anchor_noisy_compare.json
echo "BASELINE_CMP_DONE"
