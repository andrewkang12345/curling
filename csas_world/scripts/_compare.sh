#!/bin/bash
cd /mnt/data/curling2/csas_world
export PYTHONPATH=src JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -m world.eval.baselines \
  --world checkpoints/csas_world/anchor_v3/model.pt --config configs/anchor.yaml \
  --horizons 2,6,10 --h2h-roots 40 --n-candidates 24 \
  --baselines checkpoints/policy/human_prior_fullcov/model.pt,checkpoints/policy/mcts_horizon/h10/model.pt \
  --device cuda:0 --out artifacts/metrics/anchor_v3_compare.json
echo "COMPARE_DONE"
