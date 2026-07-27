#!/bin/bash
cd /mnt/data/curling2/csas_world
export PYTHONPATH=src JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 scripts/run_curriculum.py --config configs/anchor.yaml \
  --base checkpoints/csas_world/anchor_v3/model.pt \
  --work checkpoints/csas_world/curriculum \
  --sim-dir artifacts/replay/sim \
  --start 1 --max 3 --rounds 1 --roots 150 --epochs 10 --gpus 0,1,2,3
echo "CURRICULUM_DONE"
