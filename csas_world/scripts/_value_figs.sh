#!/bin/bash
cd /mnt/data/curling2/csas_world
source scripts/setup_gpu.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -m world.eval.figures \
  --world checkpoints/csas_world/anchor_v3/model.pt \
  --out-root artifacts/figures \
  --horizons 1 3 5 8 10 --n-real 4 --noise-samples 16 --kind both --device cuda:1
echo "VALUE_FIGS_DONE"
