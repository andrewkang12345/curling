#!/bin/bash
# Launch the AlphaZero-style iterative-to-convergence run. The driver itself runs
# under the GPU-JAX env (its head-to-head uses the GPU simulator); collection
# subprocesses re-source it, and the training subprocess clears LD_LIBRARY_PATH so
# torch uses its own cuDNN.
set -uo pipefail
cd /mnt/data/curling2/csas_world
source scripts/setup_gpu.sh
python3 scripts/az_converge.py --max-iters 3 --roots 80 \
  --horizons 1,2,3,4,5,6,7,8,9,10 --h2h-horizons 4,8 --h2h-roots 30 --band 0.04 \
  --work checkpoints/csas_world/az
echo "AZ_RUN_DONE"
