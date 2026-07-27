#!/bin/bash
# Option 1 on the clean-value base: KR-UCT with n_sims=120 (stronger search) +
# value_from_mcts=false (value head on the clean realized-ValueDiff buffer), full
# roots at every horizon. Separate work dir + MCTS replay namespace (az_v3_*) so it
# does not reuse the n_sims=48 EXP-002 collection.
set -uo pipefail
cd /mnt/data/curling2/csas_world
source scripts/setup_gpu.sh
python3 scripts/az_converge.py --max-iters 3 --roots 80 \
  --horizons 1,2,3,4,5,6,7,8,9,10 --h2h-horizons 4,8 --h2h-roots 30 --band 0.04 \
  --work checkpoints/csas_world/az_v3
echo "AZ_V3_RUN_DONE"
