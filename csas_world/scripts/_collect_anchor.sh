#!/bin/bash
set -e
cd /mnt/data/curling2/csas_world
export PYTHONPATH=src JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "[$(date +%H:%M:%S)] START sim collection"
python3 scripts/collect_global.py --kind sim --policy /mnt/data/curling2/csas_v3/checkpoints/policy/human_prior_fullcov/best.pt --value /mnt/data/curling2/csas_v3/checkpoints/value/holdout0/model.pt   --out artifacts/replay/sim --horizons 3,5,7,9 --roots-per-horizon 250 --gpus 0,1,2,3
echo "[$(date +%H:%M:%S)] START mcts anchor collection"
python3 scripts/collect_global.py --kind mcts --policy /mnt/data/curling2/csas_v3/checkpoints/policy/human_prior_fullcov/best.pt --value /mnt/data/curling2/csas_v3/checkpoints/value/holdout0/model.pt   --out artifacts/replay/mcts/anchor --horizons 1,2,3,4,5,6,7,8,9,10 --roots-per-horizon 200 --gpus 0,1,2,3
echo "[$(date +%H:%M:%S)] COLLECTION DONE"
