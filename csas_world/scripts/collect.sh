#!/bin/bash
# Collect anchor replay shards (sim-transition + base MCTS) with the canonical
# fixed prior policy + frozen value model.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
PRIOR=/mnt/data/curling2/csas_v3/checkpoints/policy/human_prior_fullcov/best.pt
VALUE=/mnt/data/curling2/csas_v3/checkpoints/value/holdout0/model.pt
ROOTS="${1:-200}"
python3 scripts/collect_global.py --kind sim --policy "$PRIOR" --value "$VALUE" \
  --out artifacts/replay/sim --horizons 3,5,7,9 --roots-per-horizon "$ROOTS" --gpus 0,1,2,3
python3 scripts/collect_global.py --kind mcts --policy "$PRIOR" --value "$VALUE" \
  --out artifacts/replay/mcts/anchor --horizons 1,2,3,4,5,6,7,8,9,10 \
  --roots-per-horizon "$ROOTS" --gpus 0,1,2,3
