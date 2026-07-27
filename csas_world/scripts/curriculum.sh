#!/bin/bash
# Run the horizon-staged MCTS curriculum from an anchor checkpoint, comparing each
# stage head-to-head (both throwing orders) to decide convergence.
# Usage: scripts/curriculum.sh <anchor_ckpt> [start] [max] [rounds] [roots]
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
ANCHOR="${1:?usage: curriculum.sh <anchor_ckpt> [start] [max] [rounds] [roots]}"
python3 scripts/run_curriculum.py --config configs/base.yaml \
  --base "$ANCHOR" --work checkpoints/csas_world/curriculum \
  --sim-dir artifacts/replay/sim \
  --start "${2:-1}" --max "${3:-10}" --rounds "${4:-2}" --roots "${5:-200}" \
  --gpus 0,1,2,3
