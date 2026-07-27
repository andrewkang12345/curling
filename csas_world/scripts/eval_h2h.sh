#!/bin/bash
# Compare a trained csas_world checkpoint's value/policy heads + game strength to
# the previous best versions (human prior, mcts_horizon, gaussian value).
# Usage: scripts/eval_h2h.sh <world_ckpt> [horizons] [h2h_roots]
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
WORLD="${1:?usage: eval_h2h.sh <world_ckpt> [horizons] [roots]}"
HZ="${2:-2,5,8,10}"
ROOTS="${3:-150}"
python3 -m world.eval.baselines --world "$WORLD" --config configs/base.yaml \
  --horizons "$HZ" --h2h-roots "$ROOTS" --device cuda:0 \
  --out artifacts/metrics/baseline_compare.json
