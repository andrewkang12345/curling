#!/usr/bin/env bash
# Launch the Curling Arena (from csas_world/): bash arena/run.sh [port]
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${1:-8020}"

if [ "${ARENA_JAX:-cpu}" = "gpu" ]; then
  # GPU-JAX sim (vendored tree): ~30x batched throughput -> noise-robust solves
  source scripts/setup_gpu.sh
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${ARENA_XLA_FRACTION:-0.18}"
  export PYTHONPATH="src:${CSAS_V3_SRC:-/mnt/data/curling2/csas_v3/src}:${PYTHONPATH}"
else
  export PYTHONPATH="src:${CSAS_V3_SRC:-/mnt/data/curling2/csas_v3/src}"
  export JAX_PLATFORMS=cpu
  unset LD_LIBRARY_PATH || true
fi
export WORLD_BOUNDARY_REMOVAL=1   # real takeout rules (explicit; also the stack default)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# GraphTF feature modes used by every csas_world training/eval run
export GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing
export GNN_NODE_FEATURE_MODE=none
export GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary
export GNN_EDGE_PRUNE_MODE=none

exec python3 -m uvicorn arena.app:app --host 0.0.0.0 --port "$PORT" --workers 1
