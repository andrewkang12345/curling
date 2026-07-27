#!/usr/bin/env bash
# EXP-018 consolidation: DDP joint-train on the union of per-horizon buffers (no collection, no GPU sim).
# Clean LD_LIBRARY_PATH for torch DDP + cuDNN; GNN env; no setup_gpu.sh (no JAX sim needed).
set -uo pipefail
cd /mnt/data/curling2/csas_world
unset LD_LIBRARY_PATH
export PYTHONPATH=/mnt/data/curling2/csas_world/src:/mnt/data/curling2/csas_v3/src
export JAX_PLATFORMS=cpu
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing
export GNN_NODE_FEATURE_MODE=none
export GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary
export GNN_EDGE_PRUNE_MODE=none
CONFIG="${1:?config}"; UNION="${2:?union dir}"; INIT="${3:?init ckpt}"; OUT="${4:?out dir}"; EPOCHS="${5:-}"
mkdir -p "$OUT"
EP_ARG=""; [ -n "$EPOCHS" ] && EP_ARG="--epochs $EPOCHS"
python3 scripts/run_consolidate.py --config "$CONFIG" --union "$UNION" --init "$INIT" --out "$OUT" $EP_ARG
echo "CONSOLIDATE_EXIT=$?"
