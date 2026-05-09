#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/curling2/csas_fixed
ENV_PY=/mnt/data/curling2/testBrax/brax/brEnv/bin/python
LIB=/mnt/data/curling2/testBrax/brax/brEnv/lib/python3.12/site-packages/nvidia

export JAX_WHEEL_CUDNN_DIR="$LIB/cudnn/lib"
export LD_LIBRARY_PATH="$LIB/nvjitlink/lib:$LIB/cuda_runtime/lib:$LIB/cuda_nvrtc/lib:$LIB/cublas/lib:$LIB/cusparse/lib:$LIB/cusolver/lib:$LIB/cufft/lib:$LIB/cudnn/lib:${LD_LIBRARY_PATH:-}"
export CSAS_PRELOAD_NVIDIA_LIBS=1

FAST_DIR="$ROOT/rescue/current_fast"
FAST_PREFIX="$FAST_DIR/stones_with_estimates"
FAST_CSV="$FAST_DIR/stones_with_estimates.chunk0000.csv"
OUT_DIR="$ROOT/inverse_current"
FINAL_CSV="$OUT_DIR/stones_with_estimates.chunk0000.csv"
WARM_GLOB="${WARM_GLOB:-$ROOT/old/inverse_previous/stones_with_estimates.chunk*.csv}"

mkdir -p "$FAST_DIR" "$OUT_DIR"
rm -rf "$FAST_DIR/stones_with_estimates.parts" "$OUT_DIR/stones_with_estimates.parts"
rm -f "$FAST_DIR"/stones_with_estimates.chunk*.csv "$FINAL_CSV"

# Stage 1: faster all-rows solve.
CUDA_VISIBLE_DEVICES=0,1,2,3 "$ENV_PY" "$ROOT/inverse/make_BC_data.py" \
  --csv "$ROOT/2026/Stones.csv" \
  --out-prefix "$FAST_PREFIX" \
  --chunk-size 25000 \
  --flush-every 500 \
  --solver-method portfolio \
  --warm-start-glob "$WARM_GLOB" \
  --loss-variant slot_identity \
  --sim-c-damp 165 \
  --sim-c-damp-sep-frac 1.0 \
  --sim-c-tangent 20 \
  --sim-mu-tangent 0.05 \
  --sim-spin-contact 0.08 \
  --sim-k-curl 0.12 \
  --sim-a-linear 0.10 \
  --sim-gamma-spin 0.12

# Stage 2: stronger rescue only on rows still above threshold.
CUDA_VISIBLE_DEVICES=0,1,2,3 "$ENV_PY" "$ROOT/rescue/grid_rescue.py" \
  --input-csv "$FAST_CSV" \
  --out-csv "$FINAL_CSV" \
  --threshold 0.1 \
  --loss-variant slot_identity
