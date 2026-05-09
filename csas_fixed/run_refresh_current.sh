#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
INVERSE_GLOB="$ROOT/inverse_current/stones_with_estimates.chunk*.csv"

echo "[1/3] inverse full rerun"
bash "$ROOT/run_inverse_current_4gpu.sh"

echo "[2/3] holdout scoring + reports"
SKIP_TRAIN=1 \
INVERSE_GLOB="$INVERSE_GLOB" \
NOISE_HARD_LOSS_MAX=0.25 \
SCORE_HARD_LOSS_MAX=0.5 \
bash "$ROOT/run_holdouts_4gpu.sh"

echo "[3/3] xscore examples"
INVERSE_GLOB="$INVERSE_GLOB" \
bash "$ROOT/run_xscore_examples.sh"

echo "[done] refresh complete"
