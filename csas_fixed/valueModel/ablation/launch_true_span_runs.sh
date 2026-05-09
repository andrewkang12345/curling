#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
PROJECT="curling-value-gnn-unoccluded-span"
OUT_ROOT="${ROOT_DIR}/gnn_results_true_span_${RUN_TS}"
LOG_ROOT="/tmp/gnn_true_span_${RUN_TS}"

mkdir -p "${OUT_ROOT}"
cd "${ROOT_DIR}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] launching true-span sweep"
python3 -u run_gnn_ablation.py \
  --gpu 0 \
  --only_configs gnn_egnn_small \
  --out_dir "${OUT_ROOT}/egnn_small" \
  --wandb_project "${PROJECT}" \
  > "${LOG_ROOT}_gpu0.txt" 2>&1 &
PID0=$!

python3 -u run_gnn_ablation.py \
  --gpu 1 \
  --only_configs gnn_egnn_medium \
  --out_dir "${OUT_ROOT}/egnn_medium" \
  --wandb_project "${PROJECT}" \
  > "${LOG_ROOT}_gpu1.txt" 2>&1 &
PID1=$!

python3 -u run_gnn_ablation.py \
  --gpu 2 \
  --only_configs gnn_transformer_small \
  --out_dir "${OUT_ROOT}/graph_transformer_small" \
  --wandb_project "${PROJECT}" \
  > "${LOG_ROOT}_gpu2.txt" 2>&1 &
PID2=$!

python3 -u run_gnn_ablation.py \
  --gpu 3 \
  --only_configs gnn_transformer_medium \
  --out_dir "${OUT_ROOT}/graph_transformer_medium" \
  --wandb_project "${PROJECT}" \
  > "${LOG_ROOT}_gpu3.txt" 2>&1 &
PID3=$!

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] launched PIDs: ${PID0} ${PID1} ${PID2} ${PID3}"
echo "logs: ${LOG_ROOT}_gpu{0,1,2,3}.txt"
echo "out_dir: ${OUT_ROOT}"

wait "${PID0}" "${PID1}" "${PID2}" "${PID3}"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] true-span sweep finished"
