#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs mcts_targets checkpoints

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.25}"
source ./setup_runtime_env.sh

HOLDOUT="${HOLDOUT:-0}"
POLICY="${POLICY:-checkpoints/policy_prior_h${HOLDOUT}/model.pt}"
VALUE="${VALUE:-/mnt/data/curling2/csas_fixed/holdouts/${HOLDOUT}/model_settf_gaussian/model.pt}"

CUDA_VISIBLE_DEVICES=0 "$PYTHON" generate_mcts_iteration_targets.py \
  --holdout "$HOLDOUT" \
  --policy "$POLICY" \
  --value "$VALUE" \
  --out-value "mcts_targets/iter1_value_smoke.csv" \
  --out-policy "mcts_targets/iter1_policy_smoke.csv" \
  --max-rows "${MAX_ROWS:-8}" \
  --root-candidates "${ROOT_CANDIDATES:-16}" \
  --rollout-candidates "${ROLLOUT_CANDIDATES:-6}" \
  --top-k "${TOP_K:-6}" \
  --early-mid-oversample 1.0

"$PYTHON" train_policy_search_distilled.py \
  --init-policy "$POLICY" \
  --targets "mcts_targets/iter1_policy_smoke.csv" \
  --out-dir "checkpoints/policy_mcts_iter1_smoke" \
  --epochs 2 \
  --batch-size 128

"$PYTHON" train_value_search_distilled.py \
  --holdout "$HOLDOUT" \
  --search_targets "mcts_targets/iter1_value_smoke.csv" \
  --target_col terminal_return \
  --out_dir "checkpoints/value_mcts_iter1_smoke" \
  --epochs 2 \
  --batch_size 256 \
  --hidden_dim 64 \
  --n_layers 1 \
  --n_heads 2 \
  --limit_train 1024
