#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs search_targets checkpoints

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.30}"
source ./setup_runtime_env.sh

HOLDOUT="${HOLDOUT:-0}"
OLD_VALUE="/mnt/data/curling2/csas_fixed/holdouts/${HOLDOUT}/model_settf_gaussian/model.pt"

"$PYTHON" train_policy_prior.py \
  --holdout "$HOLDOUT" \
  --out_dir checkpoints/policy_prior_smoke \
  --epochs 2 \
  --patience 2 \
  --batch_size 256 \
  --hidden_dim 64 \
  --n_layers 1 \
  --n_heads 2 \
  --n_mixtures 4 \
  --limit_train 1024

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" generate_search_targets.py \
  --holdout "$HOLDOUT" \
  --policy checkpoints/policy_prior_smoke/model.pt \
  --value "$OLD_VALUE" \
  --out search_targets/holdout${HOLDOUT}_smoke.csv \
  --max_rows 8 \
  --candidates 16 \
  --rollout_depth 1 \
  --device auto

"$PYTHON" train_value_search_distilled.py \
  --holdout "$HOLDOUT" \
  --search_targets search_targets/holdout${HOLDOUT}_smoke.csv \
  --out_dir checkpoints/value_search_distilled_smoke \
  --epochs 2 \
  --patience 2 \
  --batch_size 256 \
  --hidden_dim 64 \
  --n_layers 1 \
  --n_heads 2 \
  --limit_train 1024

"$PYTHON" evaluate_value_by_shot.py \
  --holdout "$HOLDOUT" \
  --old "$OLD_VALUE" \
  --new checkpoints/value_search_distilled_smoke/model.pt \
  --out logs/smoke_value_by_shot.csv
