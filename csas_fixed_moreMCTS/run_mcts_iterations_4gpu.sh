#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs mcts_targets checkpoints

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.25}"
source ./setup_runtime_env.sh

HOLDOUT="${HOLDOUT:-0}"
GPUS="${GPUS:-0 1 2 3}"
ITERS="${ITERS:-3}"
POLICY="checkpoints/policy_prior_h${HOLDOUT}/model.pt"
VALUE="/mnt/data/curling2/csas_fixed/holdouts/${HOLDOUT}/model_settf_gaussian/model.pt"

for iter in $(seq 1 "$ITERS"); do
  rm -f "mcts_targets/iter${iter}_value_shard"*.csv "mcts_targets/iter${iter}_policy_shard"*.csv
  i=0
  for gpu in $GPUS; do
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" generate_mcts_iteration_targets.py \
      --holdout "$HOLDOUT" \
      --policy "$POLICY" \
      --value "$VALUE" \
      --out-value "mcts_targets/iter${iter}_value_shard${i}.csv" \
      --out-policy "mcts_targets/iter${iter}_policy_shard${i}.csv" \
      --shard-index "$i" \
      --num-shards 4 \
      --root-candidates "${ROOT_CANDIDATES:-96}" \
      --rollout-candidates "${ROLLOUT_CANDIDATES:-24}" \
      --top-k "${TOP_K:-16}" \
      --early-mid-oversample "${EARLY_MID_OVERSAMPLE:-1.5}" \
      --device auto &
    i=$((i + 1))
  done
  wait

  ITER="$iter" "$PYTHON" - <<'PY'
import os
from pathlib import Path
import pandas as pd
it = os.environ["ITER"]
root = Path("mcts_targets")
for kind in ("value", "policy"):
    parts = sorted(root.glob(f"iter{it}_{kind}_shard*.csv"))
    if not parts:
        raise SystemExit(f"missing {kind} shards")
    pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).to_csv(root / f"iter{it}_{kind}.csv", index=False)
PY

  "$PYTHON" train_policy_search_distilled.py \
    --init-policy "$POLICY" \
    --targets "mcts_targets/iter${iter}_policy.csv" \
    --out-dir "checkpoints/policy_mcts_iter${iter}" \
    --epochs "${POLICY_EPOCHS:-40}" \
    --batch-size "${POLICY_BATCH:-2048}"

  CUDA_VISIBLE_DEVICES="$(echo "$GPUS" | tr ' ' ',')" "$PYTHON" train_value_search_distilled.py \
    --holdout "$HOLDOUT" \
    --search_targets "mcts_targets/iter${iter}_value.csv" \
    --target_col terminal_return \
    --out_dir "checkpoints/value_mcts_iter${iter}" \
    --epochs "${VALUE_EPOCHS:-100}" \
    --batch_size "${VALUE_BATCH:-1536}" \
    --search_weight "${SEARCH_WEIGHT:-2}" \
    --init_checkpoint "$VALUE"

  "$PYTHON" evaluate_value_by_shot.py \
    --holdout "$HOLDOUT" \
    --old "/mnt/data/curling2/csas_fixed/holdouts/${HOLDOUT}/model_settf_gaussian/model.pt" \
    --new "checkpoints/value_mcts_iter${iter}/model.pt" \
    --out "logs/mcts_iter${iter}_value_by_shot_comparison.csv"

  POLICY="checkpoints/policy_mcts_iter${iter}/model.pt"
  VALUE="checkpoints/value_mcts_iter${iter}/model.pt"
done
