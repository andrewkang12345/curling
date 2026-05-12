#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs mcts_targets checkpoints

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.25}"
source ./setup_runtime_env.sh

HOLDOUT="${HOLDOUT:-0}"
GPUS="${GPUS:-0 1 2 3}"
MAX_ROWS_PER_SHARD="${MAX_ROWS_PER_SHARD:-100}"
POLICY="${POLICY:-checkpoints/policy_prior_h${HOLDOUT}/model.pt}"
VALUE="${VALUE:-/mnt/data/curling2/csas_fixed/holdouts/${HOLDOUT}/model_settf_gaussian/model.pt}"

rm -f mcts_targets/diag_value_shard*.csv mcts_targets/diag_policy_shard*.csv
i=0
for gpu in $GPUS; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" generate_mcts_iteration_targets.py \
    --holdout "$HOLDOUT" \
    --policy "$POLICY" \
    --value "$VALUE" \
    --out-value "mcts_targets/diag_value_shard${i}.csv" \
    --out-policy "mcts_targets/diag_policy_shard${i}.csv" \
    --shard-index "$i" \
    --num-shards 4 \
    --max-rows "$MAX_ROWS_PER_SHARD" \
    --root-candidates "${ROOT_CANDIDATES:-32}" \
    --rollout-candidates "${ROLLOUT_CANDIDATES:-8}" \
    --top-k "${TOP_K:-8}" \
    --early-mid-oversample "${EARLY_MID_OVERSAMPLE:-1.2}" \
    --device auto &
  i=$((i + 1))
done
wait

"$PYTHON" - <<'PY'
from pathlib import Path
import pandas as pd
root = Path("mcts_targets")
for kind in ("value", "policy"):
    parts = sorted(root.glob(f"diag_{kind}_shard*.csv"))
    if not parts:
        raise SystemExit(f"missing {kind} shards")
    pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).to_csv(root / f"diag_{kind}.csv", index=False)
PY

"$PYTHON" train_policy_search_distilled.py \
  --init-policy "$POLICY" \
  --targets "mcts_targets/diag_policy.csv" \
  --out-dir "checkpoints/policy_mcts_diag" \
  --epochs "${POLICY_EPOCHS:-12}" \
  --batch-size "${POLICY_BATCH:-1024}"

CUDA_VISIBLE_DEVICES="$(echo "$GPUS" | tr ' ' ',')" "$PYTHON" train_value_search_distilled.py \
  --holdout "$HOLDOUT" \
  --search_targets "mcts_targets/diag_value.csv" \
  --target_col terminal_return \
  --out_dir "checkpoints/value_mcts_diag" \
  --epochs "${VALUE_EPOCHS:-40}" \
  --batch_size "${VALUE_BATCH:-1536}" \
  --search_weight "${SEARCH_WEIGHT:-2}" \
  --init_checkpoint "$VALUE"

"$PYTHON" evaluate_value_by_shot.py \
  --holdout "$HOLDOUT" \
  --old "/mnt/data/curling2/csas_fixed/holdouts/${HOLDOUT}/model_settf_gaussian/model.pt" \
  --new "checkpoints/value_mcts_diag/model.pt" \
  --out "logs/mcts_diag_value_by_shot_comparison.csv"
