#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs search_targets checkpoints

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.30}"
source ./setup_runtime_env.sh

HOLDOUT="${HOLDOUT:-0}"
GPUS="${GPUS:-0 1 2 3}"
OLD_VALUE="/mnt/data/curling2/csas_fixed/holdouts/${HOLDOUT}/model_settf_gaussian/model.pt"

CUDA_VISIBLE_DEVICES="$(echo "$GPUS" | tr ' ' ',')" "$PYTHON" train_policy_prior.py \
  --holdout "$HOLDOUT" \
  --out_dir checkpoints/policy_prior_h${HOLDOUT} \
  --epochs "${POLICY_EPOCHS:-80}" \
  --batch_size "${POLICY_BATCH:-1024}" \
  --n_mixtures 16 \
  --max_loss 0.12

rm -f search_targets/holdout${HOLDOUT}_search_shard*.csv
i=0
for gpu in $GPUS; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" generate_search_targets.py \
    --holdout "$HOLDOUT" \
    --policy checkpoints/policy_prior_h${HOLDOUT}/model.pt \
    --value "$OLD_VALUE" \
    --out "search_targets/holdout${HOLDOUT}_search_shard${i}.csv" \
    --shard_index "$i" \
    --num_shards 4 \
    --candidates "${CANDIDATES:-384}" \
    --rollout_depth "${ROLLOUT_DEPTH:-2}" \
    --child_candidates "${CHILD_CANDIDATES:-96}" \
    --kernel_bandwidth "${KERNEL_BANDWIDTH:-0.75}" \
    --early_mid_oversample "${EARLY_MID_OVERSAMPLE:-1.75}" \
    --device auto &
  i=$((i + 1))
done
wait

HOLDOUT="$HOLDOUT" "$PYTHON" - <<'PY'
import os
from pathlib import Path
import pandas as pd
root = Path("search_targets")
holdout = os.environ["HOLDOUT"]
parts = sorted(root.glob(f"holdout{holdout}_search_shard*.csv"))
if not parts:
    raise SystemExit("no search shards found")
out = root / f"holdout{holdout}_search_targets.csv"
pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).to_csv(out, index=False)
print(out)
PY

SEARCH_TARGETS="search_targets/holdout${HOLDOUT}_search_targets.csv"
CUDA_VISIBLE_DEVICES="$(echo "$GPUS" | tr ' ' ',')" "$PYTHON" train_value_search_distilled.py \
  --holdout "$HOLDOUT" \
  --search_targets "$SEARCH_TARGETS" \
  --out_dir "checkpoints/value_search_distilled_h${HOLDOUT}" \
  --epochs "${VALUE_EPOCHS:-140}" \
  --batch_size "${VALUE_BATCH:-1536}" \
  --search_weight "${SEARCH_WEIGHT:-2}"

"$PYTHON" evaluate_value_by_shot.py \
  --holdout "$HOLDOUT" \
  --old "$OLD_VALUE" \
  --new "checkpoints/value_search_distilled_h${HOLDOUT}/model.pt" \
  --out "logs/holdout${HOLDOUT}_value_by_shot_comparison.csv"
