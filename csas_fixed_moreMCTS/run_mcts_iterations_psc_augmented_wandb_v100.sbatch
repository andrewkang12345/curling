#!/bin/bash
#SBATCH -J curling-mcts-v100
#SBATCH -A cis260099p
#SBATCH -p GPU
#SBATCH --gres=gpu:v100-32:8
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH -o logs/psc_mcts_v100_%j.out
#SBATCH -e logs/psc_mcts_v100_%j.err

set -euo pipefail

ROOT="${ROOT:-/ocean/projects/cis260099p/akang3/curling2}"
EXP="${EXP:-$ROOT/csas_fixed_moreMCTS}"
FIXED="${FIXED:-$ROOT/csas_fixed}"
PY="${PY:-/ocean/projects/cis260099p/akang3/polymer/envs/poly_torch/bin/python}"
DEPS="${DEPS:-$EXP/psc_pydeps}"
export DEPS
cd "$EXP"
mkdir -p logs mcts_targets checkpoints "$DEPS"

bootstrap_python_deps() {
  local jax_version="${JAX_CUDA_VERSION:-0.4.38}"
  if ! PYTHONPATH="$DEPS:${PYTHONPATH:-}" "$PY" - <<'PY' >/dev/null 2>&1
import jax
assert jax.devices("gpu")
PY
  then
    rm -rf \
      "$DEPS"/jax "$DEPS"/jax-* \
      "$DEPS"/jaxlib "$DEPS"/jaxlib-* \
      "$DEPS"/jax_plugins "$DEPS"/jax_cuda* \
      "$DEPS"/nvidia "$DEPS"/nvidia_*
    "$PY" -m pip install --upgrade --force-reinstall --target "$DEPS" \
      "jax[cuda12]==${jax_version}" wandb >/dev/null
  fi
}

bootstrap_python_deps

export PYTHONPATH="$DEPS:$EXP:$FIXED:$FIXED/inverse:$FIXED/valueModel:$FIXED/valueModel/ablation:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$("$PY" - <<'PY'
import os
from pathlib import Path
roots = [Path(os.environ["DEPS"]) / "nvidia"]
libs = []
for root in roots:
    if root.exists():
        libs.extend(str(p) for p in root.glob("*/lib") if p.is_dir())
print(":".join(libs))
PY
):${LD_LIBRARY_PATH:-}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.18}"
export CSAS_PRELOAD_NVIDIA_LIBS="${CSAS_PRELOAD_NVIDIA_LIBS:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-curling-moreMCTS}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-psc_augmented_warm_h0_v100}"

HOLDOUT="${HOLDOUT:-0}"
MAX_ITERS="${MAX_ITERS:-8}"
CONV_PATIENCE="${CONV_PATIENCE:-2}"
MIN_DELTA="${MIN_DELTA:-0.002}"
ROOT_CANDIDATES="${ROOT_CANDIDATES:-96}"
ROLLOUT_CANDIDATES="${ROLLOUT_CANDIDATES:-24}"
TOP_K="${TOP_K:-16}"
EARLY_MID_OVERSAMPLE="${EARLY_MID_OVERSAMPLE:-1.5}"
POLICY_EPOCHS="${POLICY_EPOCHS:-40}"
VALUE_EPOCHS="${VALUE_EPOCHS:-100}"
SEARCH_WEIGHT="${SEARCH_WEIGHT:-2}"

POLICY_WARM="checkpoints/policy_prior_preplaced_h${HOLDOUT}_flip_team/model.pt"
VALUE_WARM="checkpoints/value_preplaced_baseline_h${HOLDOUT}_flip_team/model.pt"
BASE_VALUE="$FIXED/holdouts/${HOLDOUT}/model_settf_gaussian/model.pt"

log_wandb() {
  PHASE="$1" ITERATION="$2" SUMMARY_JSON="$3" "$PY" - <<'PY'
import json, os
from pathlib import Path
import wandb
phase = os.environ["PHASE"]
iteration = int(os.environ["ITERATION"])
summary_path = Path(os.environ["SUMMARY_JSON"])
run = wandb.init(
    project=os.environ.get("WANDB_PROJECT", "curling-moreMCTS"),
    entity=(os.environ.get("WANDB_ENTITY") or None),
    group=os.environ.get("WANDB_RUN_GROUP", "psc_augmented_warm_h0_v100"),
    job_type=phase,
    name=f"{phase}-iter{iteration}",
    reinit=True,
    config={"phase": phase, "iteration": iteration, "summary_path": str(summary_path)},
)
if summary_path.exists():
    data = json.loads(summary_path.read_text())
    metrics = {"iteration": iteration}
    for split in ("val_metrics", "test_metrics"):
        if isinstance(data.get(split), dict):
            prefix = split.replace("_metrics", "")
            for k, v in data[split].items():
                if isinstance(v, (int, float)):
                    metrics[f"{prefix}/{k}"] = float(v)
    for k in ("best_epoch", "best_val_nll", "best_val_weighted_nll"):
        if isinstance(data.get(k), (int, float)):
            metrics[k] = float(data[k])
    run.log(metrics)
    run.summary.update(metrics)
    art = wandb.Artifact(f"{phase}-iter{iteration}-summary", type="summary")
    art.add_file(str(summary_path))
    run.log_artifact(art)
else:
    run.log({"iteration": iteration, "missing_summary": 1})
run.finish()
PY
}

echo "START $(date) host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset}"
"$PY" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
import jax
print("jax devices", jax.devices())
PY

echo "Train augmented policy prior $(date)"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PY" train_policy_prior.py \
  --holdout "$HOLDOUT" \
  --out_dir "checkpoints/policy_prior_preplaced_h${HOLDOUT}_flip_team" \
  --include_preplaced \
  --preplaced_max_loss 0.5 \
  --preplaced_weight 1 \
  --max_loss 0.12 \
  --epochs 80 \
  --patience 12 \
  --batch_size 2048 \
  --device auto
log_wandb policy_prior 0 "checkpoints/policy_prior_preplaced_h${HOLDOUT}_flip_team/summary.json"

echo "Train augmented value warm start $(date)"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PY" train_value_search_distilled.py \
  --holdout "$HOLDOUT" \
  --out_dir "checkpoints/value_preplaced_baseline_h${HOLDOUT}_flip_team" \
  --include_preplaced \
  --preplaced_weight 1 \
  --include_synth_terminal \
  --epochs 200 \
  --patience 30 \
  --batch_size 3072 \
  --device auto
log_wandb value_warm 0 "checkpoints/value_preplaced_baseline_h${HOLDOUT}_flip_team/summary.json"

POLICY="$POLICY_WARM"
VALUE="$VALUE_WARM"
BEST=999999
BAD=0
rm -f logs/psc_aug_convergence_state.json

for iter in $(seq 1 "$MAX_ITERS"); do
  echo "ITER $iter target generation $(date) policy=$POLICY value=$VALUE"
  rm -f "mcts_targets/iter${iter}_value_shard"*.csv "mcts_targets/iter${iter}_policy_shard"*.csv
  for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES="$i" "$PY" generate_mcts_iteration_targets.py \
      --holdout "$HOLDOUT" \
      --policy "$POLICY" \
      --value "$VALUE" \
      --out-value "mcts_targets/iter${iter}_value_shard${i}.csv" \
      --out-policy "mcts_targets/iter${iter}_policy_shard${i}.csv" \
      --shard-index "$i" \
      --num-shards 8 \
      --root-candidates "$ROOT_CANDIDATES" \
      --rollout-candidates "$ROLLOUT_CANDIDATES" \
      --top-k "$TOP_K" \
      --early-mid-oversample "$EARLY_MID_OVERSAMPLE" \
      --include-preplaced-roots \
      --device auto &
  done
  wait

  ITER="$iter" "$PY" - <<'PY'
import os
from pathlib import Path
import pandas as pd
it = os.environ["ITER"]
root = Path("mcts_targets")
for kind in ("value", "policy"):
    parts = sorted(root.glob(f"iter{it}_{kind}_shard*.csv"))
    if not parts:
        raise SystemExit(f"missing {kind} shards")
    frames = [pd.read_csv(p) for p in parts]
    pd.concat(frames, ignore_index=True).to_csv(root / f"iter{it}_{kind}.csv", index=False)
    print(kind, len(parts), "shards", "rows", sum(len(f) for f in frames))
PY

  echo "ITER $iter policy distill $(date)"
  CUDA_VISIBLE_DEVICES=0 "$PY" train_policy_search_distilled.py \
    --init-policy "$POLICY" \
    --targets "mcts_targets/iter${iter}_policy.csv" \
    --out-dir "checkpoints/policy_mcts_iter${iter}_aug" \
    --epochs "$POLICY_EPOCHS" \
    --batch-size 4096
  log_wandb policy_mcts "$iter" "checkpoints/policy_mcts_iter${iter}_aug/summary.json"

  echo "ITER $iter value train warm-start $(date) init=$VALUE"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PY" train_value_search_distilled.py \
    --holdout "$HOLDOUT" \
    --search_targets "mcts_targets/iter${iter}_value.csv" \
    --target_col terminal_return \
    --out_dir "checkpoints/value_mcts_iter${iter}_aug" \
    --epochs "$VALUE_EPOCHS" \
    --batch_size 3072 \
    --search_weight "$SEARCH_WEIGHT" \
    --init_checkpoint "$VALUE" \
    --include_preplaced \
    --preplaced_weight 1
  log_wandb value_mcts "$iter" "checkpoints/value_mcts_iter${iter}_aug/summary.json"

  "$PY" evaluate_value_by_shot.py \
    --holdout "$HOLDOUT" \
    --old "$BASE_VALUE" \
    --new "checkpoints/value_mcts_iter${iter}_aug/model.pt" \
    --out "logs/psc_mcts_iter${iter}_aug_value_by_shot_comparison.csv"

  ITER="$iter" BEST="$BEST" BAD="$BAD" MIN_DELTA="$MIN_DELTA" CONV_PATIENCE="$CONV_PATIENCE" "$PY" - <<'PY' > logs/psc_aug_convergence_next.env
import json, os
from pathlib import Path
it = int(os.environ["ITER"])
best = float(os.environ["BEST"])
bad = int(os.environ["BAD"])
min_delta = float(os.environ["MIN_DELTA"])
patience = int(os.environ["CONV_PATIENCE"])
summary = json.loads(Path(f"checkpoints/value_mcts_iter{it}_aug/summary.json").read_text())
val = float(summary["val_metrics"]["mse"])
if val < best - min_delta:
    best = val
    bad = 0
else:
    bad += 1
stop = bad >= patience
Path("logs/psc_aug_convergence_state.json").write_text(json.dumps({"iteration": it, "val_mse": val, "best": best, "bad": bad, "stop": stop}, indent=2) + "\n")
print(f"BEST={best}")
print(f"BAD={bad}")
print(f"STOP={1 if stop else 0}")
PY
  source logs/psc_aug_convergence_next.env
  echo "ITER $iter convergence best=$BEST bad=$BAD stop=$STOP"

  POLICY="checkpoints/policy_mcts_iter${iter}_aug/model.pt"
  VALUE="checkpoints/value_mcts_iter${iter}_aug/model.pt"
  if [ "$STOP" = "1" ]; then
    echo "CONVERGED at iter=$iter $(date)"
    break
  fi
done

echo "DONE $(date)"
