#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <competition_id> <gpu_id>" >&2
  exit 2
fi

COMP_ID="$1"
GPU_ID="$2"

ROOT="$(cd "$(dirname "$0")" && pwd)"
SCORE_PYTHON="/opt/pytorch/bin/python"
GPU_LIBS="/opt/pytorch/lib/python3.12/site-packages/nvidia/nvjitlink/lib:/opt/pytorch/lib/python3.12/site-packages/nvidia/cusparse/lib:/opt/pytorch/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/pytorch/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/pytorch/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/pytorch/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/opt/pytorch/lib/python3.12/site-packages/nvidia/cusolver/lib:/opt/pytorch/lib/python3.12/site-packages/nvidia/cufft/lib"
GPU_PYTHONPATH="/mnt/data/curling2/testBrax/testEnv/lib/python3.12/site-packages:/opt/pytorch/lib/python3.12/site-packages"
INVERSE_GLOB="${INVERSE_GLOB:-$ROOT/inverse_current/stones_with_estimates.chunk*.csv}"
NOISE_HARD_LOSS_MAX="${NOISE_HARD_LOSS_MAX:-0.25}"
SCORE_HARD_LOSS_MAX="${SCORE_HARD_LOSS_MAX:-0.5}"
VAL_END_FRAC="${VAL_END_FRAC:-0.10}"
SPLIT_SEED="${SPLIT_SEED:-123}"
PHYSICS_DIST="${PHYSICS_DIST:-}"
LOCAL_NOISE_TEMPLATE="${LOCAL_NOISE_TEMPLATE:-$ROOT/noise_versions/v1_bowling_tight.json}"
RUN_DIR="$ROOT/holdouts/$COMP_ID"
MODEL_DIR="$RUN_DIR/model"
SCORING_DIR="$RUN_DIR/scoring"
NOISE_DIR="$SCORING_DIR/noise"
REPORT_DIR="$RUN_DIR/reports"
LOG_DIR="$RUN_DIR/logs"

mkdir -p "$MODEL_DIR" "$SCORING_DIR/global" "$SCORING_DIR/local" "$NOISE_DIR" "$REPORT_DIR" "$LOG_DIR"

MODEL_CKPT="$MODEL_DIR/model.pt"
TRAIN_LOG="$LOG_DIR/train.log"
VAL_COMP_CSV="$MODEL_DIR/val_competitions.csv"
GLOBAL_NOISE="$NOISE_DIR/global.json"
LOCAL_NOISE="$NOISE_DIR/local.json"
GLOBAL_OUT_DIR="$SCORING_DIR/global"
LOCAL_OUT_DIR="$SCORING_DIR/local"
GLOBAL_SCORE_CSV="$SCORING_DIR/shot_scores_global.csv"
LOCAL_SCORE_CSV="$SCORING_DIR/shot_scores_local.csv"
PLAYER_TASK_CSV="$REPORT_DIR/player_task_skill.csv"
PLAYER_SUMMARY_CSV="$REPORT_DIR/player_summary.csv"
PLAYER_TASK_LOCAL_CSV="$REPORT_DIR/player_task_skill_local.csv"
PLAYER_SUMMARY_LOCAL_CSV="$REPORT_DIR/player_summary_local.csv"
PLAYER_TASK_SETTF_CSV="$REPORT_DIR/player_task_skill_settf.csv"
PLAYER_SUMMARY_SETTF_CSV="$REPORT_DIR/player_summary_settf.csv"
PLAYER_TASK_LOCAL_SETTF_CSV="$REPORT_DIR/player_task_skill_local_settf.csv"
PLAYER_SUMMARY_LOCAL_SETTF_CSV="$REPORT_DIR/player_summary_local_settf.csv"
COACH_REPORT_DIR="$REPORT_DIR/coach_report"
COACH_REPORT_MC_DIR="$REPORT_DIR/coach_report_mc"
COACH_FIG_SETTF="$COACH_REPORT_DIR/figures/settf"
COACH_MC_FIG_SETTF="$COACH_REPORT_MC_DIR/figures/settf"
COACH_REPORT_SETTF_SUMMARY="$COACH_REPORT_DIR/summary_settf.md"
COACH_REPORT_MC_SETTF_SUMMARY="$COACH_REPORT_MC_DIR/summary_settf.md"
COACH_REPORT_MC_SETTF_MERGED="$COACH_REPORT_MC_DIR/shot_scores_local_vs_global_merged_settf.csv"
PIPELINE_LOG="$LOG_DIR/pipeline.log"

unique_trash_dest() {
  local trash_dir="$1"
  local base_name="$2"
  local dest="$trash_dir/$base_name"
  if [[ -e "$dest" ]]; then
    dest="$trash_dir/${base_name}_$(date -u +%Y%m%d_%H%M%S)"
  fi
  printf '%s\n' "$dest"
}

move_into_trash() {
  local src="$1"
  local trash_dir="$2"
  if [[ ! -e "$src" ]]; then
    return 0
  fi
  mkdir -p "$trash_dir"
  local base_name
  base_name="$(basename "$src")"
  local dest
  dest="$(unique_trash_dest "$trash_dir" "$base_name")"
  mv "$src" "$dest"
  echo "[trash] moved $src -> $dest"
}

trash_legacy_report_artifacts() {
  local report_dir="$1"
  local coach_dir="$2"
  local coach_mc_dir="$3"
  local report_trash="$report_dir/old_viz_trash"
  local coach_trash="$coach_dir/old_viz_trash"
  local coach_mc_trash="$coach_mc_dir/old_viz_trash"

  mkdir -p "$report_dir" "$coach_dir" "$coach_mc_dir"

  move_into_trash "$report_dir/old_viz" "$report_trash"
  move_into_trash "$report_dir/v1_bowling_largenoise" "$report_trash"
  move_into_trash "$report_dir/old_viz_2_player_summary.csv" "$report_trash"
  move_into_trash "$report_dir/old_viz_2_player_summary_local.csv" "$report_trash"
  move_into_trash "$report_dir/old_viz_2_player_task_skill.csv" "$report_trash"
  move_into_trash "$report_dir/old_viz_2_player_task_skill_local.csv" "$report_trash"
  move_into_trash "$report_dir/player_task_skill_settf.csv" "$report_trash"
  move_into_trash "$report_dir/player_summary_settf.csv" "$report_trash"
  move_into_trash "$report_dir/player_task_skill_local_settf.csv" "$report_trash"
  move_into_trash "$report_dir/player_summary_local_settf.csv" "$report_trash"

  move_into_trash "$coach_dir/figures" "$coach_trash"
  move_into_trash "$coach_dir/old_viz_2" "$coach_trash"
  move_into_trash "$coach_dir/v1_bowling_largenoise" "$coach_trash"
  move_into_trash "$coach_dir/summary_settf.md" "$coach_trash"

  move_into_trash "$coach_mc_dir/figures" "$coach_mc_trash"
  move_into_trash "$coach_mc_dir/old_viz_2" "$coach_mc_trash"
  move_into_trash "$coach_mc_dir/v1_bowling_largenoise" "$coach_mc_trash"
  move_into_trash "$coach_mc_dir/summary_settf.md" "$coach_mc_trash"
  move_into_trash "$coach_mc_dir/shot_scores_local_vs_global_merged_settf.csv" "$coach_mc_trash"
}

exec > >(tee -a "$PIPELINE_LOG") 2>&1

echo "[start] competition=$COMP_ID gpu=$GPU_ID inverse_glob=$INVERSE_GLOB"

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  "$SCORE_PYTHON" "$ROOT/train_holdout_models_cond3.py" \
    --only_holdout "$COMP_ID" \
    --device "cuda:$GPU_ID" \
    --val_end_frac "$VAL_END_FRAC" \
    --split_seed "$SPLIT_SEED" \
    --log_file "$TRAIN_LOG"
else
  echo "[skip] training competition=$COMP_ID"
fi

python "$ROOT/fit_execution_noise.py" \
  --inverse-glob "$INVERSE_GLOB" \
  --stones-csv "$ROOT/2026/Stones.csv" \
  --out "$GLOBAL_NOISE" \
  --exclude-competition-id "$COMP_ID" \
  --hard-loss-max "$NOISE_HARD_LOSS_MAX"

if [[ ! -f "$LOCAL_NOISE_TEMPLATE" ]]; then
  echo "missing local noise template: $LOCAL_NOISE_TEMPLATE" >&2
  exit 1
fi
cp "$LOCAL_NOISE_TEMPLATE" "$LOCAL_NOISE"

PHYSICS_DIST_FLAG=""
if [[ -n "$PHYSICS_DIST" && -f "$PHYSICS_DIST" ]]; then
  PHYSICS_DIST_FLAG="--physics-dist $PHYSICS_DIST"
fi

env \
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  JAX_PLATFORMS=cuda \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
  CSAS_PRELOAD_NVIDIA_LIBS=1 \
  LD_LIBRARY_PATH="$GPU_LIBS" \
  PYTHONPATH="$GPU_PYTHONPATH" \
  "$SCORE_PYTHON" "$ROOT/score_shots_mc_seq.py" \
  --device cuda \
  --stones-csv "$ROOT/2026/Stones.csv" \
  --inverse-glob "$INVERSE_GLOB" \
  --value-model "$MODEL_CKPT" \
  --noise-config "$GLOBAL_NOISE" \
  --only-competition "$COMP_ID" \
  --out-dir "$GLOBAL_OUT_DIR" \
  --hard-loss-max "$SCORE_HARD_LOSS_MAX" \
  --rule-based-terminal \
  $PHYSICS_DIST_FLAG

env \
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  JAX_PLATFORMS=cuda \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
  CSAS_PRELOAD_NVIDIA_LIBS=1 \
  LD_LIBRARY_PATH="$GPU_LIBS" \
  PYTHONPATH="$GPU_PYTHONPATH" \
  "$SCORE_PYTHON" "$ROOT/score_shots_mc_seq.py" \
  --device cuda \
  --stones-csv "$ROOT/2026/Stones.csv" \
  --inverse-glob "$INVERSE_GLOB" \
  --value-model "$MODEL_CKPT" \
  --noise-config "$LOCAL_NOISE" \
  --only-competition "$COMP_ID" \
  --out-dir "$LOCAL_OUT_DIR" \
  --hard-loss-max "$SCORE_HARD_LOSS_MAX" \
  --rule-based-terminal \
  $PHYSICS_DIST_FLAG

cp "$GLOBAL_OUT_DIR/shot_scores_comp_${COMP_ID}.csv" "$GLOBAL_SCORE_CSV"
cp "$LOCAL_OUT_DIR/shot_scores_comp_${COMP_ID}.csv" "$LOCAL_SCORE_CSV"

trash_legacy_report_artifacts "$REPORT_DIR" "$COACH_REPORT_DIR" "$COACH_REPORT_MC_DIR"

python "$ROOT/player_skill_model.py" \
  --shot-scores "$GLOBAL_SCORE_CSV" \
  --competitors-csv "$ROOT/2026/Competitors.csv" \
  --teams-csv "$ROOT/2026/Teams.csv" \
  --out-player-task "$PLAYER_TASK_CSV" \
  --out-player-summary "$PLAYER_SUMMARY_CSV"

python "$ROOT/player_skill_model.py" \
  --shot-scores "$LOCAL_SCORE_CSV" \
  --competitors-csv "$ROOT/2026/Competitors.csv" \
  --teams-csv "$ROOT/2026/Teams.csv" \
  --out-player-task "$PLAYER_TASK_LOCAL_CSV" \
  --out-player-summary "$PLAYER_SUMMARY_LOCAL_CSV"

python "$ROOT/make_coach_report.py" \
  --shot-scores "$GLOBAL_SCORE_CSV" \
  --player-task-skill "$PLAYER_TASK_CSV" \
  --stones-csv "$ROOT/2026/Stones.csv" \
  --competitors-csv "$ROOT/2026/Competitors.csv" \
  --teams-csv "$ROOT/2026/Teams.csv" \
  --competitions-csv "$ROOT/2026/Competition.csv" \
  --games-csv "$ROOT/2026/Games.csv" \
  --out-dir "$COACH_REPORT_DIR"

python "$ROOT/make_coach_report_mc.py" \
  --scores_local "$LOCAL_SCORE_CSV" \
  --scores_global "$GLOBAL_SCORE_CSV" \
  --stones-csv "$ROOT/2026/Stones.csv" \
  --competitors-csv "$ROOT/2026/Competitors.csv" \
  --teams-csv "$ROOT/2026/Teams.csv" \
  --competitions-csv "$ROOT/2026/Competition.csv" \
  --games-csv "$ROOT/2026/Games.csv" \
  --out-dir "$COACH_REPORT_MC_DIR"

mkdir -p "$COACH_FIG_SETTF" "$COACH_MC_FIG_SETTF"
cp -f "$PLAYER_TASK_CSV" "$PLAYER_TASK_SETTF_CSV"
cp -f "$PLAYER_SUMMARY_CSV" "$PLAYER_SUMMARY_SETTF_CSV"
cp -f "$PLAYER_TASK_LOCAL_CSV" "$PLAYER_TASK_LOCAL_SETTF_CSV"
cp -f "$PLAYER_SUMMARY_LOCAL_CSV" "$PLAYER_SUMMARY_LOCAL_SETTF_CSV"
cp -f "$COACH_REPORT_DIR/summary.md" "$COACH_REPORT_SETTF_SUMMARY"
cp -f "$COACH_REPORT_MC_DIR/summary.md" "$COACH_REPORT_MC_SETTF_SUMMARY"
cp -f "$COACH_REPORT_MC_DIR/shot_scores_local_vs_global_merged.csv" "$COACH_REPORT_MC_SETTF_MERGED"
cp -f "$COACH_REPORT_DIR/figures/"*.png "$COACH_FIG_SETTF/" 2>/dev/null || true
cp -f "$COACH_REPORT_MC_DIR/figures/"*.png "$COACH_MC_FIG_SETTF/" 2>/dev/null || true

echo "[done] competition=$COMP_ID gpu=$GPU_ID"
