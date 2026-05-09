#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-/opt/pytorch/bin/python}"
GPU_ID="${GPU_ID:-0}"
INVERSE_GLOB="${INVERSE_GLOB:-$ROOT/inverse_current/stones_with_estimates.chunk*.csv}"
SCORE_HARD_LOSS_MAX="${SCORE_HARD_LOSS_MAX:-0.5}"
HOLDOUTS="${HOLDOUTS:-0 22230015 23240026 24250026}"
NUM_SAMPLES="${NUM_SAMPLES:-128}"
TOP_N="${TOP_N:-1000000}"

GPU_LIBS="/opt/pytorch/lib/python3.12/site-packages/nvidia/nvjitlink/lib:/opt/pytorch/lib/python3.12/site-packages/nvidia/cusparse/lib:/opt/pytorch/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/pytorch/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/pytorch/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/pytorch/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/opt/pytorch/lib/python3.12/site-packages/nvidia/cusolver/lib:/opt/pytorch/lib/python3.12/site-packages/nvidia/cufft/lib"
GPU_PYTHONPATH="/mnt/data/curling2/testBrax/testEnv/lib/python3.12/site-packages:/opt/pytorch/lib/python3.12/site-packages"

score_one() {
  local comp_id="$1"
  local run_dir="$ROOT/holdouts/$comp_id"
  local model_ckpt="$run_dir/model_settf_gaussian/model.pt"
  local scoring_dir="$run_dir/scoring_settf_gaussian"
  local report_dir="$run_dir/reports"
  local global_noise="$run_dir/scoring_graphtf_v1bowlingBest/noise/global.json"
  local local_noise="$run_dir/scoring_graphtf_v1bowlingBest/noise/local.json"

  if [[ ! -f "$global_noise" ]]; then
    global_noise="$run_dir/scoring/noise/global.json"
  fi
  if [[ ! -f "$local_noise" ]]; then
    local_noise="$ROOT/noise_versions/v1_bowling.json"
  fi
  if [[ ! -f "$model_ckpt" ]]; then
    echo "missing Gaussian SetTransformer checkpoint: $model_ckpt" >&2
    return 1
  fi

  mkdir -p "$scoring_dir/global" "$scoring_dir/local" "$scoring_dir/noise" "$report_dir"
  cp -f "$global_noise" "$scoring_dir/noise/global.json"
  cp -f "$local_noise" "$scoring_dir/noise/local.json"

  echo "[score] comp=$comp_id global"
  env \
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    JAX_PLATFORMS=cuda \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
    CSAS_PRELOAD_NVIDIA_LIBS=1 \
    LD_LIBRARY_PATH="$GPU_LIBS" \
    PYTHONPATH="$GPU_PYTHONPATH" \
    "$PYTHON" "$ROOT/score_shots_mc_seq.py" \
      --device cuda \
      --stones-csv "$ROOT/2026/Stones.csv" \
      --inverse-glob "$INVERSE_GLOB" \
      --value-model "$model_ckpt" \
      --noise-config "$scoring_dir/noise/global.json" \
      --num-samples "$NUM_SAMPLES" \
      --only-competition "$comp_id" \
      --out-dir "$scoring_dir/global" \
      --hard-loss-max "$SCORE_HARD_LOSS_MAX" \
      --rule-based-terminal

  echo "[score] comp=$comp_id local"
  env \
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    JAX_PLATFORMS=cuda \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
    CSAS_PRELOAD_NVIDIA_LIBS=1 \
    LD_LIBRARY_PATH="$GPU_LIBS" \
    PYTHONPATH="$GPU_PYTHONPATH" \
    "$PYTHON" "$ROOT/score_shots_mc_seq.py" \
      --device cuda \
      --stones-csv "$ROOT/2026/Stones.csv" \
      --inverse-glob "$INVERSE_GLOB" \
      --value-model "$model_ckpt" \
      --noise-config "$scoring_dir/noise/local.json" \
      --num-samples "$NUM_SAMPLES" \
      --only-competition "$comp_id" \
      --out-dir "$scoring_dir/local" \
      --hard-loss-max "$SCORE_HARD_LOSS_MAX" \
      --rule-based-terminal

  cp -f "$scoring_dir/global/shot_scores_comp_${comp_id}.csv" "$scoring_dir/shot_scores_global.csv"
  cp -f "$scoring_dir/local/shot_scores_comp_${comp_id}.csv" "$scoring_dir/shot_scores_local.csv"

  echo "[report] comp=$comp_id"
  "$PYTHON" "$ROOT/player_skill_model.py" \
    --shot-scores "$scoring_dir/shot_scores_global.csv" \
    --competitors-csv "$ROOT/2026/Competitors.csv" \
    --teams-csv "$ROOT/2026/Teams.csv" \
    --out-player-task "$report_dir/player_task_skill_settf_gaussian.csv" \
    --out-player-summary "$report_dir/player_summary_settf_gaussian.csv"

  "$PYTHON" "$ROOT/player_skill_model.py" \
    --shot-scores "$scoring_dir/shot_scores_local.csv" \
    --competitors-csv "$ROOT/2026/Competitors.csv" \
    --teams-csv "$ROOT/2026/Teams.csv" \
    --out-player-task "$report_dir/player_task_skill_local_settf_gaussian.csv" \
    --out-player-summary "$report_dir/player_summary_local_settf_gaussian.csv"

  "$PYTHON" "$ROOT/make_coach_report.py" \
    --shot-scores "$scoring_dir/shot_scores_global.csv" \
    --player-task-skill "$report_dir/player_task_skill_settf_gaussian.csv" \
    --stones-csv "$ROOT/2026/Stones.csv" \
    --competitors-csv "$ROOT/2026/Competitors.csv" \
    --teams-csv "$ROOT/2026/Teams.csv" \
    --competitions-csv "$ROOT/2026/Competition.csv" \
    --games-csv "$ROOT/2026/Games.csv" \
    --out-dir "$report_dir/coach_report/settf_gaussian_work" \
    --top-n-overall "$TOP_N" \
    --top-n-handle "$TOP_N"

  "$PYTHON" "$ROOT/make_coach_report_mc.py" \
    --scores_local "$scoring_dir/shot_scores_local.csv" \
    --scores_global "$scoring_dir/shot_scores_global.csv" \
    --stones-csv "$ROOT/2026/Stones.csv" \
    --competitors-csv "$ROOT/2026/Competitors.csv" \
    --teams-csv "$ROOT/2026/Teams.csv" \
    --competitions-csv "$ROOT/2026/Competition.csv" \
    --games-csv "$ROOT/2026/Games.csv" \
    --out-dir "$report_dir/coach_report_mc/settf_gaussian_work" \
    --top-n-players "$TOP_N"

  mkdir -p "$report_dir/coach_report/figures/settf_gaussian" "$report_dir/coach_report_mc/figures/settf_gaussian"
  cp -f "$report_dir/coach_report/settf_gaussian_work/figures/"*.png "$report_dir/coach_report/figures/settf_gaussian/" 2>/dev/null || true
  cp -f "$report_dir/coach_report_mc/settf_gaussian_work/figures/"*.png "$report_dir/coach_report_mc/figures/settf_gaussian/" 2>/dev/null || true
  cp -f "$report_dir/coach_report/settf_gaussian_work/summary.md" "$report_dir/coach_report/summary_settf_gaussian.md"
  cp -f "$report_dir/coach_report_mc/settf_gaussian_work/summary.md" "$report_dir/coach_report_mc/summary_settf_gaussian.md"
  cp -f "$report_dir/coach_report_mc/settf_gaussian_work/shot_scores_local_vs_global_merged.csv" \
    "$report_dir/coach_report_mc/shot_scores_local_vs_global_merged_settf_gaussian.csv"
}

for comp_id in $HOLDOUTS; do
  score_one "$comp_id"
done
