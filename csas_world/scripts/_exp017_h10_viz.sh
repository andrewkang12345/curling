#!/usr/bin/env bash
# After the EXP-017 h10-extension finishes training, render two full-end viz (like
# artifacts/figures/game_anchor_vs_v3) for the h10-trained model: (1) self-play, (2) vs the human
# prior using the winrate setup (noisy robust selection + noisy realized execution). CPU so it can
# run concurrently with the GPU eval.
set -u
cd /mnt/data/curling2/csas_world
export PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu
LOG=checkpoints/csas_world/exp_017_deploy_robust/exp017_h10.log
DIR=checkpoints/csas_world/exp_017_deploy_robust

for i in $(seq 1 180); do   # wait for training done (up to ~6h)
  grep -qa "CURRICULUM_EXIT" "$LOG" 2>/dev/null && break
  pgrep -f "run_curriculum.py --config configs/exp_017_deploy" >/dev/null 2>&1 || { sleep 20; grep -qa CURRICULUM_EXIT "$LOG" || break; }
  sleep 120
done

CH="$DIR/h10/r1/model.pt"; [ -f "$CH" ] || CH="$DIR/h10/r0/model.pt"
echo "=== viz model (h10-trained, full end): $CH ==="
[ -f "$CH" ] || { echo ">> h10 ckpt missing; cannot viz"; echo "VIZ_DONE"; exit 0; }

echo "=== VIZ 1: self-play (ours vs ours; deterministic, clean trajectories) ==="
python3 scripts/viz_game_match.py --device cpu \
  --player-a "$CH" --label-a ours --player-b "$CH" --label-b ours_mirror \
  --seed 7 --root-idx 0 --out artifacts/figures/game_ours_vs_ours 2>&1 \
  | grep -aoaE "\[viz\] wrote[^$]*" | tail -1

echo "=== VIZ 2: vs human prior (winrate setup: noisy robust selection + noisy execution) ==="
python3 scripts/viz_game_match.py --device cpu \
  --player-a "$CH" --label-a ours --player-b prior --label-b human_prior \
  --noisy-select --realize-noise --seed 7 --root-idx 0 --out artifacts/figures/game_ours_vs_prior 2>&1 \
  | grep -aoaE "\[viz\] wrote[^$]*" | tail -1

echo "=== frames written ==="
echo "self-play: $(ls artifacts/figures/game_ours_vs_ours/*.png 2>/dev/null | wc -l) PNGs"
echo "vs prior:  $(ls artifacts/figures/game_ours_vs_prior/*.png 2>/dev/null | wc -l) PNGs"
echo "VIZ_DONE"
