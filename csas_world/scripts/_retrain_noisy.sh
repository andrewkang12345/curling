#!/bin/bash
# Noise-aware retrain: GPU-collect MCTS targets with local execution noise
# (noise_samples=8), then train the joint model. Collection and training run in
# separate subshells so training uses a clean torch env (not the vendored-JAX
# LD_LIBRARY_PATH).
set -uo pipefail
cd /mnt/data/curling2/csas_world
PRIOR=/mnt/data/curling2/csas_v3/checkpoints/policy/human_prior_fullcov/best.pt
VALUE=/mnt/data/curling2/csas_v3/checkpoints/value/holdout0/model.pt
OUT=artifacts/replay/mcts/anchor_noisy
CFG=configs/anchor_noisy.yaml
ROOTS=200
mkdir -p "$OUT"

echo "[$(date +%H:%M:%S)] === GPU noisy MCTS collection (noise_samples=8) ==="
(
  source scripts/setup_gpu.sh
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  for h in 1 2 3 4 5 6 7 8 9 10; do
    hh=$(printf "h%02d" "$h")
    t0=$(date +%s)
    for k in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$k XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
      python3 -m world.search.collect --config "$CFG" --horizon "$h" --max-roots "$ROOTS" \
        --kind mcts --policy "$PRIOR" --value "$VALUE" \
        --out "$OUT/$hh/shard$k.npz" --num-shards 4 --shard-id $k --device cuda:0 --seed $((100+h)) &
    done
    wait
    echo "[$(date +%H:%M:%S)] $hh done in $(( $(date +%s) - t0 ))s ($(find $OUT/$hh -name '*.npz'|wc -l) shards)"
  done
)
echo "[$(date +%H:%M:%S)] === noisy MCTS collection done; tally ==="
PYTHONPATH=src python3 -c "import world; from world.replay.buffers import load_shards; d=load_shards('$OUT',5,24); print('noisy mcts records:', len(d) if d else 0)" 2>/dev/null | grep -a records

echo "[$(date +%H:%M:%S)] === train joint model on noisy targets (4-GPU DDP) ==="
(
  export PYTHONPATH=src
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python3 scripts/train_world.py --config "$CFG" --ablation default \
    --mcts-dir "$OUT" --sim-dir artifacts/replay/sim \
    --out checkpoints/csas_world/anchor_noisy --run-name anchor_noisy
)
echo "[$(date +%H:%M:%S)] RETRAIN_NOISY_DONE"
