#!/usr/bin/env bash
set -uo pipefail
cd /mnt/data/curling2/csas_world
OUT=eval_out/exp066_search
LOG="$OUT/chain.log"
source scripts/setup_gpu.sh
export PYTHONUNBUFFERED=1 WORLD_BOUNDARY_REMOVAL=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.20
export VALUE_EVAL_BATCH=128 POLICY_BATCH_CAP=96 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[exp066v2] start $(date -u +%FT%TZ)" | tee -a "$LOG"
pids=()
for k in $(seq 0 19); do
  env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=$((k % 4)) \
    python3 scripts/exp066_search_validation.py \
    --phase tree --shard-id $k --num-shards 20 --state-subset 30 --inner-pool 8 --out-cap 8 --out-dir "$OUT" \
    >> "$OUT/tree_shard$k.log" 2>&1 &
  pids+=($!); sleep 5
done
wait "${pids[@]}" || true
echo "[exp066v2] tree done" | tee -a "$LOG"
pids=()
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k JAX_PLATFORMS=cuda python3 scripts/exp066_search_validation.py \
    --phase adjudicate --shard-id $k --num-shards 4 --state-subset 30 --out-dir "$OUT" \
    >> "$OUT/adj_shard$k.log" 2>&1 &
  pids+=($!); sleep 20
done
wait "${pids[@]}" || true
echo "[exp066v2] adjudicate done" | tee -a "$LOG"
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu python3 scripts/exp066_search_validation.py \
  --phase aggregate --out-dir "$OUT" | tee -a "$LOG" | tee -a experiments_log.md
echo "EXP066_DONE $(date -u +%FT%TZ)" | tee -a "$LOG"
