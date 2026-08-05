#!/usr/bin/env bash
# EXP-066 chain: wait for EXP-065 to finish, then run the search-validation
# benchmark: states -> ref+flat (GPU-JAX phases) -> tree (CPU-JAX, 4 shards) -> aggregate.
set -uo pipefail
cd /mnt/data/curling2/csas_world
OUT=eval_out/exp066_search
LOG="$OUT/chain.log"
mkdir -p "$OUT"
while ! grep -aq "EXP065_DONE" artifacts/replay/mcts/az_v25_br/exp065.log 2>/dev/null; do sleep 600; done
echo "[exp066] start $(date -u +%FT%TZ)" | tee -a "$LOG"
source scripts/setup_gpu.sh   # GPU-JAX env for states/ref/flat phases
export PYTHONUNBUFFERED=1 WORLD_BOUNDARY_REMOVAL=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
export VALUE_EVAL_BATCH=128 POLICY_BATCH_CAP=96 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

[ -f "$OUT/states.npz" ] || python3 scripts/exp066_search_validation.py --phase states \
  --out-dir "$OUT" >> "$LOG" 2>&1
echo "[exp066] states done" | tee -a "$LOG"

pids=()
for k in 0 1; do
  JAX_PLATFORMS=cuda python3 scripts/exp066_search_validation.py --phase ref \
    --shard-id $k --num-shards 2 --out-dir "$OUT" >> "$OUT/ref_shard$k.log" 2>&1 &
  pids+=($!); sleep 30
done
wait "${pids[@]}" || true
echo "[exp066] ref done" | tee -a "$LOG"

JAX_PLATFORMS=cuda python3 scripts/exp066_search_validation.py --phase flat \
  --shard-id 0 --num-shards 1 --out-dir "$OUT" >> "$OUT/flat.log" 2>&1
echo "[exp066] flat done" | tee -a "$LOG"

pids=()
for k in 0 1 2 3; do
  env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu python3 scripts/exp066_search_validation.py \
    --phase tree --shard-id $k --num-shards 4 --out-dir "$OUT" \
    >> "$OUT/tree_shard$k.log" 2>&1 &
  pids+=($!); sleep 20
done
wait "${pids[@]}" || true
echo "[exp066] tree done" | tee -a "$LOG"

env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu python3 scripts/exp066_search_validation.py \
  --phase aggregate --out-dir "$OUT" | tee -a "$LOG" | tee -a experiments_log.md
echo "EXP066_DONE $(date -u +%FT%TZ)" | tee -a "$LOG"
