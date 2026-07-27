#!/usr/bin/env bash
# Chained launcher: wait for EXP-023's eval to finish, then run EXP-024 + eval.
# EXP-024 = fine-tune EXP-022's best.pt (= EXP-019-equivalent) with value_from_mcts=true,
#           held-out MCTS val + early-stop by val_value_mse_mcts.
set -u
cd /mnt/data/curling2/csas_world
EXP023_LOG=checkpoints/csas_world/exp_023_exp019_longtrain/exp023.log
EXP024_DIR=checkpoints/csas_world/exp_024_finetune_valuemcts
WARM=checkpoints/csas_world/exp_022_exp019_earlystop/best.pt

# --- 1) wait for EXP-023 chain (training + eval) to finish ---
# EXP-023 watcher emits EXP023_DONE at /tmp/_exp023_watch_eval.out (or whichever log). We poll the
# in-house watcher's bash log indirectly via process status: the watcher proc holds two GPUs while
# evals are sharded; we just wait until no _eval_parallel.py running on our exp_023 paths.
echo "[chain] waiting for EXP-023 + eval to finish..."
for i in $(seq 1 240); do
  pgrep -af "run_consolidate.py.*exp_023_exp019_longtrain\|_eval_parallel.py.*exp023_" >/dev/null 2>&1 || break
  sleep 60
done
sleep 30                     # let any GPU memory release fully
echo "[chain] EXP-023 chain done. Launching EXP-024."

# --- 2) launch EXP-024 training (DDP across 4 GPUs) ---
mkdir -p "$EXP024_DIR"
unset LD_LIBRARY_PATH
export PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing
export GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none
python3 scripts/run_consolidate.py \
  --config configs/exp_024_finetune_valuemcts.yaml \
  --union artifacts/replay/exp021_train_mcts \
  --mcts-val artifacts/replay/exp021_val_mcts \
  --init "$WARM" \
  --out "$EXP024_DIR" \
  >> "$EXP024_DIR/exp024.log" 2>&1
echo "[chain] EXP-024 training CONSOLIDATE_EXIT=$? -> $EXP024_DIR"
echo "=== EXP-024 per-epoch val curves ==="
grep -aoaE "\[epoch [0-9]+\][^|]+val_[^|]*" "$EXP024_DIR/exp024.log" 2>/dev/null
echo
echo "=== best.pt (early-stop ckpt by val_value_mse_mcts) ==="
python3 -c "
import json
d=json.load(open('$EXP024_DIR/results.json'))
print(' final last.pt metrics:', d.get('metrics', {}))
print(' best_val (val_value_mse_mcts):', d.get('best_val'))
" 2>/dev/null

# --- 3) eval both last.pt and best.pt vs prior ---
for tag in last best; do
  CK="$EXP024_DIR/${tag}.pt"
  [ -f "$CK" ] || continue
  echo "=== EVAL $tag.pt vs PRIOR (NOISY N=400) ==="
  python3 scripts/_eval_parallel.py --champion "$CK" --vs prior \
    --N 400 --horizons 1,2,3,4,5,6,7,8,9,10 --gpus 0,1,2,3 --shards 4 --noisy \
    --out-dir "eval_out/proper/exp024_${tag}_vs_prior"
done
echo "EXP024_DONE"
