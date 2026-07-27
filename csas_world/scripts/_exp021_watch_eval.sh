#!/usr/bin/env bash
# Wait for EXP-021 (value_from_mcts=true + held-out MCTS val + early-stop ckpt by val_value_mse_mcts)
# to finish, then eval BOTH last.pt (most-trained) and best.pt (early-stop) vs prior in NOISY 1->10.
set -u
cd /mnt/data/curling2/csas_world
LOG=checkpoints/csas_world/exp_021_valuemcts_earlystop/exp021.log
DIR=checkpoints/csas_world/exp_021_valuemcts_earlystop

for i in $(seq 1 240); do
  grep -qa "CONSOLIDATE_EXIT" "$LOG" 2>/dev/null && break
  sleep 60
done
echo "=== EXP-021 per-epoch val (incl new val_*_mcts) ==="
grep -aoaE "\[epoch [0-9]+\][^|]+val_[^|]*" "$LOG" 2>/dev/null
echo
echo "=== best.pt epoch (early-stop ckpt by val_value_mse_mcts) ==="
python3 -c "
import json
d=json.load(open('$DIR/results.json'))
print(' final last.pt metrics:', d.get('metrics', {}))
print(' best_val (val_value_mse_mcts):', d.get('best_val'))
" 2>/dev/null

for tag in last best; do
  CK="$DIR/${tag}.pt"
  [ -f "$CK" ] || continue
  echo "=== EVAL $tag.pt vs PRIOR (NOISY N=400) ==="
  python3 scripts/_eval_parallel.py --champion "$CK" --vs prior \
    --N 400 --horizons 1,2,3,4,5,6,7,8,9,10 --gpus 0,1,2,3 --shards 4 --noisy \
    --out-dir "eval_out/proper/exp021_${tag}_vs_prior"
done
echo "EXP021_DONE"
