#!/usr/bin/env bash
# Wait for EXP-023 (EXP-022 recipe at 40 epochs) to finish, then eval BOTH last.pt and best.pt
# vs prior. Tests whether EXP-019 had headroom past 20 epochs.
set -u
cd /mnt/data/curling2/csas_world
LOG=checkpoints/csas_world/exp_023_exp019_longtrain/exp023.log
DIR=checkpoints/csas_world/exp_023_exp019_longtrain

for i in $(seq 1 360); do  # up to ~6 h
  grep -qa "CONSOLIDATE_EXIT" "$LOG" 2>/dev/null && break
  sleep 60
done
echo "=== EXP-023 per-epoch val curves ==="
grep -aoaE "\[epoch [0-9]+\][^|]+val_[^|]*" "$LOG" 2>/dev/null
echo
echo "=== best.pt epoch (early-stop ckpt by val_total_mcts) ==="
python3 -c "
import json
d=json.load(open('$DIR/results.json'))
print(' final last.pt metrics:', d.get('metrics', {}))
print(' best_val (val_total_mcts):', d.get('best_val'))
" 2>/dev/null

for tag in last best; do
  CK="$DIR/${tag}.pt"
  [ -f "$CK" ] || continue
  echo "=== EVAL $tag.pt vs PRIOR (NOISY N=400) ==="
  python3 scripts/_eval_parallel.py --champion "$CK" --vs prior \
    --N 400 --horizons 1,2,3,4,5,6,7,8,9,10 --gpus 0,1,2,3 --shards 4 --noisy \
    --out-dir "eval_out/proper/exp023_${tag}_vs_prior"
done
echo "EXP023_DONE"
