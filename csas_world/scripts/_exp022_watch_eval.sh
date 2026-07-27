#!/usr/bin/env bash
# Wait for EXP-022 (EXP-019 recipe + held-out MCTS val + early-stop by val_total_mcts) to finish,
# then eval BOTH last.pt and best.pt vs prior. Compares to EXP-019 (0.552) and EXP-021 best (0.562)
# to isolate whether EXP-019 itself benefits from per-loss MCTS val + early stop.
set -u
cd /mnt/data/curling2/csas_world
LOG=checkpoints/csas_world/exp_022_exp019_earlystop/exp022.log
DIR=checkpoints/csas_world/exp_022_exp019_earlystop

for i in $(seq 1 240); do
  grep -qa "CONSOLIDATE_EXIT" "$LOG" 2>/dev/null && break
  sleep 60
done
echo "=== EXP-022 per-epoch val curves ==="
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
    --out-dir "eval_out/proper/exp022_${tag}_vs_prior"
done
echo "EXP022_DONE"
