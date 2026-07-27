#!/usr/bin/env bash
# Wait for EXP-020 (value_from_mcts=true) to finish, then run a 1->10 NOISY eval vs the human prior
# at the same protocol as EXP-019 (proper full val pool, both orders, 4-GPU within-horizon shards).
set -u
cd /mnt/data/curling2/csas_world
LOG=checkpoints/csas_world/exp_020_consolidate_valuemcts/exp020.log
CK=checkpoints/csas_world/exp_020_consolidate_valuemcts/last.pt

for i in $(seq 1 240); do  # up to ~8h
  grep -qa "CONSOLIDATE_EXIT" "$LOG" 2>/dev/null && break
  sleep 60
done
echo "=== EXP-020 training tail ==="
grep -aoaE "\[e[0-9]+ s[0-9]+\] policy_distill=[0-9.]+[^|]*|val_policy_nll=[0-9.na]+|val_value_mse=[0-9.]+|CONSOLIDATE_EXIT=[0-9]" "$LOG" 2>/dev/null | tail -5

[ -f "$CK" ] || { echo ">> $CK missing; abort"; echo "EXP020_EVAL_DONE"; exit 0; }
echo "=== EXP-020 1->10 NOISY eval vs prior (N=400 per horizon, full pool used) ==="
python3 scripts/_eval_parallel.py --champion "$CK" --vs prior \
  --N 400 --horizons 1,2,3,4,5,6,7,8,9,10 --gpus 0,1,2,3 --shards 4 --noisy \
  --out-dir eval_out/proper/exp020_valuemcts_vs_prior
echo "EXP020_EVAL_DONE"
