#!/usr/bin/env bash
# Wait for EXP-018 consolidation to finish, then NOISY 1->10 eval (pre-placed h10, 4-GPU sharded) of the
# consolidated model vs prior. Eval BOTH last.pt (most-trained) and best.pt (lowest val-nll) and compare
# the hammer-neutral MEAN-of-pairs to h07/r0 (0.530) and h10/r1 (0.515).
set -u
cd /mnt/data/curling2/csas_world
LOG=checkpoints/csas_world/exp_018_consolidate/consolidate.log
DIR=checkpoints/csas_world/exp_018_consolidate

for i in $(seq 1 180); do   # up to ~6h
  grep -qa "CONSOLIDATE_EXIT" "$LOG" 2>/dev/null && break
  pgrep -f "run_consolidate.py" >/dev/null 2>&1 || { sleep 20; grep -qa CONSOLIDATE_EXIT "$LOG" || { echo ">> consolidation exited WITHOUT CONSOLIDATE_EXIT"; break; }; }
  sleep 120
done
echo "=== consolidation training tail ==="
grep -aoaE "\[e[0-9]+ s[0-9]+\] policy_distill=[0-9.]+[^|]*|val_policy_nll=[0-9.na]+|CONSOLIDATE_EXIT=[0-9]" "$LOG" 2>/dev/null | tail -4
grep -aoaE "val_policy_nll=nan|out of memory" "$LOG" 2>/dev/null | tail -2 || true

for tag in last best; do
  CH="$DIR/${tag}.pt"
  [ -f "$CH" ] || { echo ">> $CH missing, skip"; continue; }
  echo "=== EVAL ${tag}.pt vs PRIOR (NOISY N=700, h=1..10, pre-placed h10, 4-GPU sharded) ==="
  python3 scripts/_eval_parallel.py --champion "$CH" --vs prior \
    --N 700 --horizons 1,2,3,4,5,6,7,8,9,10 --gpus 0,1,2,3 --shards 4 --noisy \
    --out-dir eval_out/exp018_${tag}_vs_prior
done
echo "=== COMPARISON: consolidation vs baselines (MEAN-of-pairs) ==="
echo "  h07/r0 (per-stage champ, no real h10): 0.530"
echo "  h10/r1 (sequential 1->10, forgot h7-10): 0.515"
echo "  (EXP-018 MEAN-of-pairs printed above per checkpoint)"
echo "EXP018_EVAL_DONE"
