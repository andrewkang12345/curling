#!/usr/bin/env bash
# Wait for the EXP-017 h10-extension (resume from h09/r1, --start 10, pre-placed h10) to finish,
# then run the final 4-GPU NOISY eval of the h10-trained model vs prior across the WHOLE end (1..10,
# pre-placed h10), reporting the per-horizon hammer split + odd/even pair averages.
set -u
cd /mnt/data/curling2/csas_world
LOG=checkpoints/csas_world/exp_017_deploy_robust/exp017_h10.log
DIR=checkpoints/csas_world/exp_017_deploy_robust

for i in $(seq 1 180); do   # up to ~6h
  grep -qa "CURRICULUM_EXIT" "$LOG" 2>/dev/null && break
  pgrep -f "run_curriculum.py --config configs/exp_017_deploy" >/dev/null 2>&1 || { sleep 20; grep -qa CURRICULUM_EXIT "$LOG" || { echo ">> h10-extension exited WITHOUT CURRICULUM_EXIT"; break; }; }
  sleep 120
done

echo "=== h10-extension per-stage winrate + divergence check (the fix should hold) ==="
grep -aoaE "curriculum h[0-9]+ r[0-9]+\] winrate_vs_prev_best=[0-9.]+ \([^)]*\)|converged|CURRICULUM_EXIT=[0-9]" "$LOG" 2>/dev/null
echo "OOM/nan recurrence:"; grep -aoaE "val_policy_nll=nan|out of memory|eval skipped" "$LOG" 2>/dev/null | tail -3 || echo "  (none — fix held)"
echo "h10 records written:"; grep -aoaE "wrote [0-9]+ mcts records" "$LOG" 2>/dev/null | sort | uniq -c

# the full-end model = the h10-trained checkpoint
CHAMP="$DIR/h10/r1/model.pt"; [ -f "$CHAMP" ] || CHAMP="$DIR/h10/r0/model.pt"
echo "=== full-end model (h10-trained): $CHAMP ==="
[ -f "$CHAMP" ] || { echo ">> h10 ckpt missing; skipping eval"; echo "H10_EVAL_DONE"; exit 0; }

echo "=== FINAL EVAL: h10-trained model vs PRIOR (NOISY, N=700, h=1..10, pre-placed h10, 4-GPU sharded) ==="
python3 scripts/_eval_parallel.py --champion "$CHAMP" --vs prior \
  --N 700 --horizons 1,2,3,4,5,6,7,8,9,10 --gpus 0,1,2,3 --shards 4 --noisy \
  --out-dir eval_out/exp017_h10_vs_prior
echo "H10_EVAL_DONE"
