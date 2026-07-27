#!/usr/bin/env bash
# Wait for the EXP-017 RESUME (h8->10 from h07/r0, OOM fix) to finish, then run the final
# 4-GPU NOISY high-N eval of the new (cleanly-trained-through-10) champion vs prior.
set -u
cd /mnt/data/curling2/csas_world
LOG=checkpoints/csas_world/exp_017_deploy_robust/exp017_resume.log
SUMMARY=checkpoints/csas_world/exp_017_deploy_robust/curriculum_summary.json

# 1) wait for CURRICULUM_EXIT (up to ~8h)
for i in $(seq 1 240); do
  grep -qa "CURRICULUM_EXIT" "$LOG" 2>/dev/null && break
  pgrep -f "run_curriculum.py --config configs/exp_017_deploy" >/dev/null 2>&1 || { sleep 20; grep -qa CURRICULUM_EXIT "$LOG" || { echo ">> resume exited WITHOUT CURRICULUM_EXIT"; break; }; }
  sleep 120
done

echo "=== EXP-017 resume per-stage winrates (NOISY in-loop) ==="
grep -aoaE "curriculum h[0-9]+ r[0-9]+\] winrate_vs_prev_best=[0-9.]+ \([^)]*\)|converged|CURRICULUM_EXIT=[0-9]" "$LOG" 2>/dev/null
echo "=== check for recurrence of the OOM/nan ==="
grep -aoaE "val_policy_nll=nan|out of memory|eval skipped" "$LOG" 2>/dev/null | tail -3 || echo "(none — OOM fix held)"

# 2) deployable champion = best_ckpt of the highest completed stage (ideally h10)
CHAMP=$(python3 -c "
import json
d=json.load(open('$SUMMARY'))
for k in sorted(d.keys(), reverse=True):
    c=(d[k] or {}).get('best_ckpt')
    if c: print(c); break
" 2>/dev/null)
echo "=== deployable champion (post-resume): $CHAMP ==="
if [ -z "$CHAMP" ] || [ ! -f "$CHAMP" ]; then echo ">> champion ckpt missing; skipping eval"; echo "RESUME_EVAL_DONE"; exit 0; fi

# 3) final NOISY high-N eval vs prior across all horizons (N=350 bounds the h10 long-pole)
echo "=== FINAL EVAL: post-resume champion vs PRIOR (NOISY, N=350, h=1..10) ==="
python3 scripts/_eval_parallel.py --champion "$CHAMP" --vs prior \
  --N 350 --horizons 1,2,3,4,5,6,7,8,9,10 --gpus 0,1,2,3 --noisy \
  --out-dir eval_out/exp017_resume_vs_prior
echo "RESUME_EVAL_DONE"
