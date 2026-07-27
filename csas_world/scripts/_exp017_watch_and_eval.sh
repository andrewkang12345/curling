#!/usr/bin/env bash
# Wait for EXP-017 (full 1->10 deployable) to finish, then run the final 4-GPU NOISY
# high-N eval (champion vs prior across all horizons incl FGZ h>=8) via _eval_parallel.py.
set -u
cd /mnt/data/curling2/csas_world
LOG=checkpoints/csas_world/exp_017_deploy_robust/exp017.log
SUMMARY=checkpoints/csas_world/exp_017_deploy_robust/curriculum_summary.json

# 1) wait for CURRICULUM_EXIT (up to ~12h)
for i in $(seq 1 360); do
  grep -qa "CURRICULUM_EXIT" "$LOG" 2>/dev/null && break
  pgrep -f "run_curriculum.py --config configs/exp_017_deploy" >/dev/null 2>&1 || { sleep 20; grep -qa CURRICULUM_EXIT "$LOG" || { echo ">> EXP-017 exited WITHOUT CURRICULUM_EXIT"; break; }; }
  sleep 120
done

echo "=== EXP-017 per-stage winrates (NOISY in-loop, vs prev champion) ==="
grep -aoaE "curriculum h[0-9]+ r[0-9]+\] winrate_vs_prev_best=[0-9.]+ \([^)]*\)|converged|CURRICULUM_EXIT=[0-9]" "$LOG" 2>/dev/null

# 2) pick the deployable champion = best_ckpt of the highest completed stage
CHAMP=$(python3 -c "
import json
d=json.load(open('$SUMMARY'))
ks=sorted(d.keys())
for k in reversed(ks):
    c=(d[k] or {}).get('best_ckpt')
    if c: print(c); break
" 2>/dev/null)
echo "=== deployable champion: $CHAMP ==="
if [ -z "$CHAMP" ] || [ ! -f "$CHAMP" ]; then echo ">> champion ckpt missing; skipping eval"; echo "WATCH_EVAL_DONE"; exit 0; fi

# 3) final NOISY high-N eval vs prior across ALL horizons (1..10), 4-GPU fan-out
echo "=== FINAL EVAL: EXP-017 champion vs PRIOR (NOISY, N=700, h=1..10) ==="
python3 scripts/_eval_parallel.py --champion "$CHAMP" --vs prior \
  --N 700 --horizons 1,2,3,4,5,6,7,8,9,10 --gpus 0,1,2,3 --noisy \
  --out-dir eval_out/exp017_vs_prior
echo "WATCH_EVAL_DONE"
