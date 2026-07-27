#!/usr/bin/env bash
# Autonomous chain: wait for EXP-008 (exp_a_tuned) to finish, then GPU-smoke the EXP-009
# reward path, and only if the smoke exits 0 with a step_reward metric, run EXP-009 full.
# Run in the background; it blocks on each curriculum (the launcher runs them foreground).
set -uo pipefail
cd /mnt/data/curling2/csas_world
ALOG=checkpoints/csas_world/exp_a_tuned/exp_a.log

echo "[chain] waiting for EXP-008 to finish..."
for i in $(seq 1 600); do          # up to ~10h
  grep -qa "CURRICULUM_EXIT" "$ALOG" 2>/dev/null && break
  pgrep -f "run_curriculum.py --config configs/exp_a_tuned" >/dev/null 2>&1 || { sleep 15; grep -qa "CURRICULUM_EXIT" "$ALOG" 2>/dev/null || { echo "[chain] EXP-008 process gone (no EXIT)"; }; break; }
  sleep 60
done
echo "[chain] EXP-008 winrates:"; grep -aoaE "curriculum h[0-9]+ r[0-9]+\] winrate_vs_prev_best=[0-9.]+ \([^)]*\)" "$ALOG" 2>/dev/null
sleep 10

echo "[chain] EXP-009 GPU smoke..."
rm -rf checkpoints/csas_world/exp_b_smoke; mkdir -p checkpoints/csas_world/exp_b_smoke
bash scripts/_run_curriculum_fullsheet.sh configs/exp_b_smoke.yaml \
  checkpoints/csas_world/exp_b_smoke artifacts/replay/sim_none \
  > checkpoints/csas_world/exp_b_smoke/smoke.log 2>&1
SLOG=checkpoints/csas_world/exp_b_smoke/smoke.log
if grep -qa "CURRICULUM_EXIT=0" "$SLOG" && grep -qa "step_reward=" "$SLOG"; then
  echo "[chain] EXP-009 smoke PASSED (reward head trained); launching full EXP-009."
  rm -rf checkpoints/csas_world/exp_b_reward; mkdir -p checkpoints/csas_world/exp_b_reward
  bash scripts/_run_curriculum_fullsheet.sh configs/exp_b_reward_4h.yaml \
    checkpoints/csas_world/exp_b_reward artifacts/replay/sim_none \
    > checkpoints/csas_world/exp_b_reward/exp_b.log 2>&1
  echo "[chain] EXP-009 winrates:"; grep -aoaE "curriculum h[0-9]+ r[0-9]+\] winrate_vs_prev_best=[0-9.]+ \([^)]*\)" checkpoints/csas_world/exp_b_reward/exp_b.log 2>/dev/null
else
  echo "[chain] EXP-009 SMOKE FAILED — not launching full run. Tail:"
  grep -aviE 'register|ffi|not compatible|xla|warn|plugin|jax_plugins|discover|policy_bc=|\[e[0-9]' "$SLOG" 2>/dev/null | tail -15
fi
echo "[chain] DONE"
