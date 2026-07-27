#!/usr/bin/env bash
# Wait until the az_v5 orchestrator releases its PID lock (i.e. finishes or dies),
# then launch az_v6. This is a thin sequencer so the operator can fire-and-forget
# the next experiment without manually polling.
set -uo pipefail
cd /mnt/data/curling2/csas_world

V5_LOCK=checkpoints/csas_world/az_v5_novaluemcts/launcher.pid
V6_LOG=checkpoints/csas_world/az_v6_2ply_unfrozen/run.log
mkdir -p "$(dirname "$V6_LOG")"

echo "[chain] waiting for az_v5 lock $V5_LOCK to release..." | tee -a "$V6_LOG"
while true; do
  if [ ! -f "$V5_LOCK" ]; then
    echo "[chain] az_v5 lock gone -> proceed" | tee -a "$V6_LOG"
    break
  fi
  pid=$(cat "$V5_LOCK" 2>/dev/null || echo "")
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
    echo "[chain] az_v5 lock-holder pid $pid no longer alive -> proceed" | tee -a "$V6_LOG"
    break
  fi
  sleep 30
done

# Tiny extra wait so any in-flight GPU memory release completes cleanly.
sleep 15
echo "[chain] launching az_v6 at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$V6_LOG"
bash scripts/_az_v6_2ply_unfrozen_launch.sh
echo "[chain] az_v6 launcher returned at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$V6_LOG"
