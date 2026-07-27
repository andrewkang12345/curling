#!/usr/bin/env bash
# EXP-052: retrain under REAL-CURLING boundary-removal semantics and measure strength.
#
# WORLD_BOUNDARY_REMOVAL=1 (env_bridge): stones whose final center is past the back
# line, or that touch a side board, are removed from play -- so the mixed-doubles
# early-takeout legality rule binds the way real rules intend.
#
# Chain (fully resumable; stop collection early: touch $OUT/STOP):
#   A. collect champion self-play under NEW rules (sig-gated screen_tree, exp_037
#      operator; az_v14d exported policy) -- N workers x G games per round to TARGET.
#   B. train/val split (shard N-1 = val), fine-tune az_v14d L8 (exp_052 = exp_048 recipe).
#   C. eval: 3 draws h2h vs az_v14d UNDER NEW RULES + 1 draw under OLD rules
#      (regression check). dScore primary.
#   D. append per-draw aggregates to experiments_log.md (raw; verdict finalised by hand).
#
#   usage: _exp052_newrules_loop.sh [N_workers=6] [games_per_worker_per_round=16] [target_games=600]
set -uo pipefail
cd /mnt/data/curling2/csas_world
N=${1:-6}; G=${2:-16}; TARGET=${3:-600}
OUT=artifacts/replay/mcts/az_v19_newrules
WORK=checkpoints/csas_world/az_v19_newrules
INC=checkpoints/csas_world/az_v14d/best.pt
POL=checkpoints/csas_world/az_v15_L8/incumbent0_policy_csas.pt
VAL=/mnt/data/curling2/csas_v3/checkpoints/value/holdout0/model.pt
LOG="$OUT/exp052.log"
LOCK="$OUT/launcher.pid"
mkdir -p "$OUT" "$WORK"
if [ -f "$LOCK" ]; then p=$(cat "$LOCK"); [ -n "$p" ] && kill -0 "$p" 2>/dev/null && { echo "REFUSING: $p alive" | tee -a "$LOG"; exit 1; }; fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
source scripts/setup_gpu.sh
export PYTHONUNBUFFERED=1
ENVV="WORLD_BOUNDARY_REMOVAL=1 PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu \
CUDA_VISIBLE_DEVICES=0 POLICY_BATCH_CAP=96 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing \
GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none"
echo "[exp052] N=$N G=$G TARGET=$TARGET; start $(date -u +%FT%TZ)" | tee -a "$LOG"

total_games() { ls "$OUT"/r*_shard*.npz 2>/dev/null | wc -l | awk -v g=$G '{print $1*g}'; }

# ---------------- A. collection under NEW rules ----------------
round=1
while true; do
  [ -f "$OUT/STOP" ] && { echo "[exp052] STOP file found" | tee -a "$LOG"; break; }
  done_games=$(total_games)
  [ "$done_games" -ge "$TARGET" ] && { echo "[exp052] target reached: $done_games games" | tee -a "$LOG"; break; }
  R=$(printf "%04d" $round)
  if [ ! -f "$OUT/r${R}_shard0.npz" ]; then
    pids=()
    for k in $(seq 0 $((N-1))); do
      [ "$k" -gt 0 ] && sleep 90   # stagger the JIT-compilation window (LLVM OOM fix)
      unset LD_LIBRARY_PATH
      env $ENVV timeout 21600 python3 -m world.search.selfplay \
        --config configs/exp_037_sig_screen_tree.yaml \
        --games "$G" --num-shards "$N" --shard-id "$k" --split train \
        --seed $((520000 + round*100 + k)) --scorer screen_tree \
        --policy "$POL" --value "$VAL" \
        --out "$OUT/r${R}_shard$k.npz" --device cuda:0 \
        > "$OUT/r${R}_shard$k.log" 2>&1 &
      pids+=($!)
    done
    wait "${pids[@]}" || true
  fi
  python3 - <<PYEOF 2>/dev/null | tee -a "$LOG"
import numpy as np, glob
tot = act = 0
for f in glob.glob("$OUT/r${R}_shard*.npz"):
    d = np.load(f, allow_pickle=True); m = d["dist_mask"]; tot += len(m); act += int((m>0).sum())
print(f"[exp052] round $R done: +{tot} records ({act} sig, {act/max(tot,1):.1%}); cumulative games: $(total_games) / $TARGET")
PYEOF
  round=$((round+1))
done

# ---------------- B. split + train ----------------
T=artifacts/replay/az_v19_train; V=artifacts/replay/az_v19_val
rm -rf "$T" "$V"; mkdir -p "$T" "$V"
LASTK=$((N-1))
for f in "$OUT"/r*_shard*.npz; do
  b=$(basename "$f"); k=${b##*shard}; k=${k%.npz}
  if [ "$k" = "$LASTK" ]; then ln -s "$(readlink -f "$f")" "$V/$b"; else ln -s "$(readlink -f "$f")" "$T/$b"; fi
done
echo "[exp052] split: $(ls $T | wc -l) train / $(ls $V | wc -l) val shards" | tee -a "$LOG"

if [ ! -f "$WORK/best.pt" ]; then
  echo "[exp052] TRAIN start $(date -u +%H:%M)" | tee -a "$LOG"
  unset LD_LIBRARY_PATH
  env $ENVV python3 scripts/run_consolidate.py \
    --config configs/exp_052_L8_newrules.yaml --union "$T" --mcts-val "$V" \
    --init "$INC" --out "$WORK" >> "$WORK/train.log" 2>&1
  echo "[exp052] train rc=$?" | tee -a "$LOG"
  grep -aE "early-stop" "$WORK/train.log" | tail -1 | tee -a "$LOG"
fi
CK="$WORK/best.pt"; [ -f "$CK" ] || CK="$WORK/model.pt"

# ---------------- C. eval ----------------
for d in 1 2 3; do
  O="eval_out/az_v19_newrules/newrules_run$d"
  [ -f "$O/summary.json" ] && continue
  mkdir -p "$O"; unset LD_LIBRARY_PATH
  env $ENVV python3 scripts/_eval_parallel.py --champion "$CK" --vs "$INC" \
    --N 400 --horizons 1,2,3,4,5,6,7,8,9,10 --gpus 0,0,0,0 --shards 4 --noisy \
    --out-dir "$O" >> "$WORK/eval.log" 2>&1
  echo "[exp052] NEW-rules draw $d done $(date -u +%H:%M)" | tee -a "$LOG"
done
O="eval_out/az_v19_newrules/oldrules_run1"
if [ ! -f "$O/summary.json" ]; then
  mkdir -p "$O"; unset LD_LIBRARY_PATH
  env ${ENVV/WORLD_BOUNDARY_REMOVAL=1 /} python3 scripts/_eval_parallel.py --champion "$CK" --vs "$INC" \
    --N 400 --horizons 1,2,3,4,5,6,7,8,9,10 --gpus 0,0,0,0 --shards 4 --noisy \
    --out-dir "$O" >> "$WORK/eval.log" 2>&1
  echo "[exp052] OLD-rules draw done $(date -u +%H:%M)" | tee -a "$LOG"
fi

# ---------------- D. aggregate + append raw results to the log ----------------
python3 - <<'PYEOF' | tee -a "$LOG" | tee -a experiments_log.md
import json, glob, math
print("\n**EXP-052 raw eval aggregates (auto-appended by _exp052_newrules_loop.sh):**\n")
for tag in ["newrules_run1", "newrules_run2", "newrules_run3", "oldrules_run1"]:
    files = glob.glob(f"eval_out/az_v19_newrules/{tag}/*__h*__s*.json")
    W = M = Nn = 0.0
    ms = []
    for f in files:
        d = json.load(open(f))
        for k, v in d.items():
            if k.startswith("h") and isinstance(v, dict):
                W += v["winrate"] * v["n_ends"]; M += v["mean_margin"] * v["n_ends"]
                Nn += v["n_ends"]; ms.append((v["mean_margin"], v["n_ends"]))
    if Nn == 0:
        print(f"- {tag}: (no results)"); continue
    w, m = W / Nn, M / Nn
    k = len(ms)
    sh_mean = sum(x for x, _ in ms) / k
    se_m = math.sqrt(sum((x - sh_mean) ** 2 for x, _ in ms) / max(k - 1, 1) / k)  # shard-level SE
    se_w = math.sqrt(max(w * (1 - w), 1e-9) / Nn)
    print(f"- {tag}: winrate {w:.4f} ± {se_w:.4f}, dScore {m:+.4f} ± {se_m:.4f} /end "
          f"(n={int(Nn)} ends, {k} shard-cells)")
PYEOF
echo "EXP052_DONE $(date -u +%FT%TZ)" | tee -a "$LOG"
