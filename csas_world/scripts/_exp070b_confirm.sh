#!/usr/bin/env bash
# EXP-070b: full-N re-measurement of the three cycle-critical cells before any
# champion change (screening N=150 gave ~0.5-SE cycle edges; this is N=400, k=4).
set -uo pipefail
cd /mnt/data/curling2/csas_world
OUT=eval_out/exp070_meta/confirm
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1 WORLD_BOUNDARY_REMOVAL=1
ENV="PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu VALUE_EVAL_BATCH=64 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing GNN_NODE_FEATURE_MODE=none \
GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none"
C=checkpoints/csas_world
run() {  # tag A B gpus
  local O="$OUT/$1"; [ -f "$O/summary.json" ] && return 0
  mkdir -p "$O"
  env -u LD_LIBRARY_PATH $ENV python3 scripts/_eval_parallel.py --champion "$2" --vs "$3" \
    --N 400 --horizons 1,2,3,4,5,6,7,8,9,10 --gpus $4 --shards 2 --noisy --sel-noise 4 \
    --out-dir "$O" >> "$O/run.log" 2>&1
}
run v25_vs_v26 "$C/az_v25_br/best.pt" "$C/az_v26_br2/best.pt" 0,0 &
run v19_vs_v26 "$C/az_v19_newrules/best.pt" "$C/az_v26_br2/best.pt" 1,1 &
run v14d_vs_v19 "$C/az_v14d/best.pt" "$C/az_v19_newrules/best.pt" 2,2 &
run v14d_vs_v26 "$C/az_v14d/best.pt" "$C/az_v26_br2/best.pt" 3,3 &
wait
python3 - <<'PYEOF' | tee -a experiments_log.md
import json, glob, math
import numpy as np
print("\n**EXP-070b cycle-edge confirmation (N=400, k=4):**\n")
for tag in ("v25_vs_v26", "v19_vs_v26", "v14d_vs_v19", "v14d_vs_v26"):
    ms, W, N = [], 0.0, 0.0
    for f in glob.glob(f"eval_out/exp070_meta/confirm/{tag}/*__h*__s*.json"):
        j = json.load(open(f))
        for k, v in j.items():
            if k.startswith("h") and isinstance(v, dict):
                ms.append(v["mean_margin"]); W += v["winrate"]*v["n_ends"]; N += v["n_ends"]
    if ms:
        mu = np.mean(ms); se = np.std(ms, ddof=1)/math.sqrt(len(ms))
        print(f"- {tag}: dScore {mu:+.4f} ± {se:.4f}/end (t={mu/se:+.2f}), winrate {W/N:.4f}, n={int(N)}")
PYEOF
echo "EXP070B_DONE $(date -u +%FT%TZ)"
