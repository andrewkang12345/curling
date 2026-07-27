#!/bin/bash
set -uo pipefail
cd /mnt/data/curling2/csas_world
source scripts/setup_gpu.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
python3 - <<'PY' 2>&1 | grep -avE 'warnings|register_custom|xla_cuda12|discover_pjrt|ffi_registrations|^\s*\^|plugin'
import json, torch, world
from world.eval.head_to_head import WorldPlayer, CsasPlayer, build_h2h_roots, head_to_head
from world.search.noise import make_noise
dev = torch.device("cuda:0")
NCFG="/mnt/data/curling2/csas_v3/configs/noise/v1_bowling.json"
SEL_NS=8; NCAND=48; ROOTS=30; HZS=[4,8]
env_noise = make_noise(NCFG, 999)
def wp(ck,nm,sd): return WorldPlayer(ck,dev,n_candidates=NCAND,name=nm,noise=make_noise(NCFG,sd),sel_noise_samples=SEL_NS)
def cp(p,v,nm,sd): return CsasPlayer(p,v,dev,n_candidates=NCAND,name=nm,noise=make_noise(NCFG,sd),sel_noise_samples=SEL_NS)
A   = wp("checkpoints/csas_world/anchor_noisy/model.pt","anchor_noisy",1)
V3  = wp("checkpoints/csas_world/anchor_v3/model.pt","anchor_v3",2)
MC  = cp("/mnt/data/curling2/csas_v3/checkpoints/policy/mcts_horizon/h10/model.pt",
         "/mnt/data/curling2/csas_v3/checkpoints/value/holdout0/model.pt","mcts_h10",3)
out={}
for label,B in [("anchor_noisy_vs_anchor_v3",V3),("anchor_noisy_vs_mcts_h10",MC)]:
    per={}
    for h in HZS:
        roots=build_h2h_roots("/mnt/data/curling2/csas_v3",h,ROOTS,split="val",seed=h)
        r=head_to_head(A,B,roots,env_noise=env_noise,realize_noise=True)
        per[f"h{h:02d}"]=r
        print(f"NOISY {label} h{h:02d}: winrate={r['winrate']:.3f} (o0={r['winrate_order0']:.2f} o1={r['winrate_order1']:.2f}) margin={r['mean_margin']:+.2f} n={r['n_ends']}",flush=True)
    out[label]=per
json.dump(out,open("artifacts/metrics/noisy_robust_h2h.json","w"),indent=2)
print("NOISY_H2H_DONE")
PY
