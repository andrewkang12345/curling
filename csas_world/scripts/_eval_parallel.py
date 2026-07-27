#!/usr/bin/env python3
"""Parallel high-N head-to-head eval across ALL GPUs, with WITHIN-horizon root sharding
and per-horizon hammer-split + odd/even pair reporting.

Every (opponent, horizon, shard) h2h is independent, so this fans them over the GPU pool
(one single-horizon, single-shard `_eval_highN.py` per job, pinned via CUDA_VISIBLE_DEVICES),
then merges shards per horizon. Within-horizon sharding keeps all GPUs busy even on the
big late-horizon pools (h10 had ~700 roots -> was a 5h single-GPU long-pole; now split G ways).

Reporting (per the agreed metrics):
  * per horizon, WITH HAMMER: model-as-to-move (order0) is the realistic situation -- model HAS
    the hammer at ODD horizons, does NOT at EVEN horizons. We print both the model-with-hammer and
    model-without-hammer winrate/dScore (order0/order1 mapped by parity).
  * odd+even PAIR averages (h01+h02, h03+h04, ...) of the model-as-to-move (order0) number -- breaks
    the hammer (one has-hammer horizon + one no-hammer horizon), equal-weighted = pure skill.

    python scripts/_eval_parallel.py --champion <A.pt> --vs prior [--vs <B.pt> ...] \
        --N 700 --horizons 1,2,3,4,5,6,7,8,9,10 --gpus 0,1,2,3 --shards 4 --noisy \
        --out-dir eval_out/<tag>
"""
import argparse
import json
import math
import os
import subprocess
import time

ap = argparse.ArgumentParser()
ap.add_argument("--champion", required=True)
ap.add_argument("--vs", action="append", required=True, help="repeatable: 'prior' or a WorldModel ckpt path")
ap.add_argument("--N", type=int, default=700)
ap.add_argument("--horizons", default="1,2,3,4,5,6,7,8,9,10")
ap.add_argument("--gpus", default="0,1,2,3")
ap.add_argument("--shards", type=int, default=0, help="root shards per horizon (0 => = #gpus)")
ap.add_argument("--noisy", action="store_true")
ap.add_argument("--out-dir", default="eval_out/parallel")
args = ap.parse_args()

HORIZONS = [int(x) for x in args.horizons.split(",")]
GPUS = [int(g) for g in args.gpus.split(",")]
SHARDS = args.shards if args.shards > 0 else len(GPUS)
os.makedirs(args.out_dir, exist_ok=True)
GNN_ENV = {
    "GNN_EDGE_SCALAR_MODE": "button_visible_plus_curl_arc_reach_with_outgoing",
    "GNN_NODE_FEATURE_MODE": "none",
    "GNN_RELEASE_NODE_MODE": "three_plus_takeout_boundary",
    "GNN_EDGE_PRUNE_MODE": "none",
}


def label(v):
    if v == "prior":
        return "prior"
    parts = v.rstrip("/").split("/")
    return parts[-4] if len(parts) >= 4 else os.path.basename(v).replace(".pt", "")


def outpath(v, h, s):
    return os.path.join(args.out_dir, f"{label(v)}__h{h:02d}__s{s}of{SHARDS}.json")


jobs = [(v, h, s) for v in args.vs for h in HORIZONS for s in range(SHARDS)]


def launch(job, gpu):
    v, h, s = job
    op = outpath(v, h, s)
    cmd = ["python3", "scripts/_eval_highN.py", "--champion", args.champion, "--vs", v,
           "--N", str(args.N), "--horizons", str(h), "--root-shard", f"{s}/{SHARDS}", "--out", op]
    if args.noisy:
        cmd.append("--noisy")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), **GNN_ENV)
    return subprocess.Popen(cmd, env=env, stdout=open(op + ".log", "w"), stderr=subprocess.STDOUT)


free = list(GPUS)
running = {}
qi = 0
t0 = time.time()
print(f"[parallel-eval] {len(jobs)} jobs ({len(args.vs)} opp x {len(HORIZONS)} horizons x {SHARDS} shards) "
      f"over GPUs {GPUS}; champion={args.champion}", flush=True)
while qi < len(jobs) or running:
    while free and qi < len(jobs):
        g = free.pop(0)
        p = launch(jobs[qi], g)
        running[p] = (g, jobs[qi])
        print(f"[launch] gpu{g}  vs={label(jobs[qi][0])} h{jobs[qi][1]:02d} shard{jobs[qi][2]}", flush=True)
        qi += 1
    time.sleep(5)
    for p in [p for p in running if p.poll() is not None]:
        g, job = running.pop(p)
        free.append(g)
        if p.returncode != 0:
            print(f"[FAIL]   gpu{g}  vs={label(job[0])} h{job[1]:02d} shard{job[2]}  rc={p.returncode}", flush=True)


# ---- merge shards per (opponent, horizon) ----
def merge_horizon(v, h):
    n0 = n1 = w0 = w1 = m0 = m1 = 0.0
    got = 0
    for s in range(SHARDS):
        try:
            d = json.load(open(outpath(v, h, s)))[f"h{h:02d}"]
        except Exception:
            continue
        got += 1
        no = d["n_ends"] / 2.0          # per-order count for this shard
        n0 += no; n1 += no
        w0 += d["winrate_order0"] * no; w1 += d["winrate_order1"] * no
        m0 += d["mean_margin_order0"] * no; m1 += d["mean_margin_order1"] * no
    if n0 == 0:
        return None
    return dict(got=got,
                wr_o0=w0 / n0, wr_o1=w1 / n1, m_o0=m0 / n0, m_o1=m1 / n1,
                wr_neutral=(w0 + w1) / (n0 + n1), m_neutral=(m0 + m1) / (n0 + n1),
                n_per_order=int(n0))


def se(p, n):
    return math.sqrt(max(p * (1 - p), 1e-9) / max(n, 1))


summary = {}
for v in args.vs:
    lbl = label(v)
    H = {h: merge_horizon(v, h) for h in HORIZONS}
    H = {h: r for h, r in H.items() if r}
    print(f"\n===== {os.path.basename(args.champion)} vs {lbl} =====")
    print("PER-HORIZON, WITH HAMMER (model as the to-move team):")
    print("  h  | model w/ HAMMER (wr, dScore) | model NO hammer (wr, dScore) | n/order")
    for h in sorted(H):
        r = H[h]
        # odd h: to-move (order0) has hammer; even h: order1 has hammer
        if h % 2 == 1:
            wh, mh, woh, moh = r["wr_o0"], r["m_o0"], r["wr_o1"], r["m_o1"]
        else:
            wh, mh, woh, moh = r["wr_o1"], r["m_o1"], r["wr_o0"], r["m_o0"]
        print(f"  h{h:02d} |  {wh:.3f} ± {se(wh, r['n_per_order']):.3f}, {mh:+.3f}      "
              f"|  {woh:.3f} ± {se(woh, r['n_per_order']):.3f}, {moh:+.3f}      | {r['n_per_order']}")
    # odd+even pair averages of the model-as-to-move (order0) -> hammer broken, equal weight
    print("PAIR AVERAGES (h_odd+h_even, model-as-to-move, equal weight -> skill):")
    pair_wrs = []
    for a, b in [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]:
        if a in H and b in H:
            wr = (H[a]["wr_o0"] + H[b]["wr_o0"]) / 2.0
            ds = (H[a]["m_o0"] + H[b]["m_o0"]) / 2.0
            va = H[a]["wr_o0"] * (1 - H[a]["wr_o0"]) / H[a]["n_per_order"]
            vb = H[b]["wr_o0"] * (1 - H[b]["wr_o0"]) / H[b]["n_per_order"]
            pse = 0.5 * math.sqrt(va + vb)
            pair_wrs.append(wr)
            print(f"  h{a:02d}+h{b:02d}: winrate={wr:.3f} ± {pse:.3f}  dScore={ds:+.3f}")
    if pair_wrs:
        print(f"  MEAN of pairs (overall skill, equal-weight): {sum(pair_wrs)/len(pair_wrs):.3f}")
    summary[lbl] = {f"h{h:02d}": H[h] for h in H}
json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)
print(f"\n[parallel-eval] done in {time.time()-t0:.0f}s")
print("PARALLEL_EVAL_DONE", flush=True)
