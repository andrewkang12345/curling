#!/usr/bin/env python3
"""Decisive high-N, both-orders (hammer-balanced) head-to-head.

Compares a WorldModel champion (A) against either the human prior (CsasPlayer) or another
WorldModel (B), over many distinct roots per horizon. Winrate is averaged over BOTH throwing
orders so the last-stone (hammer) advantage cancels and reflects true skill. Deterministic
selection (matches the in-loop protocol). Reports winrate +/- SE + a verdict per horizon + overall.

    python scripts/_eval_highN.py --champion <A.pt> --vs prior   [--N 250] [--horizons 2,3,4,5]
    python scripts/_eval_highN.py --champion <A.pt> --vs <B.pt>  (A vs world model B)
"""
import sys, math, json, argparse
sys.path.insert(0, "src")
import world  # noqa: F401
from world.config import Config
from world.train.horizon_loop import h2h_eval

ap = argparse.ArgumentParser()
ap.add_argument("--champion", required=True, help="WorldModel checkpoint (player A)")
ap.add_argument("--vs", default="prior", help="'prior' or a WorldModel checkpoint path (player B)")
ap.add_argument("--N", type=int, default=250)
ap.add_argument("--horizons", default="2,3,4,5")
ap.add_argument("--out", default=None)
ap.add_argument("--noisy", action="store_true", help="robust selection (avg over noise_samples) + noisy realized throws")
ap.add_argument("--root-shard", default=None, help="k/n: evaluate only the disjoint root subset k of n (within-horizon parallel sharding)")
args = ap.parse_args()
root_shard = None
if args.root_shard:
    _k, _n = args.root_shard.split("/"); root_shard = (int(_k), int(_n))

cfg = Config()
horizons = [int(x) for x in args.horizons.split(",")]
if args.vs == "prior":
    opp_kind = "csas"
    opp_a = cfg.csas_path(cfg.paths.prior_policy_ckpt).as_posix()
    opp_b = cfg.csas_path(cfg.paths.prior_value_ckpt).as_posix()
    label = "prior"
else:
    opp_kind, opp_a, opp_b, label = "world", args.vs, None, "worldB"
from world import env_bridge as _eb
print(f"[eval] A={args.champion}\n[eval] B({label})={opp_a}\n[eval] N={args.N}/horizon, both orders, "
      f"{'NOISY (robust select + realized noise)' if args.noisy else 'deterministic'}\n"
      f"[eval] RULES: boundary_removal={'ON (real takeout rules)' if _eb.BOUNDARY_REMOVAL else 'OFF (HISTORICAL convention!)'}",
      flush=True)

results, tot_w, tot_n = {}, 0.0, 0
for h in horizons:
    wr = h2h_eval(cfg, args.champion, opp_kind, opp_a, opp_b, h, n_roots=args.N, noisy=args.noisy, root_shard=root_shard)
    n, w = wr["n_ends"], wr["winrate"]
    se = math.sqrt(max(w * (1 - w), 1e-9) / n)
    print(f"h{h:02d}: winrate={w:.3f} +/- {se:.3f} (n={n})  dScore={wr['mean_margin']:+.3f}  "
          f"o0={wr['winrate_order0']:.3f} o1={wr['winrate_order1']:.3f}", flush=True)
    results[f"h{h:02d}"] = wr
    tot_w += w * n
    tot_n += n

W = tot_w / tot_n
SE = math.sqrt(max(W * (1 - W), 1e-9) / tot_n)
verdict = f"A BEATS {label} (>2SE)" if W - 2 * SE > 0.5 else (f"parity vs {label}" if abs(W - 0.5) < 2 * SE else f"A below {label}")
print(f"OVERALL: winrate(A vs {label})={W:.4f} +/- {SE:.4f} (n={tot_n}) -> {verdict}", flush=True)
out = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}
out.update({"overall_winrate": W, "overall_se": SE, "verdict": verdict, "A": args.champion, "B": opp_a, "N": args.N})
if args.out:
    json.dump(out, open(args.out, "w"), indent=2)
print("EVAL_DONE", flush=True)
