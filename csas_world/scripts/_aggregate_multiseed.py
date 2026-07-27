#!/usr/bin/env python3
"""Aggregate the multi-seed eval draws for the deploy shortlist.

Each draw = one full N=400, 10-horizon, both-orders eval (one _eval_parallel.py run).
Draws are independent because candidate sampling (sample_actions_z) is unseeded.
Per candidate: mean-of-pairs winrate + dScore per draw, then mean +/- SE across draws.

exp_021's original 0.5624 draw exists only as a recorded aggregate (its shards were
overwritten in the racing incident), so it is entered as a RECORDED constant.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def mean_of_pairs(eval_out, label="prior"):
    by_h = defaultdict(lambda: dict(n=0.0, w0=0.0, m0=0.0))
    for f in sorted(Path(eval_out).glob(f"{label}__h*__s*of*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for k, v in d.items():
            if not k.startswith("h"):
                continue
            h = int(k[1:])
            no = v["n_ends"] / 2.0
            by_h[h]["n"] += no
            by_h[h]["w0"] += v["winrate_order0"] * no
            by_h[h]["m0"] += v.get("mean_margin_order0", 0.0) * no
    per_h = {h: dict(wr=r["w0"] / r["n"], m=r["m0"] / r["n"]) for h, r in by_h.items() if r["n"] > 0}
    pair_wr, pair_ds = [], []
    for a, b in ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10)):
        if a in per_h and b in per_h:
            pair_wr.append(0.5 * (per_h[a]["wr"] + per_h[b]["wr"]))
            pair_ds.append(0.5 * (per_h[a]["m"] + per_h[b]["m"]))
    if not pair_wr:
        return None
    return dict(wr=sum(pair_wr) / len(pair_wr), ds=sum(pair_ds) / len(pair_ds))


CANDIDATES = {
    "az_v6_iter1 (2-ply+VFM, best dScore)": dict(
        ckpt="checkpoints/csas_world/az_v6_2ply_unfrozen/iter1/best.pt",
        dirs=["eval_out/az_v6_2ply_unfrozen/iter1_vs_prior",
              "eval_out/az_v6b_iter2/iter0_vs_prior",
              "eval_out/az_v7_3ply_from_v6/iter0_vs_prior"],
        recorded=[],
    ),
    "az_v5_iter1 (1-ply, VFM=false)": dict(
        ckpt="checkpoints/csas_world/az_v5_novaluemcts/iter1/best.pt",
        dirs=["eval_out/az_v5_novaluemcts/iter1_vs_prior",
              "eval_out/multiseed/az_v5_iter1_run2",
              "eval_out/multiseed/az_v5_iter1_run3"],
        recorded=[],
    ),
    "exp_021 best (baseline champion)": dict(
        ckpt="checkpoints/csas_world/exp_021_valuemcts_earlystop/best.pt",
        dirs=["eval_out/az_v4/iter0_vs_prior",
              "eval_out/az_v5_novaluemcts/iter0_vs_prior",
              "eval_out/az_v6_2ply_unfrozen/iter0_vs_prior"],
        recorded=[dict(wr=0.5624, ds=0.2136, note="original proper eval (shards lost to racing incident)")],
    ),
    "exp_019 last (conservative baseline)": dict(
        ckpt="checkpoints/csas_world/exp_019_consolidate/last.pt",
        dirs=["eval_out/proper/exp019_consolidated_vs_prior",
              "eval_out/multiseed/exp019_run2",
              "eval_out/multiseed/exp019_run3"],
        recorded=[],
    ),
}


def main():
    print("=" * 88)
    print("MULTI-SEED EVAL AGGREGATE — deploy shortlist, N=400 x 10 horizons x both orders per draw")
    print("=" * 88)
    rows = []
    for name, spec in CANDIDATES.items():
        draws = []
        for d in spec["dirs"]:
            r = mean_of_pairs(ROOT / d)
            if r:
                draws.append(dict(**r, src=d))
            else:
                print(f"  [!] missing/empty: {d}")
        for rec in spec["recorded"]:
            draws.append(dict(wr=rec["wr"], ds=rec["ds"], src=f"RECORDED: {rec['note']}"))
        if not draws:
            continue
        n = len(draws)
        mw = sum(x["wr"] for x in draws) / n
        md = sum(x["ds"] for x in draws) / n
        sw = math.sqrt(sum((x["wr"] - mw) ** 2 for x in draws) / max(n - 1, 1))
        sd = math.sqrt(sum((x["ds"] - md) ** 2 for x in draws) / max(n - 1, 1))
        print(f"\n----- {name} -----")
        print(f"  ckpt: {spec['ckpt']}")
        for x in draws:
            print(f"    draw: wr={x['wr']:.4f}  ds={x['ds']:+.4f}   ({x['src']})")
        print(f"  MEAN over {n} draws: wr = {mw:.4f} ± {sw/math.sqrt(n):.4f} (SE)   "
              f"dScore = {md:+.4f} ± {sd/math.sqrt(n):.4f} (SE)")
        rows.append(dict(name=name, ckpt=spec["ckpt"], n=n, wr=mw, wr_se=sw / math.sqrt(n),
                         ds=md, ds_se=sd / math.sqrt(n),
                         draws=[{k: v for k, v in x.items()} for x in draws]))

    rows.sort(key=lambda r: -r["wr"])
    print("\n" + "=" * 88)
    print("RANKING (by mean winrate):")
    print(f"{'candidate':45s} | {'n':>2} | {'wr ± SE':>17} | {'dScore ± SE':>17}")
    for r in rows:
        print(f"{r['name']:45s} | {r['n']:2d} | {r['wr']:.4f} ± {r['wr_se']:.4f} | {r['ds']:+.4f} ± {r['ds_se']:.4f}")

    out = ROOT / "eval_out/multiseed/aggregate.json"
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\n[wrote] {out}")
    print("AGGREGATE_DONE")


if __name__ == "__main__":
    main()
