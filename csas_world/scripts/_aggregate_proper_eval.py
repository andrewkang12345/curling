#!/usr/bin/env python3
"""Aggregate the proper 3-way eval JSON shards (eval_out/proper/<tag>_vs_prior/*.json) into a
single summary CSV + pretty-print the per-horizon table that goes into main.tex and into the log.
"""
import json, glob, math, os, sys
from collections import defaultdict

ROOT = "eval_out/proper"
TAGS = ("exp019_consolidated", "exp017_perStage_h07r0", "exp017_sequential_h10r1")


def aggregate_horizon(jsons):
    """Combine per-shard JSONs for one (model, horizon). Returns dict with merged numbers."""
    if not jsons:
        return None
    n_eff = 0.0
    s_w0 = s_w1 = s_m0 = s_m1 = 0.0
    for d in jsons:
        h_key = [k for k in d if k.startswith("h")][0]
        wr = d[h_key]
        no = wr["n_ends"] / 2.0
        s_w0 += wr["winrate_order0"] * no
        s_w1 += wr["winrate_order1"] * no
        s_m0 += wr.get("mean_margin_order0", 0.0) * no
        s_m1 += wr.get("mean_margin_order1", 0.0) * no
        n_eff += no
    if n_eff == 0:
        return None
    return dict(
        n_per_order=int(n_eff),
        wr_o0=s_w0 / n_eff, wr_o1=s_w1 / n_eff,
        m_o0=s_m0 / n_eff, m_o1=s_m1 / n_eff,
    )


def collect(tag):
    """Per-horizon aggregates for one model."""
    by_h = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(ROOT, f"{tag}_vs_prior", "*.json"))):
        if "summary" in f:
            continue
        with open(f) as fh:
            by_h[int([k for k in json.load(open(f)) if k.startswith("h")][0][1:])].append(json.load(open(f)))
    out = {}
    for h, lst in by_h.items():
        agg = aggregate_horizon(lst)
        if agg:
            out[h] = agg
    return out


def hammer_split(by_h):
    """Map (with-hammer wr, dScore) and (without-hammer wr, dScore) per horizon."""
    rows = {}
    for h, r in by_h.items():
        if h % 2 == 1:                                          # odd h: order0 = to-move = with hammer
            rows[h] = dict(
                wr_h=r["wr_o0"], m_h=r["m_o0"],
                wr_noh=r["wr_o1"], m_noh=r["m_o1"],
                n=r["n_per_order"],
            )
        else:                                                   # even h: order1 = with hammer
            rows[h] = dict(
                wr_h=r["wr_o1"], m_h=r["m_o1"],
                wr_noh=r["wr_o0"], m_noh=r["m_o0"],
                n=r["n_per_order"],
            )
    return rows


def pair_average(by_h):
    """Hammer-neutral skill: average A-as-to-move (order0) across adjacent (odd, even) pairs."""
    pairs = []
    for a, b in ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10)):
        if a in by_h and b in by_h:
            wr = 0.5 * (by_h[a]["wr_o0"] + by_h[b]["wr_o0"])
            ds = 0.5 * (by_h[a]["m_o0"] + by_h[b]["m_o0"])
            # SE: independent roots in val pool -> sqrt(var/n) per horizon, then 1/2 of the sum
            v_a = by_h[a]["wr_o0"] * (1 - by_h[a]["wr_o0"]) / max(by_h[a]["n_per_order"], 1)
            v_b = by_h[b]["wr_o0"] * (1 - by_h[b]["wr_o0"]) / max(by_h[b]["n_per_order"], 1)
            pse = 0.5 * math.sqrt(v_a + v_b)
            pairs.append((a, b, wr, ds, pse))
    return pairs


def se(p, n):
    return math.sqrt(max(p * (1 - p), 1e-9) / max(n, 1))


def main():
    print("=" * 80)
    print(f"PROPER 3-WAY EVAL  vs human prior  (NOISY alternating play, both orders, "
          f"free-guard-zone rule)")
    print("=" * 80)
    all_data = {}
    for tag in TAGS:
        by_h = collect(tag)
        if not by_h:
            print(f"\n[!!] no data for {tag}")
            continue
        rows = hammer_split(by_h)
        pairs = pair_average(by_h)
        all_data[tag] = dict(by_h=by_h, rows=rows, pairs=pairs)

        print(f"\n----- {tag} -----")
        print(f"{'h':>3} | {'n':>5} | {'wr-w/-hammer':>14} | {'dScore-w/-h':>11} | "
              f"{'wr-w/o-hammer':>14} | {'dScore-w/o-h':>12}")
        for h in sorted(rows):
            r = rows[h]
            print(f"{h:3d} | {r['n']:5d} | "
                  f"{r['wr_h']:.3f} ± {se(r['wr_h'], r['n']):.3f} | "
                  f"{r['m_h']:+11.3f} | "
                  f"{r['wr_noh']:.3f} ± {se(r['wr_noh'], r['n']):.3f} | "
                  f"{r['m_noh']:+12.3f}")
        print(f"\n  Hammer-neutral pairs (skill, A-as-to-move):")
        for a, b, wr, ds, pse in pairs:
            print(f"    pair (h{a},h{b}): wr = {wr:.3f} ± {pse:.3f}    dScore = {ds:+.3f}")
        mean_wr = sum(p[2] for p in pairs) / max(len(pairs), 1)
        mean_ds = sum(p[3] for p in pairs) / max(len(pairs), 1)
        print(f"  MEAN of pairs (overall skill): {mean_wr:.3f}    MEAN dScore: {mean_ds:+.3f}")

    # write a compact summary JSON
    summary = {}
    for tag, d in all_data.items():
        summary[tag] = {
            "per_horizon": {f"h{h:02d}": d["rows"][h] for h in sorted(d["rows"])},
            "pairs": [{"a": a, "b": b, "wr": wr, "dScore": ds, "se": pse}
                      for a, b, wr, ds, pse in d["pairs"]],
            "mean_of_pairs_wr": (sum(p[2] for p in d["pairs"]) / max(len(d["pairs"]), 1)),
            "mean_of_pairs_dscore": (sum(p[3] for p in d["pairs"]) / max(len(d["pairs"]), 1)),
        }
    out_path = os.path.join(ROOT, "summary_3way.json")
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[wrote] {out_path}")
    print("AGGREGATE_DONE")


if __name__ == "__main__":
    main()
