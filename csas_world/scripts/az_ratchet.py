#!/usr/bin/env python3
"""AZ ratchet loop — the structurally-correct compounding setup (az_v8).

Fixes the two defects of the az_v4..v7 loop that prevented compounding:

  1. **Accumulating replay buffer.** az_converge_v2 trained each iter on ONLY that iter's
     1,600 fresh records, discarding all previous buffers — a fresh small-data fit every
     time (one-shot absorption, then plateau/drift). Here iter N trains on the UNION of
     iters 1..N (train partitions accumulate; held-out val partitions accumulate too).
  2. **Promotion gate (ratchet).** Each iter is evaluated with a MULTI-DRAW eval
     (--draws independent N=400 runs; SE ~0.006-0.013) and the new checkpoint is promoted
     to incumbent only if its mean winrate beats the incumbent's by more than 1x the
     combined SE. Otherwise the incumbent stays collector + warm-start and the loop
     continues (the new data still accumulates). The loop can therefore never drift
     backward the way az_v6b / az_v7 did.

Collection uses the 2-ply KR-UCT config with a concentrated sim budget
(exp_031: mcts_sims=120, k_widen=1.5) so the tree targets aren't starved.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import world  # noqa: E402,F401

import subprocess  # noqa: E402
import time  # noqa: E402

from az_converge_v2 import (  # noqa: E402
    GPU_SETUP, VALUE_CV, collect_iter, eval_iter, export_policy, mean_of_pairs, train_iter,
)


def selfplay_collect_iter(policy_ckpt, world_ckpt, out_dir, games_per_shard, seed, collect_cfg,
                          scorer="tree"):
    """Full-game self-play collection: 4 shard processes (one per GPU), each playing
    `games_per_shard` complete ends from the pre-placed openings with search at every
    ply. Tree leaves are evaluated by the INCUMBENT's value head (--value-world), so
    value-head improvements feed back into the search operator — the loop-closure the
    root-pool collector never had. Idempotent (skips if 4 shards exist)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if len(list(Path(out_dir).glob("shard*.npz"))) >= 4:
        print(f"[ratchet] skip selfplay collect: 4 shards already present (resume)", flush=True)
        return
    t0 = time.time()
    procs = []
    for k in range(4):
        out = str(Path(out_dir) / f"shard{k}.npz")
        cmd = (
            f"{GPU_SETUP}; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; "
            f"CUDA_VISIBLE_DEVICES={k} XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 "
            f"python3 -m world.search.selfplay --config {collect_cfg} "
            f"--games {games_per_shard} --num-shards 4 --shard-id {k} "
            f"--split train --seed {seed} --scorer {scorer} "
            f"--policy {policy_ckpt} --value {VALUE_CV} --value-world {world_ckpt} "
            f"--out {out} --device cuda:0"
        )
        procs.append(subprocess.Popen(["bash", "-lc", cmd], cwd=str(ROOT)))
    for p in procs:
        p.wait()
    n = len(list(Path(out_dir).glob("shard*.npz")))
    print(f"[ratchet] selfplay collected {n} shards in {time.time()-t0:.0f}s", flush=True)


def add_selfplay_to_accum(collect_dir, accum_train, accum_val, prefix):
    """Whole-game shards: shards 0-2 -> train, shard 3 -> val (75/25, as before)."""
    Path(accum_train).mkdir(parents=True, exist_ok=True)
    Path(accum_val).mkdir(parents=True, exist_ok=True)
    for k in range(4):
        src = (Path(collect_dir) / f"shard{k}.npz").resolve()
        if not src.exists():
            continue
        dst_dir = accum_train if k < 3 else accum_val
        dst = Path(dst_dir) / f"{prefix}_shard{k}.npz"
        if not dst.exists():
            dst.symlink_to(src)
    tot_t = len(list(Path(accum_train).glob("*.npz")))
    tot_v = len(list(Path(accum_val).glob("*.npz")))
    print(f"[ratchet] accum totals: {tot_t} train, {tot_v} val files", flush=True)


def add_to_accum(collect_dir, accum_train, accum_val, prefix):
    """Symlink this iter's shards into the accumulating train/val dirs.
    Per horizon: shards 0,1,2 -> train, shard 3 -> val (same 75/25 split as before)."""
    Path(accum_train).mkdir(parents=True, exist_ok=True)
    Path(accum_val).mkdir(parents=True, exist_ok=True)
    n_t = n_v = 0
    for hdir in sorted(Path(collect_dir).glob("h*")):
        hh = hdir.name
        for k in range(3):
            src = (hdir / f"shard{k}.npz").resolve()
            dst = Path(accum_train) / f"{prefix}_{hh}_shard{k}.npz"
            if src.exists() and not dst.exists():
                dst.symlink_to(src)
                n_t += 1
        src3 = (hdir / "shard3.npz").resolve()
        dst3 = Path(accum_val) / f"{prefix}_{hh}_shard3.npz"
        if src3.exists() and not dst3.exists():
            dst3.symlink_to(src3)
            n_v += 1
    tot_t = len(list(Path(accum_train).glob("*.npz")))
    tot_v = len(list(Path(accum_val).glob("*.npz")))
    print(f"[ratchet] accum += {n_t} train / {n_v} val files (totals: {tot_t} train, {tot_v} val)", flush=True)


def multi_draw_eval(ckpt, eval_base, draws, eval_n, opponent="prior"):
    """Run `draws` independent full evals of `ckpt` vs `opponent` ('prior' or a world ckpt
    path); return mean/SE of mean-of-pairs wr + dScore (ckpt's perspective)."""
    stats = []
    for d in range(1, draws + 1):
        out = f"{eval_base}_run{d}"
        if not (Path(out) / "summary.json").exists():
            eval_iter_vs(ckpt, out, eval_n, opponent)
        r = mean_of_pairs_any(out)
        if r:
            stats.append(r)
            print(f"[ratchet]   draw {d}: wr={r['mean_wr']:.4f}  ds={r['mean_ds']:+.4f}", flush=True)
    n = len(stats)
    if n == 0:
        return None
    mw = sum(s["mean_wr"] for s in stats) / n
    md = sum(s["mean_ds"] for s in stats) / n
    sw = math.sqrt(sum((s["mean_wr"] - mw) ** 2 for s in stats) / max(n - 1, 1)) / math.sqrt(n)
    sd = math.sqrt(sum((s["mean_ds"] - md) ** 2 for s in stats) / max(n - 1, 1)) / math.sqrt(n)
    return dict(wr=mw, wr_se=sw, ds=md, ds_se=sd, n=n)


def eval_iter_vs(ckpt, eval_out, eval_n, opponent):
    """Like az_converge_v2.eval_iter but with a configurable opponent ('prior' or world ckpt)."""
    import subprocess as sp
    Path(eval_out).mkdir(parents=True, exist_ok=True)
    log = f"{eval_out}/eval.log"
    env_pre = (
        "unset LD_LIBRARY_PATH; "
        "export PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1; "
        "export GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing "
        "GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none; "
    )
    cmd = (f"{env_pre} python3 scripts/_eval_parallel.py --champion {ckpt} --vs {opponent} "
           f"--N {eval_n} --horizons 1,2,3,4,5,6,7,8,9,10 --gpus 0,1,2,3 --shards 4 --noisy "
           f"--out-dir {eval_out} >> {log} 2>&1")
    rc = sp.run(["bash", "-lc", cmd], cwd=str(ROOT)).returncode
    print(f"[ratchet] eval rc={rc} -> {eval_out}", flush=True)


def mean_of_pairs_any(eval_out):
    """Label-agnostic mean-of-pairs aggregation (works for prior AND world opponents)."""
    import glob as _glob
    import json as _json
    from collections import defaultdict
    by_h = defaultdict(lambda: dict(n=0.0, w0=0.0, m0=0.0))
    for f in sorted(_glob.glob(f"{eval_out}/*__h*__s*of*.json")):
        try:
            d = _json.load(open(f))
        except Exception:
            continue
        for k, v in d.items():
            if not k.startswith("h") or not isinstance(v, dict) or "n_ends" not in v:
                continue
            h = int(k[1:]); no = v["n_ends"] / 2.0
            by_h[h]["n"] += no
            by_h[h]["w0"] += v["winrate_order0"] * no
            by_h[h]["m0"] += v.get("mean_margin_order0", 0.0) * no
    per_h = {h: dict(wr=r["w0"] / r["n"], m=r["m0"] / r["n"]) for h, r in by_h.items() if r["n"] > 0}
    pw, pd = [], []
    for a, b in ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10)):
        if a in per_h and b in per_h:
            pw.append(0.5 * (per_h[a]["wr"] + per_h[b]["wr"]))
            pd.append(0.5 * (per_h[a]["m"] + per_h[b]["m"]))
    if not pw:
        return None
    return dict(mean_wr=sum(pw) / len(pw), mean_ds=sum(pd) / len(pd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-world", default="checkpoints/csas_world/exp_021_valuemcts_earlystop/best.pt")
    ap.add_argument("--collect-config", default="configs/exp_031_2ply_sims120.yaml")
    ap.add_argument("--train-config", default="configs/exp_021_valuemcts_earlystop.yaml")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--draws", type=int, default=3, help="independent eval draws per gate decision")
    ap.add_argument("--max-roots", type=int, default=160)
    ap.add_argument("--horizons", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--eval-n", type=int, default=400)
    ap.add_argument("--gate-k", type=float, default=1.0, help="promote iff dwr > gate_k * combined SE")
    ap.add_argument("--selfplay-games", type=int, default=0,
                    help="if > 0, collect via FULL-GAME SELF-PLAY (this many games per GPU shard; "
                         "each game yields max_horizon records) instead of the root-pool collector")
    ap.add_argument("--selfplay-scorer", choices=["tree", "terminal", "tree_terminal", "screen_tree"], default="tree",
                    help="per-ply scorer for self-play collection (tree = az_v9; terminal = az_v10 value-free; "
                         "tree_terminal = az_v11 dense-root KR-UCT + terminal-rollout leaves; "
                         "screen_tree = az_v12 noise-robust screen -> depth-2 tree over survivors)")
    ap.add_argument("--gate-metric", choices=["wr", "ds"], default="wr",
                    help="primary promotion metric: wr (winrate) or ds (score differential)")
    ap.add_argument("--gate-opponent", choices=["prior", "incumbent"], default="prior",
                    help="az_v15+: 'incumbent' gates by DIRECT h2h vs the current incumbent "
                         "(promotion = beat the incumbent; the AlphaZero gate, validated by the "
                         "EXP-042 meta-game matrix). 'prior' is the legacy fixed-opponent gate.")
    ap.add_argument("--stop-after-nonpromotions", type=int, default=0,
                    help=">0: stop the loop after this many CONSECUTIVE non-promotions (gate "
                         "convergence signal of the train-to-capacity certificate)")
    ap.add_argument("--work", default="checkpoints/csas_world/az_v8_ratchet")
    ap.add_argument("--resume", action="store_true",
                    help="continue a finished run: load history.json from --work, restore the "
                         "incumbent (last promoted entry) and its multi-draw stats, and run "
                         "--iters MORE iterations starting after the last recorded one. The "
                         "accumulating buffer persists on disk, so accumulation continues.")
    args = ap.parse_args()
    horizons = [int(x) for x in args.horizons.split(",")]
    Path(args.work).mkdir(parents=True, exist_ok=True)
    base = Path(args.work).name

    accum_train = f"artifacts/replay/{base}_accum_train"
    accum_val = f"artifacts/replay/{base}_accum_val"

    if args.gate_opponent == "incumbent":
        # ---- az_v15 mode: no baseline eval needed — the gate IS direct h2h vs incumbent ----
        incumbent = args.init_world
        inc_pol = f"{args.work}/incumbent0_policy_csas.pt"
        if not Path(inc_pol).exists():
            export_policy(incumbent, inc_pol)
        inc_stats = None
        history = [dict(iter=0, world=incumbent, promoted=True)]
        json.dump(history, open(f"{args.work}/history.json", "w"), indent=2)
        start_it = 1
        print(f"===== RATCHET (incumbent-gated): incumbent = {incumbent} =====", flush=True)
    elif args.resume:
        # ---- continue a prior run: restore incumbent + stats from history.json ----
        history = json.load(open(f"{args.work}/history.json"))
        last_prom = [e for e in history if e.get("promoted")][-1]
        incumbent = last_prom["world"]
        inc_stats = dict(wr=last_prom["wr"], wr_se=last_prom["wr_se"],
                         ds=last_prom["ds"], ds_se=last_prom["ds_se"], n=last_prom.get("n", args.draws))
        if last_prom["iter"] == 0:
            inc_pol = f"{args.work}/incumbent0_policy_csas.pt"
        else:
            inc_pol = f"{args.work}/iter{last_prom['iter']}/policy_csas.pt"
        start_it = history[-1]["iter"] + 1
        print(f"===== RATCHET RESUME: incumbent = {incumbent} "
              f"(iter {last_prom['iter']}, wr={inc_stats['wr']:.4f}±{inc_stats['wr_se']:.4f}, "
              f"ds={inc_stats['ds']:+.4f}±{inc_stats['ds_se']:.4f}); continuing at iter {start_it} =====", flush=True)
    else:
        # ---- incumbent = init world; multi-draw baseline ----
        incumbent = args.init_world
        inc_pol = f"{args.work}/incumbent0_policy_csas.pt"
        if not Path(inc_pol).exists():
            export_policy(incumbent, inc_pol)
        print(f"===== RATCHET iter 0: multi-draw baseline of {incumbent} =====", flush=True)
        inc_stats = multi_draw_eval(incumbent, f"eval_out/{base}/iter0", args.draws, args.eval_n)
        print(f"[ratchet] incumbent baseline: wr={inc_stats['wr']:.4f}±{inc_stats['wr_se']:.4f}  "
              f"ds={inc_stats['ds']:+.4f}±{inc_stats['ds_se']:.4f}", flush=True)
        history = [dict(iter=0, world=incumbent, promoted=True, **inc_stats)]
        json.dump(history, open(f"{args.work}/history.json", "w"), indent=2)
        start_it = 1

    nonprom_streak = 0
    for it in range(start_it, start_it + args.iters):
        tag = f"iter{it}"
        collect_dir = f"artifacts/replay/mcts/{base}_{tag}"
        out_dir = f"{args.work}/{tag}"

        print(f"\n===== RATCHET {tag}: collect with INCUMBENT ({Path(incumbent).parent.name or incumbent}) =====", flush=True)
        if args.selfplay_games > 0:
            selfplay_collect_iter(inc_pol, incumbent, collect_dir, args.selfplay_games,
                                  seed=800 + 50 * it, collect_cfg=args.collect_config,
                                  scorer=args.selfplay_scorer)
            add_selfplay_to_accum(collect_dir, accum_train, accum_val, prefix=f"it{it}")
        else:
            collect_iter(inc_pol, collect_dir, horizons, args.max_roots,
                         seed_base=800 + 50 * it, collect_cfg=args.collect_config)
            add_to_accum(collect_dir, accum_train, accum_val, prefix=f"it{it}")

        print(f"===== RATCHET {tag}: train on ACCUMULATED union (warm-start incumbent) =====", flush=True)
        train_iter(accum_train, accum_val, incumbent, out_dir, args.train_config)
        ckpt = f"{out_dir}/best.pt"
        if not Path(ckpt).exists():
            ckpt = f"{out_dir}/model.pt"
        pol = f"{out_dir}/policy_csas.pt"
        export_policy(ckpt, pol)

        if args.gate_opponent == "incumbent":
            # direct h2h: new ckpt vs the CURRENT incumbent; parity = wr 0.5 / ds 0
            print(f"===== RATCHET {tag}: h2h gate vs incumbent ({args.draws} draws) =====", flush=True)
            stats = multi_draw_eval(ckpt, f"eval_out/{base}/{tag}_vs_inc", args.draws, args.eval_n,
                                    opponent=incumbent)
            dds, dwr = stats["ds"], stats["wr"] - 0.5
            promoted = (dds > args.gate_k * max(stats["ds_se"], 1e-9)) and (dwr > -stats["wr_se"])
            gate_str = (f"h2h ds={stats['ds']:+.4f}±{stats['ds_se']:.4f} vs gate {args.gate_k}xSE"
                        f" (wr guard: {stats['wr']:.4f} > 0.5-{stats['wr_se']:.4f})")
        else:
            print(f"===== RATCHET {tag}: multi-draw eval ({args.draws} draws) =====", flush=True)
            stats = multi_draw_eval(ckpt, f"eval_out/{base}/{tag}", args.draws, args.eval_n)
            dwr = stats["wr"] - inc_stats["wr"]
            dds = stats["ds"] - inc_stats["ds"]
            comb_se_wr = math.sqrt(stats["wr_se"] ** 2 + inc_stats["wr_se"] ** 2)
            comb_se_ds = math.sqrt(stats["ds_se"] ** 2 + inc_stats["ds_se"] ** 2)
            if args.gate_metric == "ds":
                promoted = (dds > args.gate_k * comb_se_ds) and (dwr > -comb_se_wr)
                gate_str = (f"dds={dds:+.4f} vs gate {args.gate_k}x{comb_se_ds:.4f}"
                            f" (wr guard: dwr={dwr:+.4f} > -{comb_se_wr:.4f})")
            else:
                promoted = dwr > args.gate_k * comb_se_wr
                gate_str = f"dwr={dwr:+.4f} vs gate {args.gate_k}x{comb_se_wr:.4f}"
        print(f"[ratchet] {tag}: wr={stats['wr']:.4f}±{stats['wr_se']:.4f} ds={stats['ds']:+.4f}±{stats['ds_se']:.4f}  "
              f"{gate_str}  => {'PROMOTED' if promoted else 'kept incumbent'}", flush=True)

        history.append(dict(iter=it, world=ckpt, promoted=promoted, dwr=dwr, dds=dds, **stats))
        json.dump(history, open(f"{args.work}/history.json", "w"), indent=2)

        if promoted:
            incumbent, inc_pol, inc_stats = ckpt, pol, stats
            nonprom_streak = 0
        else:
            nonprom_streak += 1
            if args.stop_after_nonpromotions > 0 and nonprom_streak >= args.stop_after_nonpromotions:
                print(f"===== RATCHET gate-converged: {nonprom_streak} consecutive non-promotions =====", flush=True)
                break

    print(f"\n===== RATCHET DONE: incumbent = {incumbent} "
          f"(wr={inc_stats['wr']:.4f}±{inc_stats['wr_se']:.4f}) =====", flush=True)
    print("AZ_RATCHET_DONE", flush=True)


if __name__ == "__main__":
    main()
