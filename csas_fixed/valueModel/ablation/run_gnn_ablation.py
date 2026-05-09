#!/usr/bin/env python3
"""
GNN ablation: EGNN and Graph Transformer architectures for curling value prediction.

Uses leave-one-competition-out CV, synth fracs [0.0, 0.5, 1.0].
Compares against SetTransformer baseline (MSE=2.122 at synth100%).

wandb project: curling-value-gnn (OFFLINE mode)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import ValueDataset, NUM_STONES, POS_MAX
from gnn_models import GNN_REGISTRY

FLIP_CENTER_X = 1500.0 / POS_MAX
EDGE_SCALAR_FEATURE = os.environ.get("GNN_EDGE_SCALAR_MODE", "thrower_masked_button_region_span").strip()
NODE_FEATURE_MODE = os.environ.get("GNN_NODE_FEATURE_MODE", "none").strip()

SYNTH_FRACS = [0.0, 0.50, 1.0]

# GNN Configurations to test
CONFIGS = {
    "gnn_egnn_small": dict(
        arch="egnn", hidden_dim=128, n_layers=3, n_heads=4,
        lr=3e-4, dropout=0.1, weight_decay=1e-4,
    ),
    "gnn_egnn_medium": dict(
        arch="egnn", hidden_dim=256, n_layers=4, n_heads=4,
        lr=3e-4, dropout=0.1, weight_decay=1e-4,
    ),
    "gnn_transformer_small": dict(
        arch="graph_transformer", hidden_dim=128, n_layers=3, n_heads=4,
        lr=3e-4, dropout=0.1, weight_decay=1e-4,
    ),
    "gnn_transformer_medium": dict(
        arch="graph_transformer", hidden_dim=256, n_layers=4, n_heads=4,
        lr=3e-4, dropout=0.1, weight_decay=1e-4,
    ),
}


def augment_flip_batch(x):
    """Random horizontal flip augmentation."""
    B = x.size(0)
    stones = x.view(B, NUM_STONES, 2).clone()
    flip_mask = (torch.rand(B, device=x.device) < 0.5).view(B, 1, 1)
    ip = ((stones.sum(dim=-1) > 0.001) & (stones.max(dim=-1).values < 0.999)).unsqueeze(-1)
    flipped_x = FLIP_CENTER_X - stones[:, :, 0:1]
    new_x = torch.where(flip_mask & ip, flipped_x, stones[:, :, 0:1])
    stones = torch.cat([new_x, stones[:, :, 1:2]], dim=-1)
    return stones.view(B, -1)


def materialize_dataset(ds):
    """Load entire dataset into tensors."""
    loader = DataLoader(ds, batch_size=8192, shuffle=False, num_workers=0)
    xs, cs, ys = [], [], []
    for x, c, y in loader:
        xs.append(x); cs.append(c); ys.append(y)
    return (torch.cat(xs, 0), torch.cat(cs, 0), torch.cat(ys, 0),
            pd.to_numeric(ds.df["CompetitionID"], errors="coerce").to_numpy(dtype="int64"))


def build_model(arch, input_dim, cond_dim, **kwargs):
    if arch in GNN_REGISTRY:
        return GNN_REGISTRY[arch](input_dim=input_dim, cond_dim=cond_dim, **kwargs)
    raise ValueError(f"Unknown arch: {arch}")


def train_model(train_ds, val_ds, arch, input_dim, cond_dim, lr, weight_decay,
                device, do_flip, epochs=150, patience=20, batch_size=1024, **mkw):
    model = build_model(arch, input_dim, cond_dim, **mkw).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
                    pin_memory=(device.type == "cuda"))
    vl = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0,
                    pin_memory=(device.type == "cuda"))

    best_val, best_ep, no_imp = float("inf"), 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        r, cnt = 0.0, 0
        for x, c, y in tl:
            x = x.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if do_flip:
                x = augment_flip_batch(x)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x, c), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            r += loss.item() * x.size(0)
            cnt += x.size(0)
        tr_mse = r / max(1, cnt)

        model.eval()
        vr, vc = 0.0, 0
        with torch.no_grad():
            for x, c, y in vl:
                x = x.to(device, non_blocking=True)
                c = c.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                vr += criterion(model(x, c), y).item() * x.size(0)
                vc += x.size(0)
        val_mse = vr / max(1, vc)

        wandb.log({
            "epoch": ep,
            "train/mse": tr_mse,
            "val/mse": val_mse,
        })

        if val_mse < best_val:
            best_val, best_ep, no_imp = val_mse, ep, 0
        else:
            no_imp += 1
        if patience > 0 and no_imp >= patience:
            break

    return {
        "best_val_mse": best_val,
        "best_val_rmse": float(np.sqrt(best_val)),
        "best_epoch": best_ep,
        "final_epoch": ep,
        "final_train_mse": tr_mse,
        "n_params": n_params,
    }


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
    elif not args.no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}", flush=True)
    print(f"Edge scalar feature: {EDGE_SCALAR_FEATURE}", flush=True)
    print(f"Node feature mode: {NODE_FEATURE_MODE}", flush=True)

    # Set wandb API key
    os.environ["WANDB_API_KEY"] = args.wandb_api_key

    print("Loading data (cond_dim=3)...", flush=True)
    real_ds = ValueDataset(args.stones_csv, args.ends_csv,
                           augment_positions=False, augment_flip=False)
    real_Xp, real_Xc, real_Y, real_comps = materialize_dataset(real_ds)
    comp_ids = sorted(set(real_comps.tolist()))
    if args.only_holdouts:
        allowed = {int(c) for c in args.only_holdouts}
        comp_ids = [c for c in comp_ids if c in allowed]
        if not comp_ids:
            raise ValueError(f"No matching holdouts found for {sorted(allowed)}")
    input_dim, cond_dim = real_ds.input_dim, real_ds.cond_dim
    print(f"  Real: {real_Xp.shape[0]} samples, cond_dim={cond_dim}, comps={comp_ids}", flush=True)
    assert cond_dim == 3

    synth_ds = ValueDataset(args.synth_stones_csv, args.synth_ends_csv,
                            augment_positions=False, augment_flip=False)
    synth_Xp, synth_Xc, synth_Y, _ = materialize_dataset(synth_ds)
    n_synth = synth_Xp.shape[0]
    print(f"  Synth: {n_synth} samples, cond_dim={synth_ds.cond_dim}", flush=True)

    rng = np.random.default_rng(42)
    synth_perm = rng.permutation(n_synth)
    selected_synth_fracs = SYNTH_FRACS
    if args.only_synth_fracs:
        selected_synth_fracs = [float(sf) for sf in args.only_synth_fracs]
    synth_subsets = {
        sf: synth_perm[:min(int(round(sf * n_synth)), n_synth)] if sf > 0
        else np.array([], dtype=np.int64)
        for sf in selected_synth_fracs
    }
    fold_indices = {
        c: (np.where(real_comps != c)[0], np.where(real_comps == c)[0])
        for c in comp_ids
    }

    # Build experiment list, optionally filtered
    experiments = []
    for cname, cfg in CONFIGS.items():
        if args.only_configs and cname not in args.only_configs:
            continue
        for sf in selected_synth_fracs:
            experiments.append({
                "config_name": cname,
                "synth_frac": sf,
                "do_flip": True,  # always use flip augmentation
                **cfg,
            })

    total = len(experiments) * len(comp_ids)
    print(f"\n{'='*80}", flush=True)
    print(f"{len(experiments)} configs x {len(comp_ids)} folds = {total} runs", flush=True)
    print(f"{'='*80}\n", flush=True)

    all_results = []
    run_idx = 0

    for exp in experiments:
        cname, arch, sf = exp["config_name"], exp["arch"], exp["synth_frac"]
        do_flip = exp["do_flip"]
        group = f"{cname}_s{int(sf*100):03d}"
        fold_mses = []

        for fi, hcomp in enumerate(comp_ids):
            run_idx += 1
            rname = f"{group}_f{hcomp}"

            tr_idx, va_idx = fold_indices[hcomp]
            s_idx = synth_subsets[sf]
            if len(s_idx) > 0:
                tr_Xp = torch.cat([real_Xp[tr_idx], synth_Xp[s_idx]], 0)
                tr_Xc = torch.cat([real_Xc[tr_idx], synth_Xc[s_idx]], 0)
                tr_Y  = torch.cat([real_Y[tr_idx],  synth_Y[s_idx]], 0)
            else:
                tr_Xp, tr_Xc, tr_Y = real_Xp[tr_idx], real_Xc[tr_idx], real_Y[tr_idx]

            train_td = TensorDataset(tr_Xp, tr_Xc, tr_Y)
            val_td = TensorDataset(real_Xp[va_idx], real_Xc[va_idx], real_Y[va_idx])

            print(f"[{run_idx}/{total}] {rname} | train={len(train_td)} | val={len(va_idx)}",
                  flush=True)

            wcfg = {
                "arch": arch,
                "config_name": cname,
                "synth_frac": sf,
                "cond_dim": cond_dim,
                "edge_scalar_feature": EDGE_SCALAR_FEATURE,
                "node_feature_mode": NODE_FEATURE_MODE,
                "holdout_comp": hcomp,
                "total_train_size": len(train_td),
            }
            for k, v in exp.items():
                if k not in wcfg and k not in ("do_flip",):
                    wcfg[k] = v

            run_handle = wandb.init(
                project=args.wandb_project,
                name=rname,
                group=group,
                job_type=arch,
                tags=[arch, cname, f"s{int(sf*100):03d}", "gnn", "cond3", EDGE_SCALAR_FEATURE, NODE_FEATURE_MODE],
                config=wcfg,
                reinit=True,
                mode="offline",
            )

            mkw = {k: v for k, v in exp.items()
                   if k not in ("config_name", "synth_frac", "arch", "lr",
                                "weight_decay", "do_flip")}
            t0 = time.time()
            try:
                result = train_model(
                    train_td, val_td,
                    arch=arch, input_dim=input_dim, cond_dim=cond_dim,
                    lr=exp["lr"], weight_decay=exp["weight_decay"],
                    device=device, do_flip=do_flip,
                    epochs=args.epochs, patience=args.patience,
                    batch_size=args.batch_size,
                    **mkw,
                )
                elapsed = time.time() - t0
                wandb.log({
                    "test/mse": result["best_val_mse"],
                    "test/rmse": result["best_val_rmse"],
                    "train/final_mse": result["final_train_mse"],
                    "best_epoch": result["best_epoch"],
                    "n_params": result["n_params"],
                    "elapsed_sec": elapsed,
                })
                wandb.summary.update({
                    "test_mse": result["best_val_mse"],
                    "test_rmse": result["best_val_rmse"],
                    "n_params": result["n_params"],
                })
                print(f"  -> mse={result['best_val_mse']:.4f} "
                      f"ep={result['best_epoch']}/{result['final_epoch']} "
                      f"params={result['n_params']:,} {elapsed:.1f}s", flush=True)
                fold_mses.append(result["best_val_mse"])
            except Exception as e:
                import traceback
                traceback.print_exc()
                elapsed = time.time() - t0
                result = {
                    "best_val_mse": float("inf"),
                    "best_val_rmse": float("inf"),
                    "best_epoch": -1,
                    "final_epoch": -1,
                    "final_train_mse": float("inf"),
                    "n_params": 0,
                }
                print(f"  -> ERROR: {e}", flush=True)

            run_handle.finish()
            all_results.append({
                "arch": arch,
                "config_name": cname,
                "synth_frac": sf,
                "cond_dim": cond_dim,
                "holdout_comp": hcomp,
                "test_mse": result["best_val_mse"],
                "test_rmse": result["best_val_rmse"],
                "best_epoch": result["best_epoch"],
                "n_params": result.get("n_params", 0),
                "elapsed_sec": elapsed,
            })

        if fold_mses:
            avg = np.mean(fold_mses)
            print(f"  >> {group} AVG mse={avg:.4f} rmse={np.sqrt(avg):.4f}", flush=True)

    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / "gnn_results.csv", index=False)

    summary = df.groupby(["config_name", "synth_frac"], as_index=False).agg(
        mean_test_mse=("test_mse", "mean"),
        std_test_mse=("test_mse", "std"),
        mean_test_rmse=("test_rmse", "mean"),
        n_params=("n_params", "first"),
        mean_elapsed=("elapsed_sec", "mean"),
    ).sort_values("mean_test_mse")
    summary.to_csv(out_dir / "gnn_summary.csv", index=False)

    # Print summary table
    print(f"\n{'='*110}", flush=True)
    print(f"GNN ABLATION SUMMARY (cond_dim=3):", flush=True)
    print(f"{'='*110}", flush=True)
    print(summary.to_string(index=False), flush=True)

    # Add baseline for comparison
    print(f"\n{'='*110}", flush=True)
    print(f"BASELINE COMPARISON:", flush=True)
    print(f"  SetTransformer small (d=128, 3L): MSE=2.122 (synth100%, cond_dim=3)", flush=True)
    print(f"{'='*110}", flush=True)

    # Best GNN result at synth_frac=1.0
    s100 = summary[summary["synth_frac"] == 1.0]
    if len(s100) > 0:
        best = s100.iloc[0]
        print(f"\nBest GNN at synth100%: {best['config_name']} MSE={best['mean_test_mse']:.4f}", flush=True)
        delta = best['mean_test_mse'] - 2.122
        print(f"  vs SetTransformer baseline: {'+' if delta > 0 else ''}{delta:.4f}", flush=True)

    # Log summary to wandb
    srun = wandb.init(
        project=args.wandb_project,
        name="gnn_summary",
        job_type="summary",
        tags=["summary", "gnn", "cond3", "thrower_masked_button_region_span"],
        reinit=True,
        mode="offline",
    )
    wandb.log({"summary": wandb.Table(dataframe=summary)})
    wandb.log({"full_results": wandb.Table(dataframe=df)})
    art = wandb.Artifact("gnn_ablation_results", type="results")
    art.add_file(str(out_dir / "gnn_results.csv"))
    art.add_file(str(out_dir / "gnn_summary.csv"))
    srun.log_artifact(art)
    srun.finish()

    print(f"\nDone! {total} runs completed.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    base = Path(__file__).resolve().parent.parent
    data_base = base / ".." / "2026"
    ap.add_argument("--stones_csv", default=str(data_base / "Stones.csv"))
    ap.add_argument("--ends_csv", default=str(data_base / "Ends.csv"))
    ap.add_argument("--synth_stones_csv", default=str(base / "synth_terminal_stones.csv"))
    ap.add_argument("--synth_ends_csv", default=str(base / "synth_terminal_ends.csv"))
    ap.add_argument("--out_dir", default=str(Path(__file__).resolve().parent / "gnn_results"))
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--no_cuda", action="store_true")
    ap.add_argument("--gpu", type=int, default=None, help="Specific GPU index to use")
    ap.add_argument("--only_configs", nargs="*", default=None, help="Run only these config names")
    ap.add_argument("--only_holdouts", nargs="*", default=None, help="Optional subset of held-out competition IDs")
    ap.add_argument("--only_synth_fracs", nargs="*", default=None, help="Optional subset of synth fractions, e.g. 0.5")
    ap.add_argument("--wandb_project", default="curling-value-gnn")
    ap.add_argument("--wandb_api_key",
                    default="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")
    args = ap.parse_args()
    run(args)
