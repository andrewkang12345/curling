#!/usr/bin/env python3
"""
Architecture ablation: tests new architectures against the transformer baseline.

Uses synth_frac in [0.0, 0.5, 1.0] and leave-one-competition-out CV.
Logs everything to the same wandb project for comparison.
"""

from __future__ import annotations

import argparse
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
from dataset import ValueDataset, NUM_STONES
from model import ValueTransformer
from new_architectures import ARCHITECTURE_REGISTRY


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

SYNTH_FRACS = [0.0, 0.50, 1.0]

# Each architecture at two sizes (small + medium) to see scaling behavior
ARCH_CONFIGS = {
    # -- Baselines from round 1 (the winner + a reference) --
    "tf_medium_ld": dict(
        arch="original_transformer",
        hidden_dim=256, n_layers=4, n_heads=4, lr=2e-4, dropout=0.05, weight_decay=1e-4,
    ),

    # -- MLP variants --
    "mlp_small": dict(
        arch="mlp",
        hidden_dim=256, n_layers=4, lr=3e-4, dropout=0.1, weight_decay=1e-4,
    ),
    "mlp_large": dict(
        arch="mlp",
        hidden_dim=512, n_layers=6, lr=2e-4, dropout=0.1, weight_decay=1e-4,
    ),

    # -- DeepSets --
    "deepsets_small": dict(
        arch="deepsets",
        hidden_dim=128, n_layers=3, lr=3e-4, dropout=0.1, weight_decay=1e-4,
    ),
    "deepsets_medium": dict(
        arch="deepsets",
        hidden_dim=256, n_layers=4, lr=2e-4, dropout=0.1, weight_decay=1e-4,
    ),

    # -- Set Transformer (no position embeddings) --
    "settf_small": dict(
        arch="set_transformer",
        hidden_dim=128, n_layers=3, n_heads=4, lr=3e-4, dropout=0.1, weight_decay=1e-4,
    ),
    "settf_medium": dict(
        arch="set_transformer",
        hidden_dim=256, n_layers=4, n_heads=4, lr=2e-4, dropout=0.05, weight_decay=1e-4,
    ),

    # -- PairNet --
    "pairnet_small": dict(
        arch="pairnet",
        hidden_dim=128, n_layers=3, lr=3e-4, dropout=0.1, weight_decay=1e-4,
    ),
    "pairnet_medium": dict(
        arch="pairnet",
        hidden_dim=192, n_layers=4, lr=2e-4, dropout=0.1, weight_decay=1e-4,
    ),

    # -- Physics Transformer --
    "phystf_small": dict(
        arch="physics_transformer",
        hidden_dim=128, n_layers=3, n_heads=4, lr=3e-4, dropout=0.1, weight_decay=1e-4,
    ),
    "phystf_medium": dict(
        arch="physics_transformer",
        hidden_dim=256, n_layers=4, n_heads=4, lr=2e-4, dropout=0.05, weight_decay=1e-4,
    ),

    # -- ResMLP --
    "resmlp_small": dict(
        arch="resmlp",
        hidden_dim=256, n_layers=4, lr=3e-4, dropout=0.1, weight_decay=1e-4,
    ),
    "resmlp_large": dict(
        arch="resmlp",
        hidden_dim=512, n_layers=8, lr=1e-4, dropout=0.1, weight_decay=1e-4,
    ),
}


# ─────────────────────────────────────────────────────────────
# Data helpers (same as run_ablation.py)
# ─────────────────────────────────────────────────────────────

def materialize_dataset(ds):
    loader = DataLoader(ds, batch_size=8192, shuffle=False, num_workers=0)
    xs, cs, ys = [], [], []
    for x, c, y in loader:
        xs.append(x); cs.append(c); ys.append(y)
    X_pos = torch.cat(xs, 0)
    X_cond = torch.cat(cs, 0)
    Y = torch.cat(ys, 0)
    comp_ids = pd.to_numeric(ds.df["CompetitionID"], errors="coerce").to_numpy(dtype="int64")
    return X_pos, X_cond, Y, comp_ids


# ─────────────────────────────────────────────────────────────
# Generic training loop
# ─────────────────────────────────────────────────────────────

def build_model(arch, input_dim, cond_dim, **kwargs):
    """Build model from architecture name and kwargs."""
    if arch == "original_transformer":
        return ValueTransformer(
            input_dim=input_dim, cond_dim=cond_dim,
            hidden_dim=kwargs["hidden_dim"], num_stones=NUM_STONES,
            n_layers=kwargs["n_layers"], n_heads=kwargs["n_heads"],
            dropout=kwargs["dropout"],
        )
    else:
        cls = ARCHITECTURE_REGISTRY[arch]
        # Pass all kwargs; each model ignores what it doesn't need
        return cls(input_dim=input_dim, cond_dim=cond_dim, **kwargs)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(
    train_ds, val_ds,
    arch, input_dim, cond_dim,
    lr, weight_decay, device,
    epochs=150, patience=20, batch_size=1024,
    **model_kwargs,
):
    model = build_model(arch, input_dim, cond_dim, **model_kwargs).to(device)
    n_params = count_parameters(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                            num_workers=0, pin_memory=(device.type == "cuda"))

    best_val_mse = float("inf")
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running, count = 0.0, 0
        for x, c, y in train_loader:
            x = x.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x, c), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * x.size(0)
            count += x.size(0)
        train_mse = running / max(1, count)

        model.eval()
        vr, vc = 0.0, 0
        with torch.no_grad():
            for x, c, y in val_loader:
                x = x.to(device, non_blocking=True)
                c = c.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                loss = criterion(model(x, c), y)
                vr += loss.item() * x.size(0)
                vc += x.size(0)
        val_mse = vr / max(1, vc)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if patience > 0 and no_improve >= patience:
            break

    return {
        "best_val_mse": best_val_mse,
        "best_val_rmse": float(np.sqrt(best_val_mse)),
        "best_epoch": best_epoch,
        "final_epoch": epoch,
        "final_train_mse": train_mse,
        "n_params": n_params,
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Device: {device}", flush=True)

    # Load data
    print("Loading real data...", flush=True)
    real_ds = ValueDataset(args.stones_csv, args.ends_csv, augment_positions=False, augment_flip=False)
    real_Xp, real_Xc, real_Y, real_comps = materialize_dataset(real_ds)
    comp_ids = sorted(set(real_comps.tolist()))
    print(f"  Real: {real_Xp.shape[0]} samples, competitions={comp_ids}", flush=True)

    print("Loading synth data...", flush=True)
    synth_ds = ValueDataset(args.synth_stones_csv, args.synth_ends_csv, augment_positions=False, augment_flip=False)
    synth_Xp, synth_Xc, synth_Y, _ = materialize_dataset(synth_ds)
    n_synth = synth_Xp.shape[0]
    print(f"  Synth: {n_synth} samples", flush=True)

    input_dim = real_ds.input_dim
    cond_dim = real_ds.cond_dim

    # Pre-compute synth subsets
    rng = np.random.default_rng(42)
    synth_perm = rng.permutation(n_synth)
    synth_subsets = {}
    for sf in SYNTH_FRACS:
        k = min(int(round(sf * n_synth)), n_synth)
        synth_subsets[sf] = synth_perm[:k] if k > 0 else np.array([], dtype=np.int64)

    # Pre-compute fold indices
    fold_indices = {}
    for cid in comp_ids:
        fold_indices[cid] = (
            np.where(real_comps != cid)[0],
            np.where(real_comps == cid)[0],
        )

    # Build experiment grid
    experiments = []
    for cname, cfg in ARCH_CONFIGS.items():
        for sf in SYNTH_FRACS:
            experiments.append({"config_name": cname, "synth_frac": sf, **cfg})

    total_runs = len(experiments) * len(comp_ids)
    print(f"\n{'='*80}", flush=True)
    print(f"{len(experiments)} configs × {len(comp_ids)} folds = {total_runs} runs", flush=True)
    print(f"{'='*80}\n", flush=True)

    all_results = []
    run_idx = 0

    for exp in experiments:
        config_name = exp["config_name"]
        arch = exp["arch"]
        synth_frac = exp["synth_frac"]
        group_name = f"{config_name}_synth{int(synth_frac*100):03d}"

        fold_mses = []

        for fold_idx, holdout_comp in enumerate(comp_ids):
            run_idx += 1
            run_name = f"{group_name}_fold{holdout_comp}"

            tr_idx, va_idx = fold_indices[holdout_comp]
            s_idx = synth_subsets[synth_frac]

            if len(s_idx) > 0:
                tr_Xp = torch.cat([real_Xp[tr_idx], synth_Xp[s_idx]], 0)
                tr_Xc = torch.cat([real_Xc[tr_idx], synth_Xc[s_idx]], 0)
                tr_Y  = torch.cat([real_Y[tr_idx],  synth_Y[s_idx]],  0)
            else:
                tr_Xp, tr_Xc, tr_Y = real_Xp[tr_idx], real_Xc[tr_idx], real_Y[tr_idx]

            train_td = TensorDataset(tr_Xp, tr_Xc, tr_Y)
            val_td = TensorDataset(real_Xp[va_idx], real_Xc[va_idx], real_Y[va_idx])

            synth_size = len(s_idx)
            print(f"[{run_idx}/{total_runs}] {run_name} | arch={arch} | "
                  f"train={len(train_td)} | val={len(va_idx)}", flush=True)

            wandb_cfg = {
                "arch": arch, "config_name": config_name,
                "synth_frac": synth_frac, "synth_size": synth_size,
                "holdout_comp": holdout_comp, "fold_idx": fold_idx,
                "real_train_size": len(tr_idx), "real_val_size": len(va_idx),
                "total_train_size": len(train_td),
            }
            for k, v in exp.items():
                if k not in wandb_cfg:
                    wandb_cfg[k] = v

            run = wandb.init(
                project=args.wandb_project,
                name=run_name, group=group_name, job_type=arch,
                tags=[arch, config_name, f"synth{int(synth_frac*100):03d}", f"fold{holdout_comp}", "arch_ablation"],
                config=wandb_cfg, reinit=True,
            )

            # Filter model kwargs (remove non-model keys)
            model_kwargs = {k: v for k, v in exp.items()
                           if k not in ("config_name", "synth_frac", "arch", "lr", "weight_decay")}

            t0 = time.time()
            try:
                result = train_model(
                    train_td, val_td,
                    arch=arch, input_dim=input_dim, cond_dim=cond_dim,
                    lr=exp["lr"], weight_decay=exp["weight_decay"],
                    device=device, epochs=args.epochs, patience=args.patience,
                    batch_size=args.batch_size,
                    **model_kwargs,
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

                print(f"  -> mse={result['best_val_mse']:.4f} rmse={result['best_val_rmse']:.4f} "
                      f"ep={result['best_epoch']}/{result['final_epoch']} "
                      f"params={result['n_params']:,} {elapsed:.1f}s", flush=True)

                fold_mses.append(result["best_val_mse"])

            except Exception as e:
                import traceback
                traceback.print_exc()
                elapsed = time.time() - t0
                result = {"best_val_mse": float("inf"), "best_val_rmse": float("inf"),
                          "best_epoch": -1, "final_epoch": -1, "final_train_mse": float("inf"),
                          "n_params": 0}
                print(f"  -> ERROR: {e}", flush=True)
                wandb.log({"error": str(e)})

            run.finish()

            row = {
                "arch": arch, "config_name": config_name,
                "synth_frac": synth_frac, "synth_size": synth_size,
                "holdout_comp": holdout_comp,
                "test_mse": result["best_val_mse"],
                "test_rmse": result["best_val_rmse"],
                "best_epoch": result["best_epoch"],
                "final_epoch": result["final_epoch"],
                "train_mse": result.get("final_train_mse", float("inf")),
                "n_params": result.get("n_params", 0),
                "elapsed_sec": elapsed,
            }
            for k, v in exp.items():
                if k not in row:
                    row[k] = v
            all_results.append(row)

        if fold_mses:
            avg = np.mean(fold_mses)
            print(f"  >> {group_name} AVG mse={avg:.4f} rmse={np.sqrt(avg):.4f}", flush=True)

    # Save results
    results_df = pd.DataFrame(all_results)
    results_csv = out_dir / "arch_ablation_results.csv"
    results_df.to_csv(results_csv, index=False)

    group_cols = ["arch", "config_name", "synth_frac"]
    summary = results_df.groupby(group_cols, as_index=False).agg(
        mean_test_mse=("test_mse", "mean"),
        std_test_mse=("test_mse", "std"),
        mean_test_rmse=("test_rmse", "mean"),
        mean_train_mse=("train_mse", "mean"),
        n_params=("n_params", "first"),
        mean_elapsed=("elapsed_sec", "mean"),
        n_folds=("test_mse", "count"),
    ).sort_values("mean_test_mse")

    summary_csv = out_dir / "arch_ablation_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print(f"\n{'='*110}", flush=True)
    print("ARCHITECTURE ABLATION SUMMARY (sorted by mean_test_mse):", flush=True)
    print(f"{'='*110}", flush=True)
    print(summary.to_string(index=False), flush=True)

    # Log to wandb
    srun = wandb.init(
        project=args.wandb_project, name="arch_ablation_summary",
        job_type="summary", tags=["summary", "arch_ablation"], reinit=True,
    )
    wandb.log({"arch_summary": wandb.Table(dataframe=summary)})
    wandb.log({"arch_full_results": wandb.Table(dataframe=results_df)})

    art = wandb.Artifact("arch_ablation_results", type="results")
    art.add_file(str(results_csv))
    art.add_file(str(summary_csv))
    srun.log_artifact(art)
    srun.finish()

    print(f"\nDone! {total_runs} runs. Results: {results_csv}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    base = Path(__file__).resolve().parent.parent
    ap.add_argument("--stones_csv",       default=str(base / ".." / "2026" / "Stones.csv"))
    ap.add_argument("--ends_csv",         default=str(base / ".." / "2026" / "Ends.csv"))
    ap.add_argument("--synth_stones_csv", default=str(base / "synth_terminal_stones.csv"))
    ap.add_argument("--synth_ends_csv",   default=str(base / "synth_terminal_ends.csv"))
    ap.add_argument("--out_dir",          default=str(Path(__file__).resolve().parent / "arch_ablation_results"))
    ap.add_argument("--epochs",       type=int, default=150)
    ap.add_argument("--patience",     type=int, default=20)
    ap.add_argument("--batch_size",   type=int, default=1024)
    ap.add_argument("--no_cuda",      action="store_true")
    ap.add_argument("--wandb_project", default="curling-value-ablation")
    args = ap.parse_args()
    run(args)
