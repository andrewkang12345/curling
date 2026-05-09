#!/usr/bin/env python3
"""
Ablative experiments for curling value model.

Varies:
  1. Synthetic data fraction: [0%, 10%, 25%, 50%, 100%] of terminal synth data
  2. Model type: Transformer vs XGBoost
  3. Hyperparameters for each model type

Evaluation: Leave-one-competition-out cross-validation.
All runs logged to wandb. Summary table logged at end.
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

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed, skipping XGBoost experiments.")


# ─────────────────────────────────────────────────────────────
# Configuration grids
# ─────────────────────────────────────────────────────────────

SYNTH_FRACS = [0.0, 0.10, 0.25, 0.50, 1.0]

TRANSFORMER_CONFIGS = {
    "tf_small":     dict(hidden_dim=128, n_layers=3, n_heads=4, lr=3e-4, dropout=0.1,  weight_decay=1e-4),
    "tf_medium":    dict(hidden_dim=256, n_layers=4, n_heads=4, lr=2e-4, dropout=0.1,  weight_decay=1e-4),
    "tf_medium_ld": dict(hidden_dim=256, n_layers=4, n_heads=4, lr=2e-4, dropout=0.05, weight_decay=1e-4),
    "tf_large":     dict(hidden_dim=512, n_layers=6, n_heads=8, lr=1e-4, dropout=0.1,  weight_decay=1e-4),
}

XGBOOST_CONFIGS = {
    "xgb_shallow": dict(max_depth=4,  learning_rate=0.03, n_estimators=2000),
    "xgb_default": dict(max_depth=8,  learning_rate=0.03, n_estimators=2000),
    "xgb_deep":    dict(max_depth=12, learning_rate=0.03, n_estimators=2000),
    "xgb_fast":    dict(max_depth=8,  learning_rate=0.1,  n_estimators=1000),
}


# ─────────────────────────────────────────────────────────────
# Pre-materialization: convert ValueDataset to tensors once
# ─────────────────────────────────────────────────────────────

def materialize_dataset(ds):
    """Convert a ValueDataset into (X_pos, X_cond, Y, comp_ids) tensors."""
    n = len(ds)
    loader = DataLoader(ds, batch_size=8192, shuffle=False, num_workers=0)
    xs, cs, ys = [], [], []
    for x, c, y in loader:
        xs.append(x)
        cs.append(c)
        ys.append(y)
    X_pos = torch.cat(xs, dim=0)   # (N, 24)
    X_cond = torch.cat(cs, dim=0)  # (N, 4)
    Y = torch.cat(ys, dim=0)       # (N, 1)

    # Extract competition IDs for splitting
    comp_ids = pd.to_numeric(ds.df["CompetitionID"], errors="coerce").to_numpy(dtype="int64")

    return X_pos, X_cond, Y, comp_ids


def make_tensor_ds(X_pos, X_cond, Y):
    return TensorDataset(X_pos, X_cond, Y)


# ─────────────────────────────────────────────────────────────
# Transformer training
# ─────────────────────────────────────────────────────────────

def train_transformer(
    train_ds, val_ds,
    input_dim, cond_dim,
    hidden_dim, n_layers, n_heads, lr, dropout, weight_decay,
    device, epochs=150, patience=20, batch_size=1024,
):
    model = ValueTransformer(
        input_dim=input_dim, cond_dim=cond_dim,
        hidden_dim=hidden_dim, num_stones=NUM_STONES,
        n_layers=n_layers, n_heads=n_heads, dropout=dropout,
    ).to(device)

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
        val_running, val_count = 0.0, 0
        with torch.no_grad():
            for x, c, y in val_loader:
                x = x.to(device, non_blocking=True)
                c = c.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                loss = criterion(model(x, c), y)
                val_running += loss.item() * x.size(0)
                val_count += x.size(0)
        val_mse = val_running / max(1, val_count)

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
    }


# ─────────────────────────────────────────────────────────────
# XGBoost training
# ─────────────────────────────────────────────────────────────

def train_xgboost(train_ds, val_ds, max_depth, learning_rate, n_estimators):
    # Materialize to numpy
    tl = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)
    x_tr, c_tr, y_tr = next(iter(tl))
    X_train = torch.cat([x_tr, c_tr], dim=1).numpy()
    y_train = y_tr.squeeze(-1).numpy()

    vl = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
    x_va, c_va, y_va = next(iter(vl))
    X_val = torch.cat([x_va, c_va], dim=1).numpy()
    y_val = y_va.squeeze(-1).numpy()

    model = xgb.XGBRegressor(
        n_estimators=n_estimators, learning_rate=learning_rate,
        max_depth=max_depth, subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, min_child_weight=1.0, gamma=0.0,
        objective="reg:squarederror", eval_metric="rmse",
        tree_method="auto", random_state=42, n_jobs=0,
        early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    best_iter = getattr(model, "best_iteration", None)
    try:
        if best_iter is not None:
            yhat_val = model.predict(X_val, iteration_range=(0, best_iter + 1))
            yhat_train = model.predict(X_train, iteration_range=(0, best_iter + 1))
        else:
            yhat_val = model.predict(X_val)
            yhat_train = model.predict(X_train)
    except TypeError:
        yhat_val = model.predict(X_val)
        yhat_train = model.predict(X_train)

    def mse(a, b):
        return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))

    val_mse = mse(y_val, yhat_val)
    train_mse = mse(y_train, yhat_train)

    return {
        "best_val_mse": val_mse,
        "best_val_rmse": float(np.sqrt(val_mse)),
        "best_epoch": best_iter if best_iter is not None else n_estimators,
        "final_epoch": best_iter if best_iter is not None else n_estimators,
        "final_train_mse": train_mse,
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def run_ablation(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Device: {device}", flush=True)

    # ── Pre-materialize everything ──
    print("Loading and materializing real data...", flush=True)
    real_ds = ValueDataset(args.stones_csv, args.ends_csv,
                           augment_positions=False, augment_flip=False)
    real_Xp, real_Xc, real_Y, real_comps = materialize_dataset(real_ds)
    comp_ids = sorted(set(real_comps.tolist()))
    print(f"  Real: {real_Xp.shape[0]} samples, competitions={comp_ids}", flush=True)

    print("Loading and materializing synth data...", flush=True)
    synth_ds = ValueDataset(args.synth_stones_csv, args.synth_ends_csv,
                            augment_positions=False, augment_flip=False)
    synth_Xp, synth_Xc, synth_Y, _ = materialize_dataset(synth_ds)
    n_synth_total = synth_Xp.shape[0]
    print(f"  Synth: {n_synth_total} samples", flush=True)

    input_dim = real_ds.input_dim
    cond_dim = real_ds.cond_dim

    # ── Build experiment grid ──
    experiments = []
    for cname, cfg in TRANSFORMER_CONFIGS.items():
        for sf in SYNTH_FRACS:
            experiments.append({"model_type": "transformer", "config_name": cname, "synth_frac": sf, **cfg})
    if HAS_XGB:
        for cname, cfg in XGBOOST_CONFIGS.items():
            for sf in SYNTH_FRACS:
                experiments.append({"model_type": "xgboost", "config_name": cname, "synth_frac": sf, **cfg})

    total_runs = len(experiments) * len(comp_ids)
    print(f"\n{'='*80}", flush=True)
    print(f"Experiments: {len(experiments)} configs × {len(comp_ids)} folds = {total_runs} runs", flush=True)
    print(f"{'='*80}\n", flush=True)

    # Pre-compute synth subsets (indices) for each frac
    rng = np.random.default_rng(42)
    synth_perm = rng.permutation(n_synth_total)
    synth_subsets = {}
    for sf in SYNTH_FRACS:
        k = int(round(sf * n_synth_total))
        k = max(0, min(k, n_synth_total))
        synth_subsets[sf] = synth_perm[:k] if k > 0 else np.array([], dtype=np.int64)

    # Pre-compute real train/val indices per holdout competition
    fold_indices = {}
    for comp_id in comp_ids:
        train_mask = real_comps != comp_id
        val_mask = real_comps == comp_id
        fold_indices[comp_id] = (
            np.where(train_mask)[0],
            np.where(val_mask)[0],
        )

    all_results = []
    run_idx = 0

    for exp in experiments:
        model_type = exp["model_type"]
        config_name = exp["config_name"]
        synth_frac = exp["synth_frac"]
        group_name = f"{config_name}_synth{int(synth_frac*100):03d}"

        fold_mses = []

        for fold_idx, holdout_comp in enumerate(comp_ids):
            run_idx += 1
            run_name = f"{group_name}_fold{holdout_comp}"

            # Build train/val tensor datasets
            tr_idx, va_idx = fold_indices[holdout_comp]
            s_idx = synth_subsets[synth_frac]

            # Concatenate real train + synth subset
            if len(s_idx) > 0:
                tr_Xp = torch.cat([real_Xp[tr_idx], synth_Xp[s_idx]], dim=0)
                tr_Xc = torch.cat([real_Xc[tr_idx], synth_Xc[s_idx]], dim=0)
                tr_Y  = torch.cat([real_Y[tr_idx],  synth_Y[s_idx]],  dim=0)
            else:
                tr_Xp = real_Xp[tr_idx]
                tr_Xc = real_Xc[tr_idx]
                tr_Y  = real_Y[tr_idx]

            train_td = TensorDataset(tr_Xp, tr_Xc, tr_Y)
            val_td = TensorDataset(real_Xp[va_idx], real_Xc[va_idx], real_Y[va_idx])

            synth_size = len(s_idx)
            total_train = len(train_td)

            print(f"[{run_idx}/{total_runs}] {run_name} | "
                  f"train={total_train} (real={len(tr_idx)}+synth={synth_size}) | "
                  f"val={len(va_idx)}", flush=True)

            # W&B init
            wandb_cfg = {
                "model_type": model_type, "config_name": config_name,
                "synth_frac": synth_frac, "synth_size": synth_size,
                "holdout_comp": holdout_comp, "fold_idx": fold_idx,
                "real_train_size": len(tr_idx), "real_val_size": len(va_idx),
                "total_train_size": total_train,
            }
            for k, v in exp.items():
                if k not in wandb_cfg:
                    wandb_cfg[k] = v

            run = wandb.init(
                project=args.wandb_project,
                name=run_name, group=group_name, job_type=model_type,
                tags=[model_type, config_name, f"synth{int(synth_frac*100):03d}", f"fold{holdout_comp}"],
                config=wandb_cfg, reinit=True,
            )

            t0 = time.time()
            try:
                if model_type == "transformer":
                    result = train_transformer(
                        train_td, val_td,
                        input_dim=input_dim, cond_dim=cond_dim,
                        hidden_dim=exp["hidden_dim"], n_layers=exp["n_layers"],
                        n_heads=exp["n_heads"], lr=exp["lr"],
                        dropout=exp["dropout"], weight_decay=exp["weight_decay"],
                        device=device, epochs=args.tf_epochs,
                        patience=args.tf_patience, batch_size=args.tf_batch_size,
                    )
                else:
                    result = train_xgboost(
                        train_td, val_td,
                        max_depth=exp["max_depth"],
                        learning_rate=exp["learning_rate"],
                        n_estimators=exp["n_estimators"],
                    )

                elapsed = time.time() - t0
                wandb.log({
                    "test/mse": result["best_val_mse"],
                    "test/rmse": result["best_val_rmse"],
                    "train/final_mse": result["final_train_mse"],
                    "best_epoch": result["best_epoch"],
                    "elapsed_sec": elapsed,
                })
                wandb.summary["test_mse"] = result["best_val_mse"]
                wandb.summary["test_rmse"] = result["best_val_rmse"]

                print(f"  -> mse={result['best_val_mse']:.4f} rmse={result['best_val_rmse']:.4f} "
                      f"ep={result['best_epoch']}/{result['final_epoch']} {elapsed:.1f}s", flush=True)

                fold_mses.append(result["best_val_mse"])

            except Exception as e:
                elapsed = time.time() - t0
                result = {"best_val_mse": float("inf"), "best_val_rmse": float("inf"),
                          "best_epoch": -1, "final_epoch": -1, "final_train_mse": float("inf")}
                print(f"  -> ERROR: {e}", flush=True)
                wandb.log({"error": str(e)})

            run.finish()

            row = {
                "model_type": model_type, "config_name": config_name,
                "synth_frac": synth_frac, "synth_size": synth_size,
                "holdout_comp": holdout_comp,
                "test_mse": result["best_val_mse"],
                "test_rmse": result["best_val_rmse"],
                "best_epoch": result["best_epoch"],
                "final_epoch": result["final_epoch"],
                "train_mse": result.get("final_train_mse", float("inf")),
                "elapsed_sec": elapsed,
            }
            for k, v in exp.items():
                if k not in row:
                    row[k] = v
            all_results.append(row)

        # Average across folds
        if fold_mses:
            avg = np.mean(fold_mses)
            print(f"  >> {group_name} AVG mse={avg:.4f} rmse={np.sqrt(avg):.4f}", flush=True)

    # ── Save results ──
    results_df = pd.DataFrame(all_results)
    results_csv = out_dir / "ablation_results.csv"
    results_df.to_csv(results_csv, index=False)

    group_cols = ["model_type", "config_name", "synth_frac"]
    summary = results_df.groupby(group_cols, as_index=False).agg(
        mean_test_mse=("test_mse", "mean"),
        std_test_mse=("test_mse", "std"),
        mean_test_rmse=("test_rmse", "mean"),
        mean_train_mse=("train_mse", "mean"),
        mean_elapsed=("elapsed_sec", "mean"),
        n_folds=("test_mse", "count"),
    ).sort_values("mean_test_mse")

    summary_csv = out_dir / "ablation_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print(f"\n{'='*100}", flush=True)
    print("SUMMARY (sorted by mean_test_mse):", flush=True)
    print(f"{'='*100}", flush=True)
    print(summary.to_string(index=False), flush=True)

    # ── Log summary to wandb ──
    srun = wandb.init(
        project=args.wandb_project, name="ablation_summary",
        job_type="summary", tags=["summary"], reinit=True,
    )

    wandb.log({"ablation_summary": wandb.Table(dataframe=summary)})
    wandb.log({"full_results": wandb.Table(dataframe=results_df)})

    # Per-competition tables
    for cid in comp_ids:
        cdf = results_df[results_df["holdout_comp"] == cid].sort_values("test_mse")
        wandb.log({f"comp_{cid}": wandb.Table(dataframe=cdf)})

    art = wandb.Artifact("ablation_results", type="results")
    art.add_file(str(results_csv))
    art.add_file(str(summary_csv))
    srun.log_artifact(art)
    srun.finish()

    print(f"\nDone! {total_runs} runs completed. Results: {results_csv}", flush=True)
    return results_df, summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    base = Path(__file__).resolve().parent.parent
    ap.add_argument("--stones_csv",       default=str(base / ".." / "2026" / "Stones.csv"))
    ap.add_argument("--ends_csv",         default=str(base / ".." / "2026" / "Ends.csv"))
    ap.add_argument("--synth_stones_csv", default=str(base / "synth_terminal_stones.csv"))
    ap.add_argument("--synth_ends_csv",   default=str(base / "synth_terminal_ends.csv"))
    ap.add_argument("--out_dir",          default=str(Path(__file__).resolve().parent / "ablation_results"))
    ap.add_argument("--tf_epochs",    type=int, default=150)
    ap.add_argument("--tf_patience",  type=int, default=20)
    ap.add_argument("--tf_batch_size", type=int, default=1024)
    ap.add_argument("--no_cuda",      action="store_true")
    ap.add_argument("--wandb_project", default="curling-value-ablation")
    args = ap.parse_args()
    run_ablation(args)
