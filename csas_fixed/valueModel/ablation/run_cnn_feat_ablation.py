#!/usr/bin/env python3
"""
CNN + Feature Engineering ablation.

Tests CNN architectures (grid-based, 1D) and feature-engineered models.
Also tests XGBoost with curling features vs raw features.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import ValueDataset, NUM_STONES
from model import ValueTransformer
from cnn_and_features import (
    CNN_REGISTRY, compute_curling_features, CURLING_FEAT_DIM,
)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

SYNTH_FRACS = [0.0, 0.50, 1.0]

CONFIGS = {
    # ── CNN variants ──
    "gridcnn_32": dict(arch="grid_cnn", grid_size=32, hidden_dim=128, n_conv_layers=4, lr=3e-4, dropout=0.1, weight_decay=1e-4),
    "gridcnn_48": dict(arch="grid_cnn", grid_size=48, hidden_dim=128, n_conv_layers=4, lr=3e-4, dropout=0.1, weight_decay=1e-4),
    "finecnn_48": dict(arch="fine_cnn", grid_size=48, hidden_dim=128, n_conv_layers=5, lr=3e-4, dropout=0.1, weight_decay=1e-4),
    "finecnn_64": dict(arch="fine_cnn", grid_size=64, hidden_dim=256, n_conv_layers=5, lr=2e-4, dropout=0.1, weight_decay=1e-4),
    "cnn1d_small": dict(arch="cnn_1d", hidden_dim=128, n_conv_layers=3, lr=3e-4, dropout=0.1, weight_decay=1e-4),
    "cnn1d_medium": dict(arch="cnn_1d", hidden_dim=256, n_conv_layers=4, lr=2e-4, dropout=0.1, weight_decay=1e-4),

    # ── Feature-engineered models ──
    "feat_mlp_small": dict(arch="feat_mlp", hidden_dim=256, n_layers=4, lr=3e-4, dropout=0.1, weight_decay=1e-4),
    "feat_mlp_large": dict(arch="feat_mlp", hidden_dim=512, n_layers=6, lr=2e-4, dropout=0.1, weight_decay=1e-4),
    "feat_tf_small": dict(arch="feat_transformer", hidden_dim=128, n_layers=3, n_heads=4, lr=3e-4, dropout=0.1, weight_decay=1e-4),
    "feat_tf_medium": dict(arch="feat_transformer", hidden_dim=256, n_layers=4, n_heads=4, lr=2e-4, dropout=0.05, weight_decay=1e-4),

    # ── XGBoost with curling features ──
    "xgb_feat_shallow": dict(arch="xgb_features", max_depth=4, learning_rate=0.03, n_estimators=2000),
    "xgb_feat_default": dict(arch="xgb_features", max_depth=8, learning_rate=0.03, n_estimators=2000),
    "xgb_raw_default": dict(arch="xgb_raw", max_depth=8, learning_rate=0.03, n_estimators=2000),
}


# ─────────────────────────────────────────────────────────────
# Helpers
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


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────
# Neural net training
# ─────────────────────────────────────────────────────────────

def train_nn(train_ds, val_ds, arch, input_dim, cond_dim, lr, weight_decay,
             device, epochs=150, patience=20, batch_size=1024, **model_kwargs):
    cls = CNN_REGISTRY[arch]
    model = cls(input_dim=input_dim, cond_dim=cond_dim, **model_kwargs).to(device)
    n_params = count_params(model)

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
            x, c, y = x.to(device, non_blocking=True), c.to(device, non_blocking=True), y.to(device, non_blocking=True)
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
                x, c, y = x.to(device, non_blocking=True), c.to(device, non_blocking=True), y.to(device, non_blocking=True)
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
        "best_val_mse": best_val_mse, "best_val_rmse": float(np.sqrt(best_val_mse)),
        "best_epoch": best_epoch, "final_epoch": epoch,
        "final_train_mse": train_mse, "n_params": n_params,
    }


# ─────────────────────────────────────────────────────────────
# XGBoost training (with/without curling features)
# ─────────────────────────────────────────────────────────────

def materialize_xgb(ds, use_curling_features=True):
    loader = DataLoader(ds, batch_size=8192, shuffle=False, num_workers=0)
    Xs, ys = [], []
    for x, c, y in loader:
        if use_curling_features:
            feats = compute_curling_features(x)
            row = torch.cat([x, c, feats], dim=1)
        else:
            row = torch.cat([x, c], dim=1)
        Xs.append(row.numpy())
        ys.append(y.squeeze(-1).numpy())
    return np.concatenate(Xs, 0), np.concatenate(ys, 0)


def train_xgb(train_ds, val_ds, max_depth, learning_rate, n_estimators, use_curling_features):
    X_tr, y_tr = materialize_xgb(train_ds, use_curling_features)
    X_va, y_va = materialize_xgb(val_ds, use_curling_features)

    model = xgb.XGBRegressor(
        n_estimators=n_estimators, learning_rate=learning_rate,
        max_depth=max_depth, subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, min_child_weight=1.0, gamma=0.0,
        objective="reg:squarederror", eval_metric="rmse",
        tree_method="auto", random_state=42, n_jobs=0,
        early_stopping_rounds=50,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    best_iter = getattr(model, "best_iteration", None)
    try:
        if best_iter is not None:
            yhat_va = model.predict(X_va, iteration_range=(0, best_iter + 1))
            yhat_tr = model.predict(X_tr, iteration_range=(0, best_iter + 1))
        else:
            yhat_va = model.predict(X_va)
            yhat_tr = model.predict(X_tr)
    except TypeError:
        yhat_va = model.predict(X_va)
        yhat_tr = model.predict(X_tr)

    def mse(a, b):
        return float(np.mean((a - b) ** 2))

    n_feats = X_tr.shape[1]
    return {
        "best_val_mse": mse(y_va, yhat_va),
        "best_val_rmse": float(np.sqrt(mse(y_va, yhat_va))),
        "best_epoch": best_iter or n_estimators,
        "final_epoch": best_iter or n_estimators,
        "final_train_mse": mse(y_tr, yhat_tr),
        "n_params": n_feats,  # store input feature count
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Device: {device}", flush=True)

    print("Loading data...", flush=True)
    real_ds = ValueDataset(args.stones_csv, args.ends_csv, augment_positions=False, augment_flip=False)
    real_Xp, real_Xc, real_Y, real_comps = materialize_dataset(real_ds)
    comp_ids = sorted(set(real_comps.tolist()))
    print(f"  Real: {real_Xp.shape[0]}, comps={comp_ids}", flush=True)

    synth_ds = ValueDataset(args.synth_stones_csv, args.synth_ends_csv, augment_positions=False, augment_flip=False)
    synth_Xp, synth_Xc, synth_Y, _ = materialize_dataset(synth_ds)
    n_synth = synth_Xp.shape[0]
    print(f"  Synth: {n_synth}", flush=True)

    input_dim = real_ds.input_dim
    cond_dim = real_ds.cond_dim

    # Pre-compute subsets
    rng = np.random.default_rng(42)
    synth_perm = rng.permutation(n_synth)
    synth_subsets = {}
    for sf in SYNTH_FRACS:
        k = min(int(round(sf * n_synth)), n_synth)
        synth_subsets[sf] = synth_perm[:k] if k > 0 else np.array([], dtype=np.int64)

    fold_indices = {cid: (np.where(real_comps != cid)[0], np.where(real_comps == cid)[0]) for cid in comp_ids}

    # Build grid
    experiments = []
    for cname, cfg in CONFIGS.items():
        for sf in SYNTH_FRACS:
            experiments.append({"config_name": cname, "synth_frac": sf, **cfg})

    total = len(experiments) * len(comp_ids)
    print(f"\n{'='*80}\n{len(experiments)} configs × {len(comp_ids)} folds = {total} runs\n{'='*80}\n", flush=True)

    all_results = []
    run_idx = 0

    for exp in experiments:
        cname = exp["config_name"]
        arch = exp["arch"]
        sf = exp["synth_frac"]
        group = f"{cname}_synth{int(sf*100):03d}"
        fold_mses = []

        for fi, hcomp in enumerate(comp_ids):
            run_idx += 1
            rname = f"{group}_fold{hcomp}"

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

            print(f"[{run_idx}/{total}] {rname} | arch={arch} | train={len(train_td)} | val={len(va_idx)}", flush=True)

            wcfg = {"arch": arch, "config_name": cname, "synth_frac": sf,
                     "synth_size": len(s_idx), "holdout_comp": hcomp,
                     "real_train_size": len(tr_idx), "real_val_size": len(va_idx),
                     "total_train_size": len(train_td)}
            for k, v in exp.items():
                if k not in wcfg:
                    wcfg[k] = v

            run = wandb.init(
                project=args.wandb_project, name=rname, group=group, job_type=arch,
                tags=[arch, cname, f"synth{int(sf*100):03d}", f"fold{hcomp}", "cnn_feat"],
                config=wcfg, reinit=True,
            )

            t0 = time.time()
            try:
                if arch.startswith("xgb_"):
                    use_feats = (arch == "xgb_features")
                    result = train_xgb(
                        train_td, val_td,
                        max_depth=exp["max_depth"],
                        learning_rate=exp["learning_rate"],
                        n_estimators=exp["n_estimators"],
                        use_curling_features=use_feats,
                    )
                else:
                    mkw = {k: v for k, v in exp.items()
                           if k not in ("config_name", "synth_frac", "arch", "lr", "weight_decay")}
                    result = train_nn(
                        train_td, val_td,
                        arch=arch, input_dim=input_dim, cond_dim=cond_dim,
                        lr=exp["lr"], weight_decay=exp["weight_decay"],
                        device=device, epochs=args.epochs, patience=args.patience,
                        batch_size=args.batch_size, **mkw,
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
                wandb.summary.update({"test_mse": result["best_val_mse"], "test_rmse": result["best_val_rmse"]})
                print(f"  -> mse={result['best_val_mse']:.4f} rmse={result['best_val_rmse']:.4f} "
                      f"ep={result['best_epoch']}/{result['final_epoch']} "
                      f"params={result['n_params']:,} {elapsed:.1f}s", flush=True)
                fold_mses.append(result["best_val_mse"])

            except Exception as e:
                import traceback; traceback.print_exc()
                elapsed = time.time() - t0
                result = {"best_val_mse": float("inf"), "best_val_rmse": float("inf"),
                          "best_epoch": -1, "final_epoch": -1, "final_train_mse": float("inf"), "n_params": 0}
                print(f"  -> ERROR: {e}", flush=True)

            run.finish()

            row = {"arch": arch, "config_name": cname, "synth_frac": sf,
                   "synth_size": len(s_idx), "holdout_comp": hcomp,
                   "test_mse": result["best_val_mse"], "test_rmse": result["best_val_rmse"],
                   "best_epoch": result["best_epoch"], "final_epoch": result["final_epoch"],
                   "train_mse": result.get("final_train_mse", float("inf")),
                   "n_params": result.get("n_params", 0), "elapsed_sec": elapsed}
            all_results.append(row)

        if fold_mses:
            avg = np.mean(fold_mses)
            print(f"  >> {group} AVG mse={avg:.4f} rmse={np.sqrt(avg):.4f}", flush=True)

    # Save
    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / "cnn_feat_results.csv", index=False)

    summary = df.groupby(["arch", "config_name", "synth_frac"], as_index=False).agg(
        mean_test_mse=("test_mse", "mean"), std_test_mse=("test_mse", "std"),
        mean_test_rmse=("test_rmse", "mean"), mean_train_mse=("train_mse", "mean"),
        n_params=("n_params", "first"), mean_elapsed=("elapsed_sec", "mean"),
        n_folds=("test_mse", "count"),
    ).sort_values("mean_test_mse")
    summary.to_csv(out_dir / "cnn_feat_summary.csv", index=False)

    print(f"\n{'='*110}\nCNN + FEATURE ENGINEERING SUMMARY:\n{'='*110}", flush=True)
    print(summary.to_string(index=False), flush=True)

    srun = wandb.init(project=args.wandb_project, name="cnn_feat_summary",
                      job_type="summary", tags=["summary", "cnn_feat"], reinit=True)
    wandb.log({"cnn_feat_summary": wandb.Table(dataframe=summary)})
    wandb.log({"cnn_feat_full": wandb.Table(dataframe=df)})
    art = wandb.Artifact("cnn_feat_results", type="results")
    art.add_file(str(out_dir / "cnn_feat_results.csv"))
    art.add_file(str(out_dir / "cnn_feat_summary.csv"))
    srun.log_artifact(art)
    srun.finish()

    print(f"\nDone! {total} runs. Results: {out_dir}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    base = Path(__file__).resolve().parent.parent
    ap.add_argument("--stones_csv",       default=str(base / ".." / "2026" / "Stones.csv"))
    ap.add_argument("--ends_csv",         default=str(base / ".." / "2026" / "Ends.csv"))
    ap.add_argument("--synth_stones_csv", default=str(base / "synth_terminal_stones.csv"))
    ap.add_argument("--synth_ends_csv",   default=str(base / "synth_terminal_ends.csv"))
    ap.add_argument("--out_dir",          default=str(Path(__file__).resolve().parent / "cnn_feat_results"))
    ap.add_argument("--epochs",       type=int, default=150)
    ap.add_argument("--patience",     type=int, default=20)
    ap.add_argument("--batch_size",   type=int, default=1024)
    ap.add_argument("--no_cuda",      action="store_true")
    ap.add_argument("--wandb_project", default="curling-value-ablation")
    args = ap.parse_args()
    run(args)
