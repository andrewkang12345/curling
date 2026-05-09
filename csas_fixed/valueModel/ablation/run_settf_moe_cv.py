#!/usr/bin/env python3
"""
Cross-validation runner for the MoE SetTransformer value model.

Uses leave-one-competition-out evaluation, mirroring the earlier architecture
ablation protocol for SetTransformer medium.
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

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(THIS_DIR))

from dataset import ValueDataset
from new_architectures import ARCHITECTURE_REGISTRY


CONFIG = dict(
    arch="set_transformer_moe",
    hidden_dim=256,
    n_layers=4,
    n_heads=4,
    n_experts=4,
    lr=2e-4,
    dropout=0.05,
    weight_decay=1e-4,
    synth_frac=1.0,
)


def materialize_dataset(ds: ValueDataset):
    loader = DataLoader(ds, batch_size=8192, shuffle=False, num_workers=0)
    xs, cs, ys = [], [], []
    for x, c, y in loader:
        xs.append(x)
        cs.append(c)
        ys.append(y)
    return (
        torch.cat(xs, 0),
        torch.cat(cs, 0),
        torch.cat(ys, 0),
        pd.to_numeric(ds.df["CompetitionID"], errors="coerce").to_numpy(dtype="int64"),
    )


def build_model(input_dim: int, cond_dim: int):
    return ARCHITECTURE_REGISTRY[CONFIG["arch"]](
        input_dim=input_dim,
        cond_dim=cond_dim,
        hidden_dim=CONFIG["hidden_dim"],
        n_layers=CONFIG["n_layers"],
        n_heads=CONFIG["n_heads"],
        n_experts=CONFIG["n_experts"],
        dropout=CONFIG["dropout"],
    )


def train_model(train_td, val_td, device, epochs=150, patience=20, batch_size=1024):
    model = build_model(train_td.tensors[0].shape[1], train_td.tensors[1].shape[1]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    criterion = nn.MSELoss()

    train_loader = DataLoader(train_td, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_td, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    best_val, best_epoch, no_imp = float("inf"), 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_sum, train_count = 0.0, 0
        for x, c, y in train_loader:
            x = x.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x, c), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_sum += loss.item() * x.size(0)
            train_count += x.size(0)
        train_mse = train_sum / max(1, train_count)

        model.eval()
        val_sum, val_count = 0.0, 0
        with torch.no_grad():
            for x, c, y in val_loader:
                x = x.to(device, non_blocking=True)
                c = c.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                loss = criterion(model(x, c), y)
                val_sum += loss.item() * x.size(0)
                val_count += x.size(0)
        val_mse = val_sum / max(1, val_count)

        wandb.log({"epoch": epoch, "train/mse": train_mse, "val/mse": val_mse})

        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            no_imp = 0
        else:
            no_imp += 1
        if patience > 0 and no_imp >= patience:
            break

    return {
        "best_val_mse": best_val,
        "best_val_rmse": float(np.sqrt(best_val)),
        "best_epoch": best_epoch,
        "final_epoch": epoch,
        "final_train_mse": train_mse,
        "n_params": n_params,
    }


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
    elif torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}", flush=True)

    real_ds = ValueDataset(args.stones_csv, args.ends_csv, augment_positions=False, augment_flip=False)
    real_Xp, real_Xc, real_Y, real_comps = materialize_dataset(real_ds)
    comp_ids = sorted(set(real_comps.tolist()))
    if args.only_holdouts:
        allowed = {int(x) for x in args.only_holdouts}
        comp_ids = [c for c in comp_ids if c in allowed]
    print(f"Real samples: {real_Xp.shape[0]}, holdouts={comp_ids}", flush=True)

    synth_ds = ValueDataset(args.synth_stones_csv, args.synth_ends_csv, augment_positions=False, augment_flip=False)
    synth_Xp, synth_Xc, synth_Y, _ = materialize_dataset(synth_ds)
    print(f"Synth samples: {synth_Xp.shape[0]}", flush=True)

    rng = np.random.default_rng(42)
    synth_perm = rng.permutation(synth_Xp.shape[0])
    take = min(int(round(CONFIG["synth_frac"] * synth_Xp.shape[0])), synth_Xp.shape[0])
    synth_idx = synth_perm[:take]

    results = []
    for holdout in comp_ids:
        tr_idx = np.where(real_comps != holdout)[0]
        va_idx = np.where(real_comps == holdout)[0]
        tr_Xp = torch.cat([real_Xp[tr_idx], synth_Xp[synth_idx]], dim=0)
        tr_Xc = torch.cat([real_Xc[tr_idx], synth_Xc[synth_idx]], dim=0)
        tr_Y = torch.cat([real_Y[tr_idx], synth_Y[synth_idx]], dim=0)

        train_td = TensorDataset(tr_Xp, tr_Xc, tr_Y)
        val_td = TensorDataset(real_Xp[va_idx], real_Xc[va_idx], real_Y[va_idx])

        run_name = f"settf_moe_medium_s100_f{holdout}"
        run_handle = wandb.init(
            project=args.wandb_project,
            name=run_name,
            group="settf_moe_medium_s100",
            job_type="set_transformer_moe",
            tags=["set_transformer_moe", "settf_moe_medium", "s100", "cond3"],
            config={**CONFIG, "holdout_comp": holdout, "cond_dim": real_ds.cond_dim},
            reinit=True,
            mode="offline",
        )

        t0 = time.time()
        result = train_model(
            train_td,
            val_td,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
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
        run_handle.finish()

        print(
            f"{holdout}: mse={result['best_val_mse']:.4f} ep={result['best_epoch']}/{result['final_epoch']} "
            f"params={result['n_params']:,} {elapsed:.1f}s",
            flush=True,
        )
        results.append({
            "arch": CONFIG["arch"],
            "config_name": "settf_moe_medium",
            "synth_frac": CONFIG["synth_frac"],
            "holdout_comp": holdout,
            "test_mse": result["best_val_mse"],
            "test_rmse": result["best_val_rmse"],
            "best_epoch": result["best_epoch"],
            "n_params": result["n_params"],
            "elapsed_sec": elapsed,
        })

    df = pd.DataFrame(results).sort_values("holdout_comp")
    df.to_csv(out_dir / "moe_results.csv", index=False)
    summary = pd.DataFrame([{
        "config_name": "settf_moe_medium",
        "synth_frac": CONFIG["synth_frac"],
        "mean_test_mse": float(df["test_mse"].mean()),
        "std_test_mse": float(df["test_mse"].std(ddof=1)),
        "mean_test_rmse": float(df["test_rmse"].mean()),
        "n_params": int(df["n_params"].iloc[0]),
        "mean_elapsed": float(df["elapsed_sec"].mean()),
    }])
    summary.to_csv(out_dir / "moe_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    base = Path(__file__).resolve().parent.parent
    data_base = base / ".." / "2026"
    ap.add_argument("--stones_csv", default=str(data_base / "Stones.csv"))
    ap.add_argument("--ends_csv", default=str(data_base / "Ends.csv"))
    ap.add_argument("--synth_stones_csv", default=str(base / "synth_terminal_stones.csv"))
    ap.add_argument("--synth_ends_csv", default=str(base / "synth_terminal_ends.csv"))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--only_holdouts", nargs="*", default=None)
    ap.add_argument("--wandb_project", default="curling-value-settf-moe")
    ap.add_argument("--no_cuda", action="store_true")
    args = ap.parse_args()
    run(args)
