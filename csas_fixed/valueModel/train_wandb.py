#!/usr/bin/env python3
"""
train_value_wandb.py

W&B-runnable trainer for the curling ValueTransformer.

What this adds vs your script:
  - wandb.init + config capture from CLI args
  - logs train/val loss, LR, epoch time, grad norm
  - optional model watching (off by default)
  - saves best checkpoint + uploads it as a W&B artifact
  - resumes from checkpoint (and optionally resumes W&B run)

Usage examples (see commands section below).
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

import wandb

from dataset import ValueDataset, NUM_STONES
from model import ValueTransformer
from split_utils import END_KEY, make_train_val_test_indices, write_test_shot_keys


def _log(msg: str, log_file: Optional[Path]):
    print(msg)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a") as f:
            f.write(msg + "\n")


def save_ckpt(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def train(args):
    # -----------------
    # Logging
    # -----------------
    log_path = Path(args.log_file) if args.log_file else None
    out_path = Path(args.out)

    # -----------------
    # W&B init
    # -----------------
    wandb_mode = "offline" if args.wandb_offline else "online"
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity if args.wandb_entity else None,
        name=args.wandb_name if args.wandb_name else None,
        group=args.wandb_group if args.wandb_group else None,
        job_type=args.wandb_job_type,
        tags=args.wandb_tags.split(",") if args.wandb_tags else None,
        notes=args.wandb_notes if args.wandb_notes else None,
        mode=wandb_mode,
        config=vars(args),
        resume="allow" if args.wandb_resume else None,
    )

    # -----------------
    # Dataset
    # -----------------
    dataset = ValueDataset(args.stones_csv, args.ends_csv)
    _log(f"Dataset size: {len(dataset)}", log_path)
    _log(
        f"input_dim={dataset.input_dim}, cond_dim={dataset.cond_dim}, value_dim=1, num_tasks={dataset.num_tasks}",
        log_path,
    )

    group_keys = dataset.df[END_KEY].to_numpy()
    train_idx, val_idx, test_idx = make_train_val_test_indices(
        n=len(dataset),
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.split_seed,
        group_keys=group_keys,
    )
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx) if len(val_idx) > 0 else None
    _log(f"Split sizes | train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}", log_path)

    test_keys_path = None
    if args.test_keys_out:
        test_keys_path, n_test_keys = write_test_shot_keys(dataset.df, test_idx, args.test_keys_out)
        _log(
            f"Held-out test shot keys written to {test_keys_path} (rows={n_test_keys}).",
            log_path,
        )

    # Optionally log dataset metadata to W&B
    wandb.config.update(
        {
            "dataset_size": len(dataset),
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
            "input_dim": dataset.input_dim,
            "cond_dim": dataset.cond_dim,
            "num_tasks": dataset.num_tasks,
            "test_keys_csv": str(test_keys_path) if test_keys_path is not None else "",
        },
        allow_val_change=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        if val_ds is not None
        else None
    )

    # -----------------
    # Device
    # -----------------
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    _log(f"Using device: {device}", log_path)
    wandb.config.update({"device": str(device)}, allow_val_change=True)

    # -----------------
    # Model
    # -----------------
    model = ValueTransformer(
        input_dim=dataset.input_dim,
        cond_dim=dataset.cond_dim,
        hidden_dim=args.hidden_dim,
        num_stones=NUM_STONES,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(device)

    if args.wandb_watch:
        # gradient + parameter logging can be heavy; keep it optional
        wandb.watch(model, log="gradients", log_freq=args.wandb_watch_freq)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # -----------------
    # Resume checkpoint
    # -----------------
    start_epoch = 1
    best_loss = float("inf")

    if args.resume and os.path.exists(args.resume):
        _log(f"Resuming from checkpoint: {args.resume}", log_path)
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
        if "best_loss" in ckpt:
            best_loss = float(ckpt["best_loss"])
    elif args.resume:
        _log(f"WARNING: --resume path {args.resume} does not exist. Starting from scratch.", log_path)

    _log(
        f"Starting from epoch {start_epoch} / {args.epochs}. Using val_split={args.val_split}, test_split={args.test_split}.",
        log_path,
    )
    _log(
        f"Will check for saving every {args.save_every} epochs (and at final), saving only if loss improves.",
        log_path,
    )

    # -----------------
    # Train loop
    # -----------------
    global_step = 0

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_count = 0

        # For grad norm logging (last batch of epoch)
        last_grad_norm = float("nan")

        for x, c, y in train_loader:
            x = x.to(device)
            c = c.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x, c)
            loss = criterion(pred, y)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
            last_grad_norm = float(grad_norm.detach().cpu().item()) if torch.isfinite(grad_norm) else float("nan")

            optimizer.step()

            bs = x.size(0)
            running_loss += float(loss.item()) * bs
            running_count += bs

            # optional per-step logging (default off)
            if args.log_every_steps > 0 and (global_step % args.log_every_steps == 0):
                wandb.log(
                    {
                        "train/loss_step": float(loss.item()),
                        "train/grad_norm_step": last_grad_norm,
                        "train/lr": float(optimizer.param_groups[0]["lr"]),
                        "global_step": global_step,
                        "epoch": epoch,
                    },
                    step=global_step,
                )

            global_step += 1

        train_loss = running_loss / max(1, running_count)

        # Validation
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_running = 0.0
            val_count = 0
            with torch.no_grad():
                for x, c, y in val_loader:
                    x = x.to(device)
                    c = c.to(device)
                    y = y.to(device)
                    pred = model(x, c)
                    v_loss = criterion(pred, y)
                    bs = x.size(0)
                    val_running += float(v_loss.item()) * bs
                    val_count += bs
            val_loss = val_running / max(1, val_count)
            current_metric = float(val_loss)
            _log(
                f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | best_loss={best_loss:.6f}",
                log_path,
            )
        else:
            current_metric = float(train_loss)
            _log(
                f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | best_loss={best_loss:.6f}",
                log_path,
            )

        epoch_time = time.time() - t0

        # Epoch-level W&B log
        metrics = {
            "epoch": epoch,
            "train/loss": float(train_loss),
            "train/lr": float(optimizer.param_groups[0]["lr"]),
            "train/grad_norm": float(last_grad_norm),
            "time/epoch_sec": float(epoch_time),
            "best/loss": float(best_loss),
        }
        if val_loss is not None:
            metrics["val/loss"] = float(val_loss)

        wandb.log(metrics, step=global_step)

        # Save best checkpoint periodically (and always at final)
        is_last = (epoch == args.epochs)
        should_check = (epoch % args.save_every == 0) or is_last

        if should_check and current_metric < best_loss:
            best_loss = float(current_metric)

            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "input_dim": dataset.input_dim,
                "cond_dim": dataset.cond_dim,
                "hidden_dim": args.hidden_dim,
                "num_stones": NUM_STONES,
                "num_tasks": dataset.num_tasks,
                "best_loss": best_loss,
                "test_keys_csv": str(test_keys_path) if test_keys_path is not None else "",
                "args": vars(args),
            }
            save_ckpt(out_path, ckpt)
            _log(f"Saved NEW BEST checkpoint to {out_path} (epoch {epoch}, metric={current_metric:.6f})", log_path)

            # Upload as artifact
            art = wandb.Artifact(
                name=args.wandb_artifact_name,
                type="model",
                metadata={
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "input_dim": dataset.input_dim,
                    "cond_dim": dataset.cond_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_stones": NUM_STONES,
                },
            )
            art.add_file(str(out_path))
            run.log_artifact(art)

    _log(
        f"Training finished. Best metric (val if available, else train): {best_loss:.6f}. Best checkpoint at {out_path}.",
        log_path,
    )

    # Always save the final model file as a run file for convenience
    if out_path.exists():
        wandb.save(str(out_path))

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer value network for curling end outcomes (W&B).")

    # Data
    parser.add_argument("--stones_csv", type=str, default="../2026/Stones.csv")
    parser.add_argument("--ends_csv", type=str, default="../2026/Ends.csv")

    # Outputs
    parser.add_argument("--out", type=str, default="value_model.pt")
    parser.add_argument("--log_file", type=str, default="train.log")
    parser.add_argument("--resume", type=str, default="")

    # Train hyperparams
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1, help="Held-out test fraction. Not used for training or validation.")
    parser.add_argument("--test_keys_out", type=str, default="value_test_shots.csv", help="CSV path for held-out test shot keys.")
    parser.add_argument("--split_seed", type=int, default=123)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every_steps", type=int, default=0, help="If >0, logs step metrics every N optimizer steps.")

    # W&B
    parser.add_argument("--wandb_project", type=str, default="curling-value")
    parser.add_argument("--wandb_entity", type=str, default="", help="Optional: your W&B entity/team")
    parser.add_argument("--wandb_name", type=str, default="", help="Optional: run name")
    parser.add_argument("--wandb_group", type=str, default="", help="Optional: group name")
    parser.add_argument("--wandb_job_type", type=str, default="train")
    parser.add_argument("--wandb_tags", type=str, default="", help="Comma-separated tags")
    parser.add_argument("--wandb_notes", type=str, default="")
    parser.add_argument("--wandb_resume", action="store_true", help="Allow resuming an existing W&B run (same run id).")
    parser.add_argument("--wandb_offline", action="store_true", help="Run W&B in offline mode.")
    parser.add_argument("--wandb_watch", action="store_true", help="Enable wandb.watch(model) for gradients.")
    parser.add_argument("--wandb_watch_freq", type=int, default=200)
    parser.add_argument("--wandb_artifact_name", type=str, default="value_model_best")

    args = parser.parse_args()

    if args.log_file == "":
        args.log_file = None

    train(args)
