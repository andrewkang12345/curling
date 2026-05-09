import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Subset

from dataset import ValueDataset, NUM_STONES
from model import ValueTransformer
from split_utils import COMPETITION_KEY, make_train_val_test_indices, write_split_competition_ids


def _log(msg: str, log_file: Path | None):
    print(msg)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a") as f:
            f.write(msg + "\n")


def _evaluate(model, loader, criterion, device):
    model.eval()
    running = 0.0
    with torch.no_grad():
        for x, c, y in loader:
            x = x.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x, c)
            loss = criterion(pred, y)
            running += loss.item() * x.size(0)
    mse = running / len(loader.dataset)
    return mse, float(np.sqrt(mse))


def train(args):
    log_path = Path(args.log_file) if args.log_file else None

    real_ds = ValueDataset(args.stones_csv, args.ends_csv)
    synth_ds = ValueDataset(args.synth_stones_csv, args.synth_ends_csv)

    _log(f"Real dataset size={len(real_ds)}, Synth dataset size={len(synth_ds)}", log_path)
    _log(f"input_dim={real_ds.input_dim}, cond_dim={real_ds.cond_dim}, num_tasks={real_ds.num_tasks}", log_path)

    if args.val_competition_id is not None:
        heldout_comp = int(args.val_competition_id)
        comp_vals = pd.to_numeric(real_ds.df["CompetitionID"], errors="coerce").astype("Int64")
        real_val_idx = np.where(comp_vals.to_numpy(dtype="int64", na_value=-1) == heldout_comp)[0].astype(np.int64)
        real_train_idx = np.where(comp_vals.to_numpy(dtype="int64", na_value=-1) != heldout_comp)[0].astype(np.int64)
        real_test_idx = np.empty((0,), dtype=np.int64)
        if len(real_val_idx) == 0:
            raise ValueError(f"No rows found for held-out CompetitionID={heldout_comp}")
    else:
        group_keys = real_ds.df[COMPETITION_KEY].to_numpy()
        real_train_idx, real_val_idx, real_test_idx = make_train_val_test_indices(
            n=len(real_ds),
            val_split=args.val_split,
            test_split=0.0,
            seed=args.split_seed,
            group_keys=group_keys,
        )
    if args.val_split > 0.0 and len(real_val_idx) == 0:
        raise ValueError(
            "Validation split is empty under competition-level grouping. "
            "Increase --val_split so at least one competition is held out."
        )

    real_train_ds = Subset(real_ds, real_train_idx)
    real_val_ds = Subset(real_ds, real_val_idx)
    train_ds = ConcatDataset([real_train_ds, synth_ds])
    _log(f"Real split: train={len(real_train_ds)}, val={len(real_val_ds)}, test={len(real_test_idx)}", log_path)

    if args.val_competitions_out:
        val_comp_path, n_val_comp = write_split_competition_ids(real_ds.df, real_val_idx, args.val_competitions_out)
        _log(
            f"Held-out val competition ids written to {val_comp_path} (rows={n_val_comp}).",
            log_path,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        real_val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    else:
        device = torch.device(args.device)
    _log(f"Using device: {device}", log_path)

    model = ValueTransformer(
        input_dim=real_ds.input_dim,
        cond_dim=real_ds.cond_dim,
        hidden_dim=args.hidden_dim,
        num_stones=NUM_STONES,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    start_epoch = 1
    best_val_mse = float("inf")
    best_epoch = 0
    epochs_without_improve = 0

    if args.resume and os.path.exists(args.resume):
        _log(f"Resuming from checkpoint: {args.resume}", log_path)
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_mse = float(ckpt.get("best_val_mse", best_val_mse))
        best_epoch = int(ckpt.get("best_epoch", best_epoch))
    elif args.resume:
        _log(f"WARNING: --resume path {args.resume} does not exist. Starting from scratch.", log_path)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for x, c, y in train_loader:
            x = x.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x, c)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * x.size(0)

        train_mse = running_loss / len(train_ds)
        train_rmse = float(np.sqrt(train_mse))
        val_mse, val_rmse = _evaluate(model, val_loader, criterion, device)

        _log(
            f"Epoch {epoch:03d} | train_mse={train_mse:.6f} | train_rmse={train_rmse:.6f} | "
            f"val_mse={val_mse:.6f} | val_rmse={val_rmse:.6f} | best_val_mse={best_val_mse:.6f}",
            log_path,
        )

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            epochs_without_improve = 0
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "input_dim": real_ds.input_dim,
                "cond_dim": real_ds.cond_dim,
                "hidden_dim": args.hidden_dim,
                "num_stones": NUM_STONES,
                "num_tasks": real_ds.num_tasks,
                "best_val_mse": best_val_mse,
                "best_epoch": best_epoch,
                "args": vars(args),
            }
            torch.save(ckpt, args.out)
            _log(
                f"Saved NEW BEST checkpoint to {args.out} (epoch {epoch}, val_mse={val_mse:.6f}, val_rmse={val_rmse:.6f})",
                log_path,
            )
        else:
            epochs_without_improve += 1

        if args.patience > 0 and epochs_without_improve >= args.patience:
            _log(
                f"Early stopping at epoch {epoch}; no validation improvement for {epochs_without_improve} epochs.",
                log_path,
            )
            break

    _log(
        f"Training finished. Best val_mse={best_val_mse:.6f} | best val_rmse={float(np.sqrt(best_val_mse)):.6f} at epoch {best_epoch}.",
        log_path,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Train transformer value model on real_train + synth, validate on held-out competitions only."
    )
    ap.add_argument("--stones_csv", type=str, default="../2026/Stones.csv")
    ap.add_argument("--ends_csv", type=str, default="../2026/Ends.csv")
    ap.add_argument("--synth_stones_csv", type=str, default="synth_terminal_stones.csv")
    ap.add_argument("--synth_ends_csv", type=str, default="synth_terminal_ends.csv")
    ap.add_argument("--out", type=str, default="model.pt")
    ap.add_argument("--log_file", type=str, default="train.log")
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--val_competitions_out", type=str, default="val_competitions.csv")
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--no_cuda", action="store_true")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--val_split", type=float, default=0.25)
    ap.add_argument("--split_seed", type=int, default=123)
    ap.add_argument("--val_competition_id", type=int, default=None, help="If set, hold out this exact CompetitionID instead of random competition split.")
    args = ap.parse_args()
    train(args)
