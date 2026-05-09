import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torch import nn

from dataset import ValueDataset, NUM_STONES
from model import ValueTransformer
from split_utils import END_KEY, make_train_val_test_indices, write_test_shot_keys


def _log(msg: str, log_file: Path | None):
    print(msg)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a") as f:
            f.write(msg + "\n")


def train(args):
    log_path = Path(args.log_file) if args.log_file else None

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

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        if val_ds is not None
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    _log(f"Using device: {device}", log_path)

    model = ValueTransformer(
        input_dim=dataset.input_dim,
        cond_dim=dataset.cond_dim,
        hidden_dim=args.hidden_dim,
        num_stones=NUM_STONES,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    start_epoch = 1
    best_loss = float("inf")

    if args.resume and os.path.exists(args.resume):
        _log(f"Resuming from checkpoint: {args.resume}", log_path)
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        if "best_loss" in ckpt:
            best_loss = ckpt["best_loss"]
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

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for x, c, y in train_loader:
            x = x.to(device)
            c = c.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x, c)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(train_ds)

        val_loss = None
        if val_loader is not None:
            model.eval()
            val_running = 0.0
            with torch.no_grad():
                for x, c, y in val_loader:
                    x = x.to(device)
                    c = c.to(device)
                    y = y.to(device)
                    pred = model(x, c)
                    v_loss = criterion(pred, y)
                    val_running += v_loss.item() * x.size(0)
            val_loss = val_running / len(val_ds)
            current_metric = val_loss
            _log(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | best_loss={best_loss:.4f}",
                log_path,
            )
        else:
            current_metric = train_loss
            _log(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | best_loss={best_loss:.4f}",
                log_path,
            )

        is_last = (epoch == args.epochs)
        should_check = (epoch % args.save_every == 0) or is_last

        if should_check and current_metric < best_loss:
            best_loss = current_metric
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
            torch.save(ckpt, args.out)
            _log(f"Saved NEW BEST checkpoint to {args.out} (epoch {epoch}, metric={current_metric:.4f})", log_path)

    _log(
        f"Training finished. Best metric (val if available, else train): {best_loss:.4f}. Best checkpoint at {args.out}.",
        log_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer value network for curling end outcomes")
    parser.add_argument("--stones_csv", type=str, default="../2026/Stones.csv")
    parser.add_argument("--ends_csv", type=str, default="../2026/Ends.csv")
    parser.add_argument("--out", type=str, default="value_model.pt")
    parser.add_argument("--log_file", type=str, default="train.log")
    parser.add_argument("--resume", type=str, default="")
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

    args = parser.parse_args()
    if args.log_file == "":
        args.log_file = None
    train(args)
