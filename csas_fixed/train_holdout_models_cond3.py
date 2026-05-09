#!/usr/bin/env python3
"""
Train the best value model (SetTransformer medium, cond_dim=3) for each holdout
competition using a leakage-free split:

- test: the held-out competition
- val: a fixed fraction of end-groups from each remaining competition
- train: the rest of the non-held-out real data, plus all synth data

Checkpoints are written to holdouts/{comp_id}/model/model.pt.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR / "valueModel"))
sys.path.insert(0, str(THIS_DIR / "valueModel" / "ablation"))

from dataset import ValueDataset, NUM_STONES, POS_MAX
from new_architectures import ValueSetTransformer

FLIP_CENTER_X = 1500.0 / POS_MAX
HOLDOUT_IDS = [0, 22230015, 23240026, 24250026]
END_KEY = ["CompetitionID", "SessionID", "GameID", "EndID"]


def _log(msg: str, log_file: Path | None):
    print(msg, flush=True)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a") as f:
            f.write(msg + "\n")


def augment_flip_batch(x):
    B = x.size(0)
    stones = x.view(B, NUM_STONES, 2).clone()
    flip_mask = (torch.rand(B, device=x.device) < 0.5).view(B, 1, 1)
    ip = ((stones.sum(dim=-1) > 0.001) & (stones.max(dim=-1).values < 0.999)).unsqueeze(-1)
    flipped_x = FLIP_CENTER_X - stones[:, :, 0:1]
    new_x = torch.where(flip_mask & ip, flipped_x, stones[:, :, 0:1])
    return torch.cat([new_x, stones[:, :, 1:2]], dim=-1).view(B, -1)


def materialize(ds):
    loader = DataLoader(ds, batch_size=8192, shuffle=False, num_workers=0)
    xs, cs, ys = [], [], []
    for x, c, y in loader:
        xs.append(x)
        cs.append(c)
        ys.append(y)
    return torch.cat(xs, 0), torch.cat(cs, 0), torch.cat(ys, 0)


def _evaluate(model, loader, criterion, device):
    model.eval()
    running = 0.0
    count = 0
    with torch.no_grad():
        for x, c, y in loader:
            x = x.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            running += criterion(model(x, c), y).item() * x.size(0)
            count += x.size(0)
    mse = running / max(1, count)
    return mse, float(np.sqrt(mse))


def _seed_for_split(base_seed: int, holdout_comp: int, comp_id: int) -> int:
    return int((base_seed * 1_000_003 + holdout_comp * 9_973 + comp_id * 37) % (2**32 - 1))


def _split_comp_rows_by_end(df: pd.DataFrame, comp_indices: np.ndarray, holdout_comp: int,
                            comp_id: int, val_end_frac: float, split_seed: int):
    local_end_keys = df.iloc[comp_indices][END_KEY].to_numpy()
    _, inverse = np.unique(local_end_keys, axis=0, return_inverse=True)
    n_groups = int(inverse.max() + 1) if len(comp_indices) > 0 else 0

    if n_groups <= 1 or val_end_frac <= 0.0:
        local_val = np.empty((0,), dtype=np.int64)
        local_train = np.arange(len(comp_indices), dtype=np.int64)
        n_val_groups = 0
    else:
        rng = np.random.default_rng(_seed_for_split(split_seed, holdout_comp, comp_id))
        perm = rng.permutation(n_groups).astype(np.int64)
        n_val_groups = max(1, int(round(n_groups * float(val_end_frac))))
        n_val_groups = min(n_groups - 1, n_val_groups)
        val_groups = set(perm[:n_val_groups].tolist())
        local_val_mask = np.isin(inverse, list(val_groups))
        local_val = np.where(local_val_mask)[0].astype(np.int64)
        local_train = np.where(~local_val_mask)[0].astype(np.int64)

    return {
        "train_idx": comp_indices[local_train],
        "val_idx": comp_indices[local_val],
        "n_end_groups": n_groups,
        "n_val_groups": n_val_groups,
    }


def make_holdout_split(df: pd.DataFrame, holdout_comp: int, val_end_frac: float, split_seed: int):
    comp_vals = pd.to_numeric(df["CompetitionID"], errors="coerce").to_numpy(dtype="int64")
    unique_comps = sorted(pd.unique(comp_vals).tolist())

    test_idx = np.where(comp_vals == holdout_comp)[0].astype(np.int64)
    if len(test_idx) == 0:
        raise ValueError(f"No rows found for held-out CompetitionID={holdout_comp}")

    train_parts = []
    val_parts = []
    per_comp = {}

    for comp_id in unique_comps:
        if int(comp_id) == int(holdout_comp):
            continue
        comp_indices = np.where(comp_vals == comp_id)[0].astype(np.int64)
        split = _split_comp_rows_by_end(df, comp_indices, holdout_comp, int(comp_id), val_end_frac, split_seed)
        train_parts.append(split["train_idx"])
        val_parts.append(split["val_idx"])
        per_comp[str(int(comp_id))] = {
            "rows_total": int(len(comp_indices)),
            "rows_train": int(len(split["train_idx"])),
            "rows_val": int(len(split["val_idx"])),
            "end_groups_total": int(split["n_end_groups"]),
            "end_groups_val": int(split["n_val_groups"]),
        }

    train_idx = np.concatenate(train_parts).astype(np.int64) if train_parts else np.empty((0,), dtype=np.int64)
    val_idx = np.concatenate(val_parts).astype(np.int64) if val_parts else np.empty((0,), dtype=np.int64)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(
            f"Invalid split for holdout {holdout_comp}: train={len(train_idx)} val={len(val_idx)}."
        )

    return train_idx, val_idx, test_idx, per_comp


def _write_table(df: pd.DataFrame, cols: list[str], indices: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(indices) == 0:
        pd.DataFrame(columns=cols).to_csv(out_path, index=False)
        return 0

    table = df.iloc[np.asarray(indices, dtype=np.int64)][cols].copy()
    for col in cols:
        table[col] = pd.to_numeric(table[col], errors="coerce").astype("Int64")
    table = table.dropna(subset=cols).astype({c: "int64" for c in cols})
    table = table.drop_duplicates(subset=cols).sort_values(cols).reset_index(drop=True)
    table.to_csv(out_path, index=False)
    return int(len(table))


def train_one_holdout(args, real_ds, real_Xp, real_Xc, real_Y, synth_Xp, synth_Xc, synth_Y, holdout_comp):
    run_dir = THIS_DIR / "holdouts" / str(holdout_comp)
    out_dir = run_dir / "model"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_file) if args.log_file else (run_dir / "logs" / "train.log")
    if log_path.exists():
        log_path.unlink()

    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"))
    _log(f"Device: {device}", log_path)
    _log(f"Holdout test competition: {holdout_comp}", log_path)

    train_idx, val_idx, test_idx, per_comp = make_holdout_split(
        real_ds.df, holdout_comp, args.val_end_frac, args.split_seed
    )

    tr_Xp = torch.cat([real_Xp[train_idx], synth_Xp], 0)
    tr_Xc = torch.cat([real_Xc[train_idx], synth_Xc], 0)
    tr_Y = torch.cat([real_Y[train_idx], synth_Y], 0)

    train_td = TensorDataset(tr_Xp, tr_Xc, tr_Y)
    val_td = TensorDataset(real_Xp[val_idx], real_Xc[val_idx], real_Y[val_idx])
    test_td = TensorDataset(real_Xp[test_idx], real_Xc[test_idx], real_Y[test_idx])

    _log(f"Real split sizes | train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}", log_path)
    _log(f"Train+synth size={len(train_td)} (synth={len(synth_Xp)})", log_path)
    _log(f"Val competitions={sorted(pd.unique(real_ds.df.iloc[val_idx]['CompetitionID']).tolist())}", log_path)
    _log("Per-train-competition val split:", log_path)
    for comp_id, stats in per_comp.items():
        _log(
            f"  comp={comp_id} rows(train/val/total)="
            f"{stats['rows_train']}/{stats['rows_val']}/{stats['rows_total']} "
            f"end_groups(val/total)={stats['end_groups_val']}/{stats['end_groups_total']}",
            log_path,
        )

    cfg = dict(hidden_dim=args.hidden_dim, n_layers=args.n_layers, n_heads=args.n_heads, dropout=args.dropout)
    model = ValueSetTransformer(
        input_dim=real_ds.input_dim,
        cond_dim=real_ds.cond_dim,
        **cfg,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log(f"Model params: {n_params:,}", log_path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    train_loader = DataLoader(
        train_td,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_td,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_td,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    best_val = float("inf")
    best_ep = 0
    no_imp = 0
    best_state = None

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running = 0.0
        count = 0
        for x, c, y in train_loader:
            x = x.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x = augment_flip_batch(x)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x, c), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * x.size(0)
            count += x.size(0)

        train_mse = running / max(1, count)
        val_mse, _ = _evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        if val_mse < best_val:
            best_val = val_mse
            best_ep = ep
            no_imp = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1

        if ep % 10 == 0 or no_imp == 0:
            _log(
                f"Epoch {ep:03d} | train_mse={train_mse:.6f} | val_mse={val_mse:.6f} | "
                f"best_val_mse={best_val:.6f}@{best_ep} | {elapsed:.1f}s",
                log_path,
            )

        if args.patience > 0 and no_imp >= args.patience:
            _log(f"Early stop at epoch {ep}", log_path)
            break

    if best_state is None:
        raise RuntimeError(f"Training for holdout {holdout_comp} never produced a best state.")

    model.load_state_dict(best_state)
    test_mse, test_rmse = _evaluate(model, test_loader, criterion, device)
    _log(
        f"Best checkpoint evaluation | test_mse={test_mse:.6f} | test_rmse={test_rmse:.6f} | "
        f"best_val_mse={best_val:.6f} at epoch {best_ep}",
        log_path,
    )

    val_comp_path = out_dir / "val_competitions.csv"
    test_comp_path = out_dir / "test_competitions.csv"
    val_end_path = out_dir / "val_end_keys.csv"
    test_end_path = out_dir / "test_end_keys.csv"
    _write_table(real_ds.df, ["CompetitionID"], val_idx, val_comp_path)
    _write_table(real_ds.df, ["CompetitionID"], test_idx, test_comp_path)
    _write_table(real_ds.df, END_KEY, val_idx, val_end_path)
    _write_table(real_ds.df, END_KEY, test_idx, test_end_path)

    split_info = {
        "holdout_competition": int(holdout_comp),
        "train_competitions": sorted(
            int(x) for x in pd.unique(real_ds.df.iloc[train_idx]["CompetitionID"]).tolist()
        ),
        "val_competitions": sorted(
            int(x) for x in pd.unique(real_ds.df.iloc[val_idx]["CompetitionID"]).tolist()
        ),
        "test_competitions": [int(holdout_comp)],
        "val_end_fraction": float(args.val_end_frac),
        "split_seed": int(args.split_seed),
        "rows": {
            "train_real": int(len(train_idx)),
            "val_real": int(len(val_idx)),
            "test_real": int(len(test_idx)),
            "synth": int(len(synth_Xp)),
        },
        "per_train_competition": per_comp,
    }

    ckpt = {
        "epoch": best_ep,
        "model_state_dict": best_state,
        "input_dim": real_ds.input_dim,
        "cond_dim": real_ds.cond_dim,
        "hidden_dim": cfg["hidden_dim"],
        "num_stones": NUM_STONES,
        "best_val_mse": best_val,
        "best_epoch": best_ep,
        "test_mse": test_mse,
        "test_rmse": test_rmse,
        "split_info": split_info,
        "args": {
            "n_layers": cfg["n_layers"],
            "n_heads": cfg["n_heads"],
            "dropout": cfg["dropout"],
            "val_end_frac": float(args.val_end_frac),
            "split_seed": int(args.split_seed),
        },
    }
    torch.save(ckpt, out_dir / "model.pt")

    split_summary_path = out_dir / "split_summary.json"
    split_summary_path.write_text(json.dumps({
        **split_info,
        "best_val_mse": best_val,
        "best_epoch": best_ep,
        "test_mse": test_mse,
        "test_rmse": test_rmse,
    }, indent=2) + "\n")

    _log(f"Saved checkpoint: {out_dir / 'model.pt'}", log_path)
    _log(f"Saved split summary: {split_summary_path}", log_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only_holdout", type=int, default=None)
    ap.add_argument("--device", default="auto", help="e.g. auto, cpu, cuda, cuda:1")
    ap.add_argument("--no_cuda", action="store_true")
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--val_end_frac", type=float, default=0.10)
    ap.add_argument("--split_seed", type=int, default=123)
    ap.add_argument("--log_file", default="")
    args = ap.parse_args()

    real_ds = ValueDataset(
        str(THIS_DIR / "2026" / "Stones.csv"),
        str(THIS_DIR / "2026" / "Ends.csv"),
        augment_positions=False,
        augment_flip=False,
    )
    real_Xp, real_Xc, real_Y = materialize(real_ds)
    assert real_ds.cond_dim == 3, f"Expected cond_dim=3, got {real_ds.cond_dim}"

    synth_ds = ValueDataset(
        str(THIS_DIR / "valueModel" / "synth_terminal_stones.csv"),
        str(THIS_DIR / "valueModel" / "synth_terminal_ends.csv"),
        augment_positions=False,
        augment_flip=False,
    )
    synth_Xp, synth_Xc, synth_Y = materialize(synth_ds)

    holdout_ids = [args.only_holdout] if args.only_holdout is not None else HOLDOUT_IDS
    for holdout_comp in holdout_ids:
        train_one_holdout(args, real_ds, real_Xp, real_Xc, real_Y, synth_Xp, synth_Xc, synth_Y, int(holdout_comp))


if __name__ == "__main__":
    main()
