#!/usr/bin/env python3
"""Train pi_{k+1} from weighted continuous search-policy targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from common import ACTION_COLS, log, random_flip_state_action_z, random_team_swap_state_cond, set_seed
from policy_model import PolicySetTransformerMDN


def _load_policy_ckpt(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = PolicySetTransformerMDN(
        input_dim=ckpt.get("input_dim", 24),
        cond_dim=ckpt.get("cond_dim", 3),
        action_dim=ckpt.get("action_dim", 4),
        hidden_dim=args.get("hidden_dim", 256),
        n_layers=args.get("n_layers", 4),
        n_heads=args.get("n_heads", 4),
        dropout=args.get("dropout", 0.10),
        n_mixtures=args.get("n_mixtures", 16),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    mean = torch.tensor(ckpt["action_mean"], dtype=torch.float32)
    std = torch.tensor(ckpt["action_std"], dtype=torch.float32).clamp(min=1e-4)
    return model, ckpt, mean, std


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    wsum = 0.0
    for x, c, z, w in loader:
        x = x.to(device)
        c = c.to(device)
        z = z.to(device)
        w = w.to(device)
        nll = model.nll_per_sample(x, c, z)
        total += float((nll * w).sum().item())
        wsum += float(w.sum().item())
    return total / max(wsum, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-policy", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--no-augment-flip", action="store_true", help="Disable horizontal flip augmentation for weighted policy targets.")
    ap.add_argument("--no-augment-team-swap", action="store_true", help="Disable team slot-block swap augmentation for weighted policy targets.")
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    if log_path.exists():
        log_path.unlink()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, ckpt, action_mean, action_std = _load_policy_ckpt(Path(args.init_policy), device)
    action_mean_device = action_mean.to(device)
    action_std_device = action_std.to(device)
    df = pd.read_csv(args.targets)
    x_cols = [f"x{i}" for i in range(24)]
    c_cols = [f"c{i}" for i in range(3)]
    x = torch.tensor(df[x_cols].to_numpy(dtype=np.float32))
    c = torch.tensor(df[c_cols].to_numpy(dtype=np.float32))
    a = torch.tensor(df[ACTION_COLS].to_numpy(dtype=np.float32))
    z = (a - action_mean) / action_std
    w = torch.tensor(df["weight"].to_numpy(dtype=np.float32))
    w = w / max(float(w.mean().item()), 1e-9)

    perm = torch.randperm(len(x))
    n_val = max(1, int(0.1 * len(x)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_loader = DataLoader(TensorDataset(x[train_idx], c[train_idx], z[train_idx], w[train_idx]), batch_size=args.batch_size, shuffle=True, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(TensorDataset(x[val_idx], c[val_idx], z[val_idx], w[val_idx]), batch_size=args.batch_size * 2, shuffle=False, pin_memory=(device.type == "cuda"))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = float("inf")
    best_state = None
    best_epoch = 0
    no_imp = 0
    log(
        f"rows={len(x)} train={len(train_idx)} val={len(val_idx)} device={device} "
        f"augment_flip={not args.no_augment_flip} augment_team_swap={not args.no_augment_team_swap}",
        log_path,
    )
    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        wsum = 0.0
        for xb, cb, zb, wb in train_loader:
            xb = xb.to(device, non_blocking=True)
            cb = cb.to(device, non_blocking=True)
            zb = zb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            if not args.no_augment_flip:
                xb, zb = random_flip_state_action_z(xb, zb, action_mean_device, action_std_device)
            if not args.no_augment_team_swap:
                xb, cb = random_team_swap_state_cond(xb, cb)
            opt.zero_grad(set_to_none=True)
            nll = model.nll_per_sample(xb, cb, zb)
            loss = (nll * wb).sum() / wb.sum().clamp(min=1e-9)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float((nll.detach() * wb).sum().item())
            wsum += float(wb.sum().item())
        val = evaluate(model, val_loader, device)
        if val < best:
            best = val
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if ep == 1 or ep % 5 == 0 or no_imp == 0:
            log(f"epoch={ep:03d} train_weighted_nll={total/max(wsum,1e-9):.4f} val_weighted_nll={val:.4f} best={best:.4f}@{best_epoch}", log_path)
        if args.patience > 0 and no_imp >= args.patience:
            break
    out = dict(ckpt)
    out["model_state_dict"] = best_state
    out["distilled_from"] = str(args.init_policy)
    out["search_targets"] = str(args.targets)
    out["best_val_weighted_nll"] = float(best)
    out["best_epoch"] = int(best_epoch)
    torch.save(out, out_dir / "model.pt")
    (out_dir / "summary.json").write_text(json.dumps({k: v for k, v in out.items() if k != "model_state_dict"}, indent=2, sort_keys=True) + "\n")
    log(f"saved {out_dir / 'model.pt'}", log_path)


if __name__ == "__main__":
    main()
