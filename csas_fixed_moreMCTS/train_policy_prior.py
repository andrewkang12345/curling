#!/usr/bin/env python3
"""Train the broad stochastic human throw prior used to seed KR-UCT search."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from common import log, random_flip_state_action_z, random_team_swap_state_cond, set_seed
from policy_dataset import build_policy_tensors
from policy_model import PolicySetTransformerMDN
from preplaced_value_data import materialize_preplaced_policy


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    count = 0
    for x, c, z in loader:
        x = x.to(device, non_blocking=True)
        c = c.to(device, non_blocking=True)
        z = z.to(device, non_blocking=True)
        loss = model.nll(x, c, z)
        total += loss.item() * x.size(0)
        count += x.size(0)
    return total / max(1, count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=0)
    ap.add_argument("--out_dir", default="checkpoints/policy_prior")
    ap.add_argument("--max_loss", type=float, default=0.12)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--n_mixtures", type=int, default=16)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--include_preplaced", action="store_true")
    ap.add_argument("--preplaced_max_loss", type=float, default=0.5)
    ap.add_argument("--preplaced_weight", type=int, default=1)
    ap.add_argument("--no_augment_flip", action="store_true", help="Disable horizontal flip augmentation for policy states/actions.")
    ap.add_argument("--no_augment_team_swap", action="store_true", help="Disable team slot-block swap augmentation for policy states/conditions.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    if log_path.exists():
        log_path.unlink()
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    tr_x, tr_c, tr_a, _, tr_meta = build_policy_tensors(holdout=args.holdout, split="train", max_loss=args.max_loss)
    va_x, va_c, va_a, _, va_meta = build_policy_tensors(holdout=args.holdout, split="val", max_loss=args.max_loss)
    extra_counts = {}
    if args.include_preplaced:
        train_comps = set(int(x) for x in tr_meta["CompetitionID"].unique().tolist())
        px, pc, pa, pdf = materialize_preplaced_policy(train_comps, max_loss=args.preplaced_max_loss)
        if args.preplaced_weight > 1:
            px = px.repeat_interleave(args.preplaced_weight, dim=0)
            pc = pc.repeat_interleave(args.preplaced_weight, dim=0)
            pa = pa.repeat_interleave(args.preplaced_weight, dim=0)
        tr_x = torch.cat([tr_x, px], dim=0)
        tr_c = torch.cat([tr_c, pc], dim=0)
        tr_a = torch.cat([tr_a, pa], dim=0)
        extra_counts["preplaced_policy_unique"] = int(len(pdf))
        extra_counts["preplaced_policy_weighted"] = int(len(px))
    if args.limit_train > 0:
        tr_x, tr_c, tr_a = tr_x[: args.limit_train], tr_c[: args.limit_train], tr_a[: args.limit_train]

    action_mean = tr_a.mean(0)
    action_std = tr_a.std(0).clamp(min=1e-4)
    tr_z = (tr_a - action_mean) / action_std
    va_z = (va_a - action_mean) / action_std
    log(
        f"train={len(tr_x)} val={len(va_x)} extras={extra_counts} device={device} "
        f"augment_flip={not args.no_augment_flip} augment_team_swap={not args.no_augment_team_swap} "
        f"action_mean={action_mean.tolist()}",
        log_path,
    )

    train_loader = DataLoader(
        TensorDataset(tr_x, tr_c, tr_z),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        TensorDataset(va_x, va_c, va_z),
        batch_size=args.batch_size * 2,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    model = PolicySetTransformerMDN(
        input_dim=tr_x.shape[1],
        cond_dim=tr_c.shape[1],
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        n_mixtures=args.n_mixtures,
    ).to(device)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        log(f"Using DataParallel over {torch.cuda.device_count()} GPUs", log_path)
        train_model = torch.nn.DataParallel(model)
    else:
        train_model = model
    opt = torch.optim.AdamW(train_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    action_mean_device = action_mean.to(device)
    action_std_device = action_std.to(device)

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    no_imp = 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        train_model.train()
        total = 0.0
        count = 0
        for x, c, z in train_loader:
            x = x.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            z = z.to(device, non_blocking=True)
            if not args.no_augment_flip:
                x, z = random_flip_state_action_z(x, z, action_mean_device, action_std_device)
            if not args.no_augment_team_swap:
                x, c = random_team_swap_state_cond(x, c)
            opt.zero_grad(set_to_none=True)
            loss = train_model.module.nll(x, c, z) if isinstance(train_model, torch.nn.DataParallel) else train_model.nll(x, c, z)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), 1.0)
            opt.step()
            total += loss.item() * x.size(0)
            count += x.size(0)
        val_loss = evaluate(model, val_loader, device)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if ep == 1 or ep % 5 == 0 or no_imp == 0:
            log(f"epoch={ep:03d} train_nll={total/max(1,count):.4f} val_nll={val_loss:.4f} best={best_loss:.4f}@{best_epoch} time={time.time()-t0:.1f}s", log_path)
        if args.patience > 0 and no_imp >= args.patience:
            log(f"early_stop epoch={ep}", log_path)
            break

    ckpt = {
        "model_state_dict": best_state,
        "arch": "policy_set_transformer_mdn",
        "input_dim": int(tr_x.shape[1]),
        "cond_dim": int(tr_c.shape[1]),
        "action_dim": int(tr_a.shape[1]),
        "action_mean": action_mean.tolist(),
        "action_std": action_std.tolist(),
        "args": vars(args),
        "best_val_nll": float(best_loss),
        "best_epoch": int(best_epoch),
    }
    torch.save(ckpt, out_dir / "model.pt")
    (out_dir / "summary.json").write_text(json.dumps({k: v for k, v in ckpt.items() if k != "model_state_dict"}, indent=2) + "\n")
    log(f"saved {out_dir / 'model.pt'}", log_path)


if __name__ == "__main__":
    main()
