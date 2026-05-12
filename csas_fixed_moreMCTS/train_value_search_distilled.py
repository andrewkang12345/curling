#!/usr/bin/env python3
"""Train Gaussian SetTransformer value model with human labels plus search targets."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from common import FIXED_ROOT, KEY_COLS, log, random_flip_state_cond, random_team_swap_state_cond, set_seed
from dataset import ValueDataset, NUM_STONES, POS_MAX
from new_architectures import ValueSetTransformerGaussian
from train_holdout_models_cond3 import make_holdout_split, materialize
from preplaced_value_data import load_preplaced_tensors_for_train

def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.exp(-logvar) * (y - mean).pow(2) + logvar).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    pred, logv, ys = [], [], []
    for x, c, y in loader:
        x = x.to(device, non_blocking=True)
        c = c.to(device, non_blocking=True)
        m, lv = model(x, c)
        pred.append(m.cpu())
        logv.append(lv.cpu())
        ys.append(y)
    pred = torch.cat(pred)
    logv = torch.cat(logv)
    ys = torch.cat(ys)
    mse = F.mse_loss(pred, ys).item()
    return {"mse": float(mse), "rmse": float(np.sqrt(mse)), "nll": float(gaussian_nll(pred, logv, ys).item())}


def load_search_rows(path: Path, ds: ValueDataset, Xp: torch.Tensor, Xc: torch.Tensor, weight: int, target_col: str):
    st = pd.read_csv(path)
    if st.empty:
        raise RuntimeError(f"Empty search target file: {path}")
    x_cols = [f"x{i}" for i in range(24)]
    c_cols = [f"c{i}" for i in range(3)]
    if all(c in st.columns for c in x_cols + c_cols + [target_col]):
        sx = torch.tensor(st[x_cols].to_numpy(dtype=np.float32))
        sc = torch.tensor(st[c_cols].to_numpy(dtype=np.float32))
        sy = torch.tensor(st[target_col].to_numpy(dtype=np.float32)[:, None])
        if weight > 1:
            sx = sx.repeat_interleave(weight, dim=0)
            sc = sc.repeat_interleave(weight, dim=0)
            sy = sy.repeat_interleave(weight, dim=0)
        return sx, sc, sy, int(len(st))
    frame = ds.df.reset_index().rename(columns={"index": "_ds_idx"})
    merge_cols = KEY_COLS + ["TeamID"]
    merged = frame.merge(st[merge_cols + [target_col]], on=merge_cols, how="inner")
    idx = merged["_ds_idx"].to_numpy(dtype=np.int64)
    y = torch.tensor(merged[target_col].to_numpy(dtype=np.float32)[:, None])
    if weight > 1:
        idx = np.repeat(idx, weight)
        y = y.repeat_interleave(weight, dim=0)
    return Xp[idx], Xc[idx], y, int(len(merged))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=0)
    ap.add_argument("--search_targets", default="")
    ap.add_argument("--out_dir", default="checkpoints/value_search_distilled")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=140)
    ap.add_argument("--patience", type=int, default=22)
    ap.add_argument("--batch_size", type=int, default=1536)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.08)
    ap.add_argument("--nll_weight", type=float, default=0.2)
    ap.add_argument("--search_weight", type=int, default=2)
    ap.add_argument("--target_col", default="search_target")
    ap.add_argument("--val_end_frac", type=float, default=0.10)
    ap.add_argument("--split_seed", type=int, default=123)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--init_checkpoint", default="", help="Optional value checkpoint to warm-start from.")
    ap.add_argument("--include_synth_terminal", action="store_true")
    ap.add_argument("--include_preplaced", action="store_true")
    ap.add_argument("--preplaced_weight", type=int, default=1)
    ap.add_argument("--no_augment_flip", action="store_true", help="Disable horizontal flip augmentation.")
    ap.add_argument("--no_augment_team_swap", action="store_true", help="Disable team slot-block swap augmentation.")
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    if log_path.exists():
        log_path.unlink()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    ds = ValueDataset(str(FIXED_ROOT / "2026" / "Stones.csv"), str(FIXED_ROOT / "2026" / "Ends.csv"), augment_positions=False, augment_flip=False)
    Xp, Xc, Y = materialize(ds)
    train_idx, val_idx, test_idx, per_comp = make_holdout_split(ds.df, args.holdout, args.val_end_frac, args.split_seed)
    if args.limit_train > 0:
        train_idx = train_idx[: args.limit_train]
    train_parts_x = [Xp[train_idx]]
    train_parts_c = [Xc[train_idx]]
    train_parts_y = [Y[train_idx]]
    extra_counts = {}

    if args.include_synth_terminal:
        synth_ds = ValueDataset(
            str(FIXED_ROOT / "valueModel" / "synth_terminal_stones.csv"),
            str(FIXED_ROOT / "valueModel" / "synth_terminal_ends.csv"),
            augment_positions=False,
            augment_flip=False,
        )
        sx0, sc0, sy0 = materialize(synth_ds)
        train_parts_x.append(sx0)
        train_parts_c.append(sc0)
        train_parts_y.append(sy0)
        extra_counts["synth_terminal"] = int(len(sx0))

    if args.include_preplaced:
        train_comps = set(int(x) for x in pd.unique(ds.df.iloc[train_idx]["CompetitionID"]).tolist())
        px, pc, py, pdf = load_preplaced_tensors_for_train(train_comps)
        if args.preplaced_weight > 1:
            px = px.repeat_interleave(args.preplaced_weight, dim=0)
            pc = pc.repeat_interleave(args.preplaced_weight, dim=0)
            py = py.repeat_interleave(args.preplaced_weight, dim=0)
        train_parts_x.append(px)
        train_parts_c.append(pc)
        train_parts_y.append(py)
        extra_counts["preplaced_unique"] = int(len(pdf))
        extra_counts["preplaced_weighted"] = int(len(px))

    if args.search_targets:
        sx, sc, sy, n_unique = load_search_rows(Path(args.search_targets), ds, Xp, Xc, args.search_weight, args.target_col)
        train_parts_x.append(sx)
        train_parts_c.append(sc)
        train_parts_y.append(sy)
        extra_counts["search_unique"] = int(n_unique)
        extra_counts["search_weighted"] = int(len(sx))

    tr_x = torch.cat(train_parts_x, 0)
    tr_c = torch.cat(train_parts_c, 0)
    tr_y = torch.cat(train_parts_y, 0)
    log(
        f"train_real={len(train_idx)} extras={extra_counts} total_train={len(tr_x)} val={len(val_idx)} test={len(test_idx)} "
        f"device={device} augment_flip={not args.no_augment_flip} augment_team_swap={not args.no_augment_team_swap}",
        log_path,
    )

    train_loader = DataLoader(TensorDataset(tr_x, tr_c, tr_y), batch_size=args.batch_size, shuffle=True, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(TensorDataset(Xp[val_idx], Xc[val_idx], Y[val_idx]), batch_size=args.batch_size * 2, shuffle=False, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(TensorDataset(Xp[test_idx], Xc[test_idx], Y[test_idx]), batch_size=args.batch_size * 2, shuffle=False, pin_memory=(device.type == "cuda"))

    model = ValueSetTransformerGaussian(
        input_dim=ds.input_dim,
        cond_dim=ds.cond_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        min_logvar=-6.0,
        max_logvar=3.5,
    ).to(device)
    if args.init_checkpoint:
        init = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        state = init.get("model_state_dict", init)
        missing, unexpected = model.load_state_dict(state, strict=False)
        log(
            f"Warm-started from {args.init_checkpoint} missing={len(missing)} unexpected={len(unexpected)}",
            log_path,
        )
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        train_model = torch.nn.DataParallel(model)
        log(f"Using DataParallel over {torch.cuda.device_count()} GPUs", log_path)
    else:
        train_model = model
    opt = torch.optim.AdamW(train_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = float("inf")
    best_state = None
    best_epoch = 0
    no_imp = 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        train_model.train()
        tot_loss = tot_mse = tot_nll = count = 0.0
        for x, c, y in train_loader:
            x = x.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if not args.no_augment_flip:
                x, c = random_flip_state_cond(x, c)
            if not args.no_augment_team_swap:
                x, c = random_team_swap_state_cond(x, c)
            opt.zero_grad(set_to_none=True)
            mean, logvar = train_model(x, c)
            mse = F.mse_loss(mean, y)
            nll = gaussian_nll(mean, logvar, y)
            loss = mse + args.nll_weight * nll
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), 1.0)
            opt.step()
            b = x.size(0)
            tot_loss += loss.item() * b
            tot_mse += mse.item() * b
            tot_nll += nll.item() * b
            count += b
        val = evaluate(model, val_loader, device)
        key = val["mse"] + 0.05 * val["nll"]
        if key < best:
            best = key
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if ep == 1 or ep % 10 == 0 or no_imp == 0:
            log(f"epoch={ep:03d} train_mse={tot_mse/max(1,count):.5f} train_nll={tot_nll/max(1,count):.5f} val_mse={val['mse']:.5f} val_nll={val['nll']:.5f} best={best:.5f}@{best_epoch} time={time.time()-t0:.1f}s", log_path)
        if args.patience > 0 and no_imp >= args.patience:
            log(f"early_stop epoch={ep}", log_path)
            break

    model.load_state_dict(best_state)
    val = evaluate(model, val_loader, device)
    test = evaluate(model, test_loader, device)
    ckpt = {
        "arch": "set_transformer_gaussian_search_distilled",
        "model_state_dict": best_state,
        "input_dim": ds.input_dim,
        "cond_dim": ds.cond_dim,
        "hidden_dim": args.hidden_dim,
        "num_stones": NUM_STONES,
        "args": vars(args),
        "extra_counts": extra_counts,
        "best_epoch": best_epoch,
        "val_metrics": val,
        "test_metrics": test,
    }
    torch.save(ckpt, out_dir / "model.pt")
    (out_dir / "summary.json").write_text(json.dumps({k: v for k, v in ckpt.items() if k != "model_state_dict"}, indent=2, sort_keys=True) + "\n")
    log(f"saved {out_dir / 'model.pt'} val={val} test={test}", log_path)


if __name__ == "__main__":
    main()
