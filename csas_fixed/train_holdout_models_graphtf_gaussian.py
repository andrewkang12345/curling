#!/usr/bin/env python3
"""Train Gaussian Graph Transformer value models for holdout competitions."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR / "valueModel"))
sys.path.insert(0, str(THIS_DIR / "valueModel" / "ablation"))

# Current best GraphTF feature stack: scorability + reachability + product.
os.environ.setdefault("GNN_EDGE_SCALAR_MODE", "button_visible_plus_release_reach_with_product")
os.environ.setdefault("GNN_NODE_FEATURE_MODE", "none")
os.environ.setdefault("GNN_RELEASE_NODE_MODE", "three")

from dataset import ValueDataset, NUM_STONES, POS_MAX  # type: ignore  # noqa: E402
from gnn_models import GNN_REGISTRY  # type: ignore  # noqa: E402
from train_holdout_models_cond3 import END_KEY, HOLDOUT_IDS, make_holdout_split, materialize, _write_table  # type: ignore  # noqa: E402

FLIP_CENTER_X = 1500.0 / POS_MAX


def _log(msg: str, log_file: Path | None):
    print(msg, flush=True)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a") as f:
            f.write(msg + "\n")


def augment_flip_batch(x: torch.Tensor) -> torch.Tensor:
    bsz = x.size(0)
    stones = x.view(bsz, NUM_STONES, 2).clone()
    flip_mask = (torch.rand(bsz, device=x.device) < 0.5).view(bsz, 1, 1)
    in_play = ((stones.sum(dim=-1) > 0.001) & (stones.max(dim=-1).values < 0.999)).unsqueeze(-1)
    flipped_x = FLIP_CENTER_X - stones[:, :, 0:1]
    new_x = torch.where(flip_mask & in_play, flipped_x, stones[:, :, 0:1])
    return torch.cat([new_x, stones[:, :, 1:2]], dim=-1).view(bsz, -1)


def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.exp(-logvar) * (y - mean).pow(2) + logvar).mean()


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    preds, logvars, ys = [], [], []
    for x, c, y in loader:
        x = x.to(device, non_blocking=True)
        c = c.to(device, non_blocking=True)
        mean, logvar = model(x, c)
        preds.append(mean.cpu())
        logvars.append(logvar.cpu())
        ys.append(y.cpu())
    mean = torch.cat(preds, 0)
    logvar = torch.cat(logvars, 0)
    y = torch.cat(ys, 0)
    mse = F.mse_loss(mean, y).item()
    nll = gaussian_nll(mean, logvar, y).item()
    sigma = torch.exp(0.5 * logvar)
    abs_err = (y - mean).abs()
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "nll": float(nll),
        "mean_sigma": float(sigma.mean().item()),
        "median_sigma": float(sigma.median().item()),
        "coverage_1sigma": float((abs_err <= sigma).float().mean().item()),
        "coverage_2sigma": float((abs_err <= 2.0 * sigma).float().mean().item()),
    }


def train_one_holdout(args, real_ds, real_Xp, real_Xc, real_Y, synth_Xp, synth_Xc, synth_Y, holdout_comp: int):
    run_dir = THIS_DIR / "holdouts" / str(holdout_comp)
    out_dir = run_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    if log_path.exists():
        log_path.unlink()

    device = torch.device(args.device)
    _log(f"Device: {device}", log_path)
    _log(f"Holdout test competition: {holdout_comp}", log_path)
    _log(
        "Graph feature env | "
        f"GNN_EDGE_SCALAR_MODE={os.environ.get('GNN_EDGE_SCALAR_MODE')} "
        f"GNN_NODE_FEATURE_MODE={os.environ.get('GNN_NODE_FEATURE_MODE')} "
        f"GNN_RELEASE_NODE_MODE={os.environ.get('GNN_RELEASE_NODE_MODE')}",
        log_path,
    )

    train_idx, val_idx, test_idx, per_comp = make_holdout_split(
        real_ds.df, holdout_comp, args.val_end_frac, args.split_seed
    )
    rng = np.random.default_rng(int(args.synth_seed))
    synth_perm = rng.permutation(len(synth_Xp))
    synth_n = min(int(round(float(args.synth_frac) * len(synth_perm))), len(synth_perm))
    synth_idx = synth_perm[:synth_n] if synth_n > 0 else np.empty((0,), dtype=np.int64)

    tr_Xp = torch.cat([real_Xp[train_idx], synth_Xp[synth_idx]], 0) if len(synth_idx) else real_Xp[train_idx]
    tr_Xc = torch.cat([real_Xc[train_idx], synth_Xc[synth_idx]], 0) if len(synth_idx) else real_Xc[train_idx]
    tr_Y = torch.cat([real_Y[train_idx], synth_Y[synth_idx]], 0) if len(synth_idx) else real_Y[train_idx]

    train_td = TensorDataset(tr_Xp, tr_Xc, tr_Y)
    val_td = TensorDataset(real_Xp[val_idx], real_Xc[val_idx], real_Y[val_idx])
    test_td = TensorDataset(real_Xp[test_idx], real_Xc[test_idx], real_Y[test_idx])
    _log(f"Real split sizes | train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}", log_path)
    _log(f"Train+synth size={len(train_td)} (synth_used={len(synth_idx)})", log_path)

    cfg = dict(
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        min_logvar=args.min_logvar,
        max_logvar=args.max_logvar,
    )
    model = GNN_REGISTRY["graph_transformer_gaussian"](
        input_dim=real_ds.input_dim,
        cond_dim=real_ds.cond_dim,
        **cfg,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log(f"Model params: {n_params:,}", log_path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(train_td, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_td, batch_size=args.batch_size * 2, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_td, batch_size=args.batch_size * 2, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    best_key = float("inf")
    best_ep = 0
    best_state = None
    no_imp = 0

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_mse = 0.0
        running_nll = 0.0
        count = 0
        for x, c, y in train_loader:
            x = augment_flip_batch(x.to(device, non_blocking=True))
            c = c.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            mean, logvar = model(x, c)
            mse = F.mse_loss(mean, y)
            nll = gaussian_nll(mean, logvar, y)
            loss = mse + args.nll_weight * nll
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            bsz = x.size(0)
            running_loss += loss.item() * bsz
            running_mse += mse.item() * bsz
            running_nll += nll.item() * bsz
            count += bsz

        val_metrics = _evaluate(model, val_loader, device)
        val_key = val_metrics["mse"] + args.val_nll_weight * val_metrics["nll"]
        if val_key < best_key:
            best_key = val_key
            best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        if ep % args.log_every == 0 or no_imp == 0:
            _log(
                f"Epoch {ep:03d} | train_loss={running_loss / max(1, count):.6f} "
                f"train_mse={running_mse / max(1, count):.6f} train_nll={running_nll / max(1, count):.6f} "
                f"val_mse={val_metrics['mse']:.6f} val_nll={val_metrics['nll']:.6f} "
                f"val_sigma={val_metrics['mean_sigma']:.4f} cov1={val_metrics['coverage_1sigma']:.3f} "
                f"cov2={val_metrics['coverage_2sigma']:.3f} best_key={best_key:.6f}@{best_ep} "
                f"{time.time() - t0:.1f}s",
                log_path,
            )
        if args.patience > 0 and no_imp >= args.patience:
            _log(f"Early stop at epoch {ep}", log_path)
            break

    if best_state is None:
        raise RuntimeError("No best checkpoint was produced.")
    model.load_state_dict(best_state)
    val_metrics = _evaluate(model, val_loader, device)
    test_metrics = _evaluate(model, test_loader, device)
    _log(f"Best validation metrics: {json.dumps(val_metrics, sort_keys=True)}", log_path)
    _log(f"Test metrics: {json.dumps(test_metrics, sort_keys=True)}", log_path)

    _write_table(real_ds.df, ["CompetitionID"], val_idx, out_dir / "val_competitions.csv")
    _write_table(real_ds.df, ["CompetitionID"], test_idx, out_dir / "test_competitions.csv")
    _write_table(real_ds.df, END_KEY, val_idx, out_dir / "val_end_keys.csv")
    _write_table(real_ds.df, END_KEY, test_idx, out_dir / "test_end_keys.csv")

    split_info = {
        "holdout_competition": int(holdout_comp),
        "train_competitions": sorted(int(x) for x in pd.unique(real_ds.df.iloc[train_idx]["CompetitionID"]).tolist()),
        "val_competitions": sorted(int(x) for x in pd.unique(real_ds.df.iloc[val_idx]["CompetitionID"]).tolist()),
        "test_competitions": [int(holdout_comp)],
        "val_end_fraction": float(args.val_end_frac),
        "split_seed": int(args.split_seed),
        "rows": {
            "train_real": int(len(train_idx)),
            "val_real": int(len(val_idx)),
            "test_real": int(len(test_idx)),
            "synth_total": int(len(synth_Xp)),
            "synth_used": int(len(synth_idx)),
        },
        "per_train_competition": per_comp,
        "best_epoch": int(best_ep),
        "best_val_key": float(best_key),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "training_loss": {
            "mean_mse_weight": 1.0,
            "nll_weight": float(args.nll_weight),
            "val_nll_weight": float(args.val_nll_weight),
        },
        "graph_feature_env": {
            "GNN_EDGE_SCALAR_MODE": os.environ.get("GNN_EDGE_SCALAR_MODE"),
            "GNN_NODE_FEATURE_MODE": os.environ.get("GNN_NODE_FEATURE_MODE"),
            "GNN_RELEASE_NODE_MODE": os.environ.get("GNN_RELEASE_NODE_MODE"),
        },
    }
    ckpt = {
        "arch": "graph_transformer_gaussian",
        "epoch": int(best_ep),
        "model_state_dict": best_state,
        "input_dim": int(real_ds.input_dim),
        "cond_dim": int(real_ds.cond_dim),
        "hidden_dim": int(args.hidden_dim),
        "num_stones": int(NUM_STONES),
        "model_class": "ValueGraphTransformerGaussianFast",
        "split_info": split_info,
        "args": vars(args),
    }
    torch.save(ckpt, out_dir / "model.pt")
    (out_dir / "split_summary.json").write_text(json.dumps(split_info, indent=2, sort_keys=True) + "\n")
    _log(f"Saved checkpoint: {out_dir / 'model.pt'}", log_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only_holdout", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--nll_weight", type=float, default=0.2)
    ap.add_argument("--val_nll_weight", type=float, default=0.05)
    ap.add_argument("--min_logvar", type=float, default=-6.0)
    ap.add_argument("--max_logvar", type=float, default=3.5)
    ap.add_argument("--val_end_frac", type=float, default=0.10)
    ap.add_argument("--split_seed", type=int, default=123)
    ap.add_argument("--synth_frac", type=float, default=0.50)
    ap.add_argument("--synth_seed", type=int, default=42)
    ap.add_argument("--out_subdir", default="model_graphtf_gaussian")
    ap.add_argument("--log_every", type=int, default=10)
    args = ap.parse_args()

    real_ds = ValueDataset(
        str(THIS_DIR / "2026" / "Stones.csv"),
        str(THIS_DIR / "2026" / "Ends.csv"),
        augment_positions=False,
        augment_flip=False,
    )
    real_Xp, real_Xc, real_Y = materialize(real_ds)
    if real_ds.cond_dim != 3:
        raise RuntimeError(f"Expected cond_dim=3, got {real_ds.cond_dim}")

    synth_ds = ValueDataset(
        str(THIS_DIR / "valueModel" / "synth_terminal_stones.csv"),
        str(THIS_DIR / "valueModel" / "synth_terminal_ends.csv"),
        augment_positions=False,
        augment_flip=False,
    )
    synth_Xp, synth_Xc, synth_Y = materialize(synth_ds)

    holdouts = [args.only_holdout] if args.only_holdout is not None else HOLDOUT_IDS
    for holdout_comp in holdouts:
        train_one_holdout(args, real_ds, real_Xp, real_Xc, real_Y, synth_Xp, synth_Xc, synth_Y, int(holdout_comp))


if __name__ == "__main__":
    main()
