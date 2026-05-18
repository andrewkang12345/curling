#!/usr/bin/env python3
"""Train Gaussian GraphTF models from precomputed graph tensors."""

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
from torch.utils.data import DataLoader, Dataset, TensorDataset

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR / "valueModel"))
sys.path.insert(0, str(THIS_DIR / "valueModel" / "ablation"))

os.environ.setdefault("GNN_EDGE_SCALAR_MODE", "button_visible_plus_release_reach_with_product")
os.environ.setdefault("GNN_NODE_FEATURE_MODE", "none")
os.environ.setdefault("GNN_RELEASE_NODE_MODE", "three")

from dataset import ValueDataset, NUM_STONES, POS_MAX  # type: ignore  # noqa: E402
from gnn_models import GNN_REGISTRY, build_graph_batch_fast, compute_edge_features_fast  # type: ignore  # noqa: E402
from train_holdout_models_cond3 import END_KEY, HOLDOUT_IDS, make_holdout_split, materialize, _write_table  # type: ignore  # noqa: E402

FLIP_CENTER_X = 1500.0 / POS_MAX


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def _log(msg: str, log_file: Path | None):
    print(msg, flush=True)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a") as f:
            f.write(msg + "\n")


def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.exp(-logvar) * (y - mean).pow(2) + logvar).mean()


def flip_state_batch(x: torch.Tensor) -> torch.Tensor:
    stones = x.view(x.size(0), NUM_STONES, 2).clone()
    in_play = ((stones.sum(dim=-1) > 0.001) & (stones.max(dim=-1).values < 0.999)).unsqueeze(-1)
    flipped_x = FLIP_CENTER_X - stones[:, :, 0:1]
    new_x = torch.where(in_play, flipped_x, stones[:, :, 0:1])
    return torch.cat([new_x, stones[:, :, 1:2]], dim=-1).view(x.size(0), -1)


def team_swap_state_cond_batch(x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    stones = x.view(x.size(0), NUM_STONES, 2)
    swapped = torch.cat([stones[:, 6:12, :], stones[:, 0:6, :]], dim=1).reshape_as(x)
    c_swapped = c.clone()
    if c_swapped.size(1) >= 3:
        c_swapped[:, 2] = 1.0 - c_swapped[:, 2]
    return swapped, c_swapped


class PrecomputedAugmentDataset(Dataset):
    def __init__(self, caches: list[dict[str, torch.Tensor]]):
        if not caches:
            raise ValueError("PrecomputedAugmentDataset requires at least one cache")
        n = int(caches[0]["y"].shape[0])
        for cache in caches[1:]:
            if int(cache["y"].shape[0]) != n:
                raise ValueError("All augmentation caches must have the same length")
        self.caches = caches
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        aug_idx = int(torch.randint(len(self.caches), (1,)).item())
        cache = self.caches[aug_idx]
        return (
            cache["node_feats"][idx],
            cache["edge_feats"][idx],
            cache["node_mask"][idx],
            cache["c"][idx],
            cache["y"][idx],
        )


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    n = 0
    sum_mse = 0.0
    sum_nll = 0.0
    abs_z = []
    sigma_all = []
    for node_feats, edge_feats, node_mask, c, y in loader:
        node_feats = node_feats.to(device, non_blocking=True).float()
        edge_feats = edge_feats.to(device, non_blocking=True).float()
        node_mask = node_mask.to(device, non_blocking=True)
        c = c.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mean, logvar = model(node_feats, edge_feats, node_mask, c)
        mse = F.mse_loss(mean, y, reduction="sum")
        nll = 0.5 * (torch.exp(-logvar) * (y - mean).pow(2) + logvar).sum()
        sigma = torch.exp(0.5 * logvar)
        z = (y - mean).abs() / sigma.clamp(min=1e-6)
        sum_mse += float(mse.item())
        sum_nll += float(nll.item())
        n += int(y.numel())
        abs_z.append(z.detach().cpu())
        sigma_all.append(sigma.detach().cpu())
    abs_z = torch.cat(abs_z, dim=0)
    sigma_all = torch.cat(sigma_all, dim=0)
    return {
        "mse": sum_mse / max(1, n),
        "rmse": (sum_mse / max(1, n)) ** 0.5,
        "nll": sum_nll / max(1, n),
        "coverage_1sigma": float((abs_z <= 1.0).float().mean().item()),
        "coverage_2sigma": float((abs_z <= 2.0).float().mean().item()),
        "mean_sigma": float(sigma_all.mean().item()),
        "median_sigma": float(sigma_all.median().item()),
    }


@torch.no_grad()
def _precompute_graph_tensors(
    x: torch.Tensor,
    c: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    node_feats_all = []
    edge_feats_all = []
    node_mask_all = []
    for start in range(0, x.shape[0], batch_size):
        xb = x[start:start + batch_size].to(device, non_blocking=True)
        cb = c[start:start + batch_size].to(device, non_blocking=True)
        node_feats, node_coords, node_mask, _n = build_graph_batch_fast(xb, device)
        edge_feats = compute_edge_features_fast(node_coords, node_feats, node_mask, c=cb)
        node_feats_all.append(node_feats.cpu().to(torch.float16))
        edge_feats_all.append(edge_feats.cpu().to(torch.float16))
        node_mask_all.append(node_mask.cpu())
    return {
        "node_feats": torch.cat(node_feats_all, dim=0),
        "edge_feats": torch.cat(edge_feats_all, dim=0),
        "node_mask": torch.cat(node_mask_all, dim=0),
        "c": c.clone().cpu(),
        "y": y.clone().cpu(),
    }


def _load_or_build_cache(
    cache_path: Path,
    x: torch.Tensor,
    c: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    batch_size: int,
    force_rebuild: bool,
    log_path: Path | None,
) -> dict[str, torch.Tensor]:
    if cache_path.exists() and not force_rebuild:
        _log(f"Loading cache: {cache_path}", log_path)
        return torch.load(cache_path, map_location="cpu")
    _log(f"Building cache: {cache_path}", log_path)
    t0 = time.time()
    cache = _precompute_graph_tensors(x, c, y, device, batch_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    _log(f"Built cache in {time.time() - t0:.1f}s: {cache_path}", log_path)
    return cache


def _train_augmented_caches(
    cache_dir: Path,
    x: torch.Tensor,
    c: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    batch_size: int,
    force_rebuild: bool,
    use_flip: bool,
    use_team_swap: bool,
    log_path: Path | None,
) -> tuple[list[dict[str, torch.Tensor]], list[str]]:
    variant_names: list[str] = ["orig"]
    variants: list[tuple[str, torch.Tensor, torch.Tensor]] = [("orig", x, c)]
    if use_flip:
        variants.append(("flip", flip_state_batch(x), c))
        variant_names.append("flip")
    if use_team_swap:
        swap_x, swap_c = team_swap_state_cond_batch(x, c)
        variants.append(("swap", swap_x, swap_c))
        variant_names.append("swap")
    if use_flip and use_team_swap:
        flip_swap_x, flip_swap_c = team_swap_state_cond_batch(flip_state_batch(x), c)
        variants.append(("flip_swap", flip_swap_x, flip_swap_c))
        variant_names.append("flip_swap")

    caches = []
    for name, x_var, c_var in variants:
        caches.append(
            _load_or_build_cache(
                cache_dir / f"train_cache_{name}.pt",
                x_var,
                c_var,
                y,
                device,
                batch_size,
                force_rebuild,
                log_path,
            )
        )
    return caches, variant_names


def _checkpoint_payload(args, model, optimizer, real_ds, epoch, best_epoch, best_val_key, val_metrics):
    return {
        "arch": "graph_transformer_gaussian_precomputed",
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "best_val_key": float(best_val_key),
        "val_metrics": val_metrics,
        "model_state_dict": _unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "input_dim": int(real_ds.input_dim),
        "cond_dim": int(real_ds.cond_dim),
        "hidden_dim": int(args.hidden_dim),
        "num_stones": int(NUM_STONES),
        "model_class": "ValueGraphTransformerGaussianPrecomputed",
        "args": vars(args),
        "graph_feature_env": {
            "GNN_EDGE_SCALAR_MODE": os.environ.get("GNN_EDGE_SCALAR_MODE"),
            "GNN_NODE_FEATURE_MODE": os.environ.get("GNN_NODE_FEATURE_MODE"),
            "GNN_RELEASE_NODE_MODE": os.environ.get("GNN_RELEASE_NODE_MODE"),
        },
    }


def train_one_holdout(args, real_ds, real_Xp, real_Xc, real_Y, synth_Xp, synth_Xc, synth_Y, holdout_comp: int):
    run_dir = THIS_DIR / "holdouts" / str(holdout_comp)
    out_dir = run_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    if log_path.exists() and not args.resume:
        log_path.unlink()

    device = torch.device(args.device)
    _log(f"Device: {device}", log_path)
    _log(f"Holdout test competition: {holdout_comp}", log_path)
    _log(
        "Graph feature env | "
        f"GNN_EDGE_SCALAR_MODE={os.environ.get('GNN_EDGE_SCALAR_MODE')} "
        f"GNN_NODE_FEATURE_MODE={os.environ.get('GNN_NODE_FEATURE_MODE')} "
        f"GNN_RELEASE_NODE_MODE={os.environ.get('GNN_RELEASE_NODE_MODE')} "
        f"augment_flip={not args.no_augment_flip} "
        f"augment_team_swap={not args.no_augment_team_swap} (precomputed path)",
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
    va_Xp, va_Xc, va_Y = real_Xp[val_idx], real_Xc[val_idx], real_Y[val_idx]
    te_Xp, te_Xc, te_Y = real_Xp[test_idx], real_Xc[test_idx], real_Y[test_idx]

    _log(f"Real split sizes | train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}", log_path)
    _log(f"Train+synth size={tr_Xp.shape[0]} (synth_used={len(synth_idx)})", log_path)

    cache_dir = out_dir / "cache"
    train_caches, train_variants = _train_augmented_caches(
        cache_dir,
        tr_Xp,
        tr_Xc,
        tr_Y,
        device,
        args.cache_batch_size,
        args.rebuild_cache,
        use_flip=not args.no_augment_flip,
        use_team_swap=not args.no_augment_team_swap,
        log_path=log_path,
    )
    val_cache = _load_or_build_cache(
        cache_dir / "val_cache.pt", va_Xp, va_Xc, va_Y, device, args.cache_batch_size, args.rebuild_cache, log_path
    )
    test_cache = _load_or_build_cache(
        cache_dir / "test_cache.pt", te_Xp, te_Xc, te_Y, device, args.cache_batch_size, args.rebuild_cache, log_path
    )

    train_td = PrecomputedAugmentDataset(train_caches)
    val_td = TensorDataset(
        val_cache["node_feats"], val_cache["edge_feats"], val_cache["node_mask"], val_cache["c"], val_cache["y"]
    )
    test_td = TensorDataset(
        test_cache["node_feats"], test_cache["edge_feats"], test_cache["node_mask"], test_cache["c"], test_cache["y"]
    )

    cfg = dict(
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        min_logvar=args.min_logvar,
        max_logvar=args.max_logvar,
    )
    model = GNN_REGISTRY["graph_transformer_gaussian_precomputed"](
        input_dim=real_ds.input_dim,
        cond_dim=real_ds.cond_dim,
        **cfg,
    ).to(device)
    if args.data_parallel:
        if device.type != "cuda":
            raise RuntimeError("--data_parallel requires a CUDA device")
        if torch.cuda.device_count() < 2:
            raise RuntimeError(f"--data_parallel requested but only {torch.cuda.device_count()} CUDA device(s) visible")
        model = nn.DataParallel(model)
        _log(f"DataParallel enabled across {torch.cuda.device_count()} visible CUDA devices", log_path)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log(f"Model params: {n_params:,}", log_path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(train_td, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    eval_batch_size = max(1, int(args.batch_size * args.eval_batch_mult))
    val_loader = DataLoader(val_td, batch_size=eval_batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_td, batch_size=eval_batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    best_key = float("inf")
    best_ep = 0
    best_state = None
    no_imp = 0
    start_ep = 1

    if args.resume:
        last_path = out_dir / "last.pt"
        if not last_path.exists():
            raise FileNotFoundError(f"--resume requested but missing {last_path}")
        last_ckpt = torch.load(last_path, map_location=device)
        _unwrap_model(model).load_state_dict(last_ckpt["model_state_dict"])
        if last_ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(last_ckpt["optimizer_state_dict"])
        start_ep = int(last_ckpt.get("epoch", 0)) + 1
        best_key = float(last_ckpt.get("best_val_key", float("inf")))
        best_ep = int(last_ckpt.get("best_epoch", 0))
        best_state = {k: v.detach().cpu().clone() for k, v in _unwrap_model(model).state_dict().items()}
        no_imp = max(0, int(last_ckpt.get("epoch", 0)) - int(best_ep))

    for ep in range(start_ep, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_mse = 0.0
        running_nll = 0.0
        count = 0
        for node_feats, edge_feats, node_mask, c, y in train_loader:
            node_feats = node_feats.to(device, non_blocking=True).float()
            edge_feats = edge_feats.to(device, non_blocking=True).float()
            node_mask = node_mask.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            mean, logvar = model(node_feats, edge_feats, node_mask, c)
            mse = F.mse_loss(mean, y)
            nll = gaussian_nll(mean, logvar, y)
            loss = mse + args.nll_weight * nll
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            bsz = y.size(0)
            running_loss += loss.item() * bsz
            running_mse += mse.item() * bsz
            running_nll += nll.item() * bsz
            count += bsz

        val_metrics = _evaluate(model, val_loader, device)
        val_key = val_metrics["mse"] + args.val_nll_weight * val_metrics["nll"]
        if val_key < best_key:
            best_key = val_key
            best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in _unwrap_model(model).state_dict().items()}
            torch.save(
                _checkpoint_payload(args, model, optimizer, real_ds, ep, best_ep, best_key, val_metrics),
                out_dir / "best.pt",
            )
            no_imp = 0
        else:
            no_imp += 1

        torch.save(
            _checkpoint_payload(args, model, optimizer, real_ds, ep, best_ep, best_key, val_metrics),
            out_dir / "last.pt",
        )

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
    _unwrap_model(model).load_state_dict(best_state)
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
        "augmentation": {
            "horizontal_flip": not bool(args.no_augment_flip),
            "team_slot_swap": not bool(args.no_augment_team_swap),
        },
        "graph_feature_env": {
            "GNN_EDGE_SCALAR_MODE": os.environ.get("GNN_EDGE_SCALAR_MODE"),
            "GNN_NODE_FEATURE_MODE": os.environ.get("GNN_NODE_FEATURE_MODE"),
            "GNN_RELEASE_NODE_MODE": os.environ.get("GNN_RELEASE_NODE_MODE"),
        },
        "cache": {
            "cache_batch_size": int(args.cache_batch_size),
            "dtype": "float16",
            "train_variants": train_variants,
        },
    }
    ckpt = {
        "arch": "graph_transformer_gaussian_precomputed",
        "epoch": int(best_ep),
        "model_state_dict": best_state,
        "input_dim": int(real_ds.input_dim),
        "cond_dim": int(real_ds.cond_dim),
        "hidden_dim": int(args.hidden_dim),
        "num_stones": int(NUM_STONES),
        "model_class": "ValueGraphTransformerGaussianPrecomputed",
        "split_info": split_info,
        "args": vars(args),
    }
    torch.save(ckpt, out_dir / "model.pt")
    torch.save(ckpt, out_dir / "best_final.pt")
    (out_dir / "split_summary.json").write_text(json.dumps(split_info, indent=2, sort_keys=True) + "\n")
    _log(f"Saved checkpoint: {out_dir / 'model.pt'}", log_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only_holdout", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--eval_batch_mult", type=float, default=2.0)
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
    ap.add_argument("--out_subdir", default="model_graphtf_gaussian_precomputed")
    ap.add_argument("--cache_batch_size", type=int, default=256)
    ap.add_argument("--rebuild_cache", action="store_true")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--data_parallel", action="store_true")
    ap.add_argument("--no_augment_flip", action="store_true")
    ap.add_argument("--no_augment_team_swap", action="store_true")
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
