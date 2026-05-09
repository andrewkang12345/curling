#!/usr/bin/env python3
"""
Evaluate value models on non-final test shots only, using leakage-free splits.

Split protocol:
- test: held-out competition
- val: fixed fraction of end-groups from each remaining competition
- train: the rest of the non-held-out real data, plus a model-specific synth slice

The default config set mirrors the models we just compared:
- SetTransformer small with synth100
- true-span EGNN small with its best prior synth setting (synth50)
- true-span EGNN medium with its best prior synth setting (synth50)
- true-span Graph Transformer small with its best prior synth setting (synth100)
- true-span Graph Transformer medium with its best prior synth setting (synth50)
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "valueModel"))
sys.path.insert(0, str(THIS_DIR))

from dataset import NUM_STONES, POS_MAX, ValueDataset
from gnn_models import GNN_REGISTRY
from new_architectures import ARCHITECTURE_REGISTRY
from train_holdout_models_cond3 import HOLDOUT_IDS, make_holdout_split

FLIP_CENTER_X = 1500.0 / POS_MAX
SYNTH_SEED = 42

CONFIGS = {
    "settf_small": dict(
        arch="set_transformer",
        synth_frac=1.0,
        hidden_dim=128,
        n_layers=3,
        n_heads=4,
        lr=3e-4,
        dropout=0.1,
        weight_decay=1e-4,
    ),
    "settf_moe_medium": dict(
        arch="set_transformer_moe",
        synth_frac=1.0,
        hidden_dim=256,
        n_layers=4,
        n_heads=4,
        n_experts=4,
        lr=2e-4,
        dropout=0.05,
        weight_decay=1e-4,
    ),
    "settf_geo_medium": dict(
        arch="set_transformer_geo",
        synth_frac=1.0,
        hidden_dim=256,
        n_layers=4,
        n_heads=4,
        lr=2e-4,
        dropout=0.05,
        weight_decay=1e-4,
    ),
    "gnn_egnn_small": dict(
        arch="egnn",
        synth_frac=0.5,
        hidden_dim=128,
        n_layers=3,
        n_heads=4,
        lr=3e-4,
        dropout=0.1,
        weight_decay=1e-4,
    ),
    "gnn_egnn_medium": dict(
        arch="egnn",
        synth_frac=0.5,
        hidden_dim=256,
        n_layers=4,
        n_heads=4,
        lr=3e-4,
        dropout=0.1,
        weight_decay=1e-4,
    ),
    "gnn_transformer_small": dict(
        arch="graph_transformer",
        synth_frac=1.0,
        hidden_dim=128,
        n_layers=3,
        n_heads=4,
        lr=3e-4,
        dropout=0.1,
        weight_decay=1e-4,
    ),
    "gnn_transformer_medium": dict(
        arch="graph_transformer",
        synth_frac=0.5,
        hidden_dim=256,
        n_layers=4,
        n_heads=4,
        lr=3e-4,
        dropout=0.1,
        weight_decay=1e-4,
    ),
    "gnn_transformer_medium_thrower_button_span": dict(
        arch="graph_transformer",
        synth_frac=0.5,
        hidden_dim=256,
        n_layers=4,
        n_heads=4,
        lr=3e-4,
        dropout=0.1,
        weight_decay=1e-4,
    ),
    "gnn_transformer_medium_button_node_plus_reach_edge": dict(
        arch="graph_transformer",
        synth_frac=0.5,
        hidden_dim=256,
        n_layers=4,
        n_heads=4,
        lr=3e-4,
        dropout=0.1,
        weight_decay=1e-4,
    ),
    "gnn_transformer_medium_button_edge_plus_reach_edge": dict(
        arch="graph_transformer",
        synth_frac=0.5,
        hidden_dim=256,
        n_layers=4,
        n_heads=4,
        lr=3e-4,
        dropout=0.1,
        weight_decay=1e-4,
    ),
    "gnn_transformer_medium_button_edge_plus_release_reach_edge": dict(
        arch="graph_transformer",
        synth_frac=0.5,
        hidden_dim=256,
        n_layers=4,
        n_heads=4,
        lr=3e-4,
        dropout=0.1,
        weight_decay=1e-4,
    ),
    "gnn_transformer_medium_button_edge_plus_release_reach_edge_product": dict(
        arch="graph_transformer",
        synth_frac=0.5,
        hidden_dim=256,
        n_layers=4,
        n_heads=4,
        lr=3e-4,
        dropout=0.1,
        weight_decay=1e-4,
    ),
    "gnn_transformer_medium_button_edge_release_product_only": dict(
        arch="graph_transformer",
        synth_frac=0.5,
        hidden_dim=256,
        n_layers=4,
        n_heads=4,
        lr=3e-4,
        dropout=0.1,
        weight_decay=1e-4,
    ),
    "gnn_transformer_medium_active_button_region_plus_release_reach_edge_product": dict(
        arch="graph_transformer",
        synth_frac=0.5,
        hidden_dim=256,
        n_layers=4,
        n_heads=4,
        lr=3e-4,
        dropout=0.1,
        weight_decay=1e-4,
    ),
    "gnn_transformer_medium_takeout_feature_stack": dict(
        arch="graph_transformer",
        synth_frac=0.5,
        hidden_dim=256,
        n_layers=4,
        n_heads=4,
        lr=3e-4,
        dropout=0.1,
        weight_decay=1e-4,
    ),
    "gnn_transformer_medium_minkowski_scoring_triplet": dict(
        arch="graph_transformer",
        synth_frac=0.5,
        hidden_dim=256,
        n_layers=4,
        n_heads=4,
        lr=3e-4,
        dropout=0.1,
        weight_decay=1e-4,
    ),
    "gnn_transformer_medium_takeout_products_only": dict(
        arch="graph_transformer",
        synth_frac=0.5,
        hidden_dim=256,
        n_layers=4,
        n_heads=4,
        lr=3e-4,
        dropout=0.1,
        weight_decay=1e-4,
    ),
    "gnn_transformer_medium_takeout_only_on_baseline": dict(
        arch="graph_transformer",
        synth_frac=0.5,
        hidden_dim=256,
        n_layers=4,
        n_heads=4,
        lr=3e-4,
        dropout=0.1,
        weight_decay=1e-4,
    ),
    "gnn_transformer_medium_baseline_edges_plus_rt_node": dict(
        arch="graph_transformer",
        synth_frac=0.5,
        hidden_dim=256,
        n_layers=4,
        n_heads=4,
        lr=3e-4,
        dropout=0.1,
        weight_decay=1e-4,
    ),
    "gnn_transformer_medium_minkowski_scoring_takeout_quintet": dict(
        arch="graph_transformer",
        synth_frac=0.5,
        hidden_dim=256,
        n_layers=4,
        n_heads=4,
        lr=3e-4,
        dropout=0.1,
        weight_decay=1e-4,
    ),
    "gnn_transformer_medium_button_edge_plus_release_reach_edge_product_sparse": dict(
        arch="graph_transformer",
        synth_frac=0.5,
        hidden_dim=256,
        n_layers=4,
        n_heads=4,
        lr=3e-4,
        dropout=0.1,
        weight_decay=1e-4,
    ),
    "gnn_transformer_medium_thrower_mask_only": dict(
        arch="graph_transformer",
        synth_frac=0.5,
        hidden_dim=256,
        n_layers=4,
        n_heads=4,
        lr=3e-4,
        dropout=0.1,
        weight_decay=1e-4,
    ),
    "gnn_transformer_medium_button_region_only": dict(
        arch="graph_transformer",
        synth_frac=0.5,
        hidden_dim=256,
        n_layers=4,
        n_heads=4,
        lr=3e-4,
        dropout=0.1,
        weight_decay=1e-4,
    ),
    "gnn_transformer_medium_button_edge_plus_reach_edge_tune_lr2e4_d005": dict(
        arch="graph_transformer",
        synth_frac=0.5,
        hidden_dim=256,
        n_layers=4,
        n_heads=4,
        lr=2e-4,
        dropout=0.05,
        weight_decay=1e-4,
    ),
    "gnn_transformer_medium_button_edge_plus_reach_edge_tune_lr1e4_d005": dict(
        arch="graph_transformer",
        synth_frac=0.5,
        hidden_dim=256,
        n_layers=4,
        n_heads=4,
        lr=1e-4,
        dropout=0.05,
        weight_decay=1e-4,
    ),
    "gnn_transformer_medium_button_edge_plus_reach_edge_tune_lr2e4_wd3e5": dict(
        arch="graph_transformer",
        synth_frac=0.5,
        hidden_dim=256,
        n_layers=4,
        n_heads=4,
        lr=2e-4,
        dropout=0.10,
        weight_decay=3e-5,
    ),
    "gnn_transformer_large_button_edge_plus_reach_edge_tune": dict(
        arch="graph_transformer",
        synth_frac=0.5,
        hidden_dim=384,
        n_layers=4,
        n_heads=4,
        lr=2e-4,
        dropout=0.05,
        weight_decay=1e-4,
    ),
    "gnn_transformer_medium_exact_line_clearance": dict(
        arch="graph_transformer",
        synth_frac=0.5,
        hidden_dim=256,
        n_layers=4,
        n_heads=4,
        lr=3e-4,
        dropout=0.1,
        weight_decay=1e-4,
    ),
    "gnn_transformer_medium_no_synth_no_final_train": dict(
        arch="graph_transformer",
        synth_frac=0.0,
        hidden_dim=256,
        n_layers=4,
        n_heads=4,
        lr=3e-4,
        dropout=0.1,
        weight_decay=1e-4,
        exclude_real_final_train=True,
        exclude_real_final_val=True,
    ),
}


def augment_flip_batch(x: torch.Tensor) -> torch.Tensor:
    batch_size = x.size(0)
    stones = x.view(batch_size, NUM_STONES, 2).clone()
    flip_mask = (torch.rand(batch_size, device=x.device) < 0.5).view(batch_size, 1, 1)
    in_play = (
        (stones.sum(dim=-1) > 0.001)
        & (stones.max(dim=-1).values < 0.999)
    ).unsqueeze(-1)
    flipped_x = FLIP_CENTER_X - stones[:, :, 0:1]
    new_x = torch.where(flip_mask & in_play, flipped_x, stones[:, :, 0:1])
    stones = torch.cat([new_x, stones[:, :, 1:2]], dim=-1)
    return stones.view(batch_size, -1)


def materialize_dataset(ds: ValueDataset):
    loader = DataLoader(ds, batch_size=8192, shuffle=False, num_workers=0)
    xs, cs, ys = [], [], []
    for x, c, y in loader:
        xs.append(x)
        cs.append(c)
        ys.append(y)
    return torch.cat(xs, 0), torch.cat(cs, 0), torch.cat(ys, 0)


def build_model(arch: str, input_dim: int, cond_dim: int, **kwargs):
    if arch in ARCHITECTURE_REGISTRY:
        return ARCHITECTURE_REGISTRY[arch](input_dim=input_dim, cond_dim=cond_dim, **kwargs)
    if arch in GNN_REGISTRY:
        return GNN_REGISTRY[arch](input_dim=input_dim, cond_dim=cond_dim, **kwargs)
    raise ValueError(f"Unknown architecture: {arch}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def final_shot_mask(df: pd.DataFrame, row_indices: np.ndarray) -> np.ndarray:
    rows = df.iloc[np.asarray(row_indices, dtype=np.int64)]
    shot_index = pd.to_numeric(rows["ShotIndex"], errors="coerce").to_numpy(dtype=np.int64)
    shots_in_end = pd.to_numeric(rows["ShotsInEnd"], errors="coerce").to_numpy(dtype=np.int64)
    return shot_index >= (shots_in_end - 1)


def evaluate_predictions(preds: np.ndarray, targets: np.ndarray, is_final: np.ndarray) -> dict[str, float | int]:
    is_nonfinal = ~is_final
    mse_all = float(np.mean((preds - targets) ** 2))
    mse_final = float(np.mean((preds[is_final] - targets[is_final]) ** 2)) if is_final.any() else float("nan")
    mse_nonfinal = (
        float(np.mean((preds[is_nonfinal] - targets[is_nonfinal]) ** 2))
        if is_nonfinal.any()
        else float("nan")
    )
    return {
        "mse_all": mse_all,
        "mse_final": mse_final,
        "mse_nonfinal": mse_nonfinal,
        "n_all": int(len(preds)),
        "n_final": int(is_final.sum()),
        "n_nonfinal": int(is_nonfinal.sum()),
    }


def train_and_eval(
    train_td: TensorDataset,
    val_td: TensorDataset,
    test_td: TensorDataset,
    test_final_mask: np.ndarray,
    *,
    arch: str,
    input_dim: int,
    cond_dim: int,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    n_layers: int,
    n_heads: int,
    dropout: float,
    seed: int,
) -> dict[str, float | int]:
    set_seed(seed)
    model = build_model(
        arch,
        input_dim=input_dim,
        cond_dim=cond_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    train_loader = DataLoader(
        train_td,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_td,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_td,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    best_val = float("inf")
    best_epoch = 0
    no_improve = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
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

        model.eval()
        running_val = 0.0
        count_val = 0
        with torch.no_grad():
            for x, c, y in val_loader:
                x = x.to(device, non_blocking=True)
                c = c.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                running_val += criterion(model(x, c), y).item() * x.size(0)
                count_val += x.size(0)

        val_mse = running_val / max(1, count_val)
        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if patience > 0 and no_improve >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training never produced a best checkpoint")

    model.load_state_dict(best_state)
    model = model.to(device).eval()

    preds, targets = [], []
    with torch.no_grad():
        for x, c, y in test_loader:
            x = x.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            preds.append(model(x, c).cpu())
            targets.append(y.cpu())

    preds_np = torch.cat(preds, 0).squeeze(-1).numpy()
    targets_np = torch.cat(targets, 0).squeeze(-1).numpy()
    metrics = evaluate_predictions(preds_np, targets_np, test_final_mask)
    metrics["best_val_mse"] = float(best_val)
    metrics["best_epoch"] = int(best_epoch)
    metrics["n_params"] = int(n_params)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--only_configs", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--val_end_frac", type=float, default=0.10)
    ap.add_argument("--split_seed", type=int, default=123)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--only_holdouts", nargs="*", default=None)
    ap.add_argument("--out_dir", type=str, default="")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Val split fraction: {args.val_end_frac:.3f} | split_seed={args.split_seed}", flush=True)

    selected = {
        name: cfg
        for name, cfg in CONFIGS.items()
        if args.only_configs is None or name in args.only_configs
    }
    if not selected:
        raise ValueError("No configs selected")

    real_ds = ValueDataset(
        str(ROOT_DIR / "2026" / "Stones.csv"),
        str(ROOT_DIR / "2026" / "Ends.csv"),
        augment_positions=False,
        augment_flip=False,
    )
    synth_ds = ValueDataset(
        str(ROOT_DIR / "valueModel" / "synth_terminal_stones.csv"),
        str(ROOT_DIR / "valueModel" / "synth_terminal_ends.csv"),
        augment_positions=False,
        augment_flip=False,
    )
    real_Xp, real_Xc, real_Y = materialize_dataset(real_ds)
    synth_Xp, synth_Xc, synth_Y = materialize_dataset(synth_ds)
    input_dim, cond_dim = real_ds.input_dim, real_ds.cond_dim
    n_synth = len(synth_Xp)
    synth_perm = np.random.default_rng(SYNTH_SEED).permutation(n_synth)

    results = []

    for config_name, cfg in selected.items():
        synth_frac = float(cfg["synth_frac"])
        synth_n = min(int(round(synth_frac * n_synth)), n_synth)
        synth_idx = synth_perm[:synth_n] if synth_n > 0 else np.empty((0,), dtype=np.int64)

        fold_results = []
        print(
            f"\n{'=' * 100}\n"
            f"{config_name} | arch={cfg['arch']} | synth_frac={synth_frac:.2f} | synth_rows={len(synth_idx)}\n"
            f"{'=' * 100}",
            flush=True,
        )

        selected_holdouts = HOLDOUT_IDS
        if args.only_holdouts is not None:
            allowed = {int(x) for x in args.only_holdouts}
            selected_holdouts = [c for c in HOLDOUT_IDS if int(c) in allowed]
        for holdout_comp in selected_holdouts:
            train_idx, val_idx, test_idx, _ = make_holdout_split(
                real_ds.df,
                int(holdout_comp),
                args.val_end_frac,
                args.split_seed,
            )

            if cfg.get("exclude_real_final_train", False):
                train_idx = train_idx[~final_shot_mask(real_ds.df, train_idx)]
            if cfg.get("exclude_real_final_val", False):
                val_idx = val_idx[~final_shot_mask(real_ds.df, val_idx)]

            if len(synth_idx) > 0:
                tr_Xp = torch.cat([real_Xp[train_idx], synth_Xp[synth_idx]], dim=0)
                tr_Xc = torch.cat([real_Xc[train_idx], synth_Xc[synth_idx]], dim=0)
                tr_Y = torch.cat([real_Y[train_idx], synth_Y[synth_idx]], dim=0)
            else:
                tr_Xp = real_Xp[train_idx]
                tr_Xc = real_Xc[train_idx]
                tr_Y = real_Y[train_idx]

            train_td = TensorDataset(tr_Xp, tr_Xc, tr_Y)
            val_td = TensorDataset(real_Xp[val_idx], real_Xc[val_idx], real_Y[val_idx])
            test_td = TensorDataset(real_Xp[test_idx], real_Xc[test_idx], real_Y[test_idx])
            test_final = final_shot_mask(real_ds.df, test_idx)

            t0 = time.time()
            metrics = train_and_eval(
                train_td,
                val_td,
                test_td,
                test_final,
                arch=cfg["arch"],
                input_dim=input_dim,
                cond_dim=cond_dim,
                device=device,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                lr=float(cfg["lr"]),
                weight_decay=float(cfg["weight_decay"]),
                hidden_dim=int(cfg["hidden_dim"]),
                n_layers=int(cfg["n_layers"]),
                n_heads=int(cfg["n_heads"]),
                dropout=float(cfg["dropout"]),
                seed=int(args.seed + 1000 * list(selected).index(config_name) + len(results) + int(holdout_comp) % 997),
            )
            elapsed = time.time() - t0

            row = {
                "config_name": config_name,
                "arch": cfg["arch"],
                "synth_frac": synth_frac,
                "holdout_comp": int(holdout_comp),
                "elapsed_sec": float(elapsed),
                "val_end_frac": float(args.val_end_frac),
                "split_seed": int(args.split_seed),
                "exclude_real_final_train": bool(cfg.get("exclude_real_final_train", False)),
                "exclude_real_final_val": bool(cfg.get("exclude_real_final_val", False)),
                **metrics,
            }
            fold_results.append(row)
            results.append(row)
            print(
                f"holdout={holdout_comp} "
                f"all={metrics['mse_all']:.4f} "
                f"nonfinal={metrics['mse_nonfinal']:.4f} "
                f"final={metrics['mse_final']:.4f} "
                f"({metrics['n_nonfinal']}/{metrics['n_all']} non-final) "
                f"best_ep={metrics['best_epoch']} "
                f"{elapsed:.0f}s",
                flush=True,
            )

        fold_df = pd.DataFrame(fold_results)
        fold_summary = fold_df[["mse_all", "mse_nonfinal", "mse_final"]].mean()
        print(
            f">> {config_name} AVG | "
            f"all={fold_summary['mse_all']:.4f} "
            f"nonfinal={fold_summary['mse_nonfinal']:.4f} "
            f"final={fold_summary['mse_final']:.4f}",
            flush=True,
        )

    results_df = pd.DataFrame(results)
    summary_df = (
        results_df.groupby(["config_name", "arch", "synth_frac"], as_index=False)
        .agg(
            mse_all=("mse_all", "mean"),
            mse_nonfinal=("mse_nonfinal", "mean"),
            mse_final=("mse_final", "mean"),
            n_params=("n_params", "first"),
            n_nonfinal=("n_nonfinal", "sum"),
            n_final=("n_final", "sum"),
            best_val_mse=("best_val_mse", "mean"),
            exclude_real_final_train=("exclude_real_final_train", "first"),
            exclude_real_final_val=("exclude_real_final_val", "first"),
        )
        .sort_values("mse_nonfinal")
        .reset_index(drop=True)
    )

    print(f"\n{'=' * 100}\nNON-FINAL TEST SUMMARY\n{'=' * 100}", flush=True)
    print(summary_df.to_string(index=False), flush=True)

    out_dir = Path(args.out_dir) if args.out_dir else (THIS_DIR / "nonfinal_eval_fixed")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / "nonfinal_results.csv", index=False)
    summary_df.to_csv(out_dir / "nonfinal_summary.csv", index=False)
    print(f"\nSaved results to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
