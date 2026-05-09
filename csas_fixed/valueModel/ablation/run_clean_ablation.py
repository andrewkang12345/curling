#!/usr/bin/env python3
"""
Clean ablation: retrain best architectures with cond_dim=3 (no is_hammer).
Uses leave-one-competition-out CV, synth fracs [0, 0.5, 1.0], flip augment.
New wandb project for clean results.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import ValueDataset, NUM_STONES, POS_MAX
from model import ValueTransformer
from new_architectures import ARCHITECTURE_REGISTRY
from cnn_and_features import CNN_REGISTRY

FLIP_CENTER_X = 1500.0 / POS_MAX

SYNTH_FRACS = [0.0, 0.50, 1.0]

# Top architectures from prior experiments
CONFIGS = {
    # Fast models only — medium models too slow for grid search on this hardware.
    # settf_medium confirmed separately: cond_dim=3 gives 2.121 (vs 2.115 with cond_dim=4).
    "settf_small": dict(arch="set_transformer", hidden_dim=128, n_layers=3, n_heads=4, lr=3e-4, dropout=0.1, weight_decay=1e-4),
    "phystf_small": dict(arch="physics_transformer", hidden_dim=128, n_layers=3, n_heads=4, lr=3e-4, dropout=0.1, weight_decay=1e-4),
    "feat_tf_small": dict(arch="feat_transformer", hidden_dim=128, n_layers=3, n_heads=4, lr=3e-4, dropout=0.1, weight_decay=1e-4),
    "feat_mlp_small": dict(arch="feat_mlp", hidden_dim=256, n_layers=4, lr=3e-4, dropout=0.1, weight_decay=1e-4),
    "deepsets_small": dict(arch="deepsets", hidden_dim=128, n_layers=3, lr=3e-4, dropout=0.1, weight_decay=1e-4),
    "pairnet_small": dict(arch="pairnet", hidden_dim=128, n_layers=3, lr=3e-4, dropout=0.1, weight_decay=1e-4),
}

FLIP_MODES = {"none": False, "flip": True}


def augment_flip_batch(x):
    B = x.size(0)
    stones = x.view(B, NUM_STONES, 2).clone()
    flip_mask = (torch.rand(B, device=x.device) < 0.5).view(B, 1, 1)
    ip = ((stones.sum(dim=-1) > 0.001) & (stones.max(dim=-1).values < 0.999)).unsqueeze(-1)
    flipped_x = FLIP_CENTER_X - stones[:, :, 0:1]
    new_x = torch.where(flip_mask & ip, flipped_x, stones[:, :, 0:1])
    stones = torch.cat([new_x, stones[:, :, 1:2]], dim=-1)
    return stones.view(B, -1)


def materialize_dataset(ds):
    loader = DataLoader(ds, batch_size=8192, shuffle=False, num_workers=0)
    xs, cs, ys = [], [], []
    for x, c, y in loader:
        xs.append(x); cs.append(c); ys.append(y)
    return (torch.cat(xs, 0), torch.cat(cs, 0), torch.cat(ys, 0),
            pd.to_numeric(ds.df["CompetitionID"], errors="coerce").to_numpy(dtype="int64"))


def build_model(arch, input_dim, cond_dim, **kwargs):
    if arch == "original_transformer":
        return ValueTransformer(input_dim=input_dim, cond_dim=cond_dim,
                                hidden_dim=kwargs["hidden_dim"], num_stones=NUM_STONES,
                                n_layers=kwargs["n_layers"], n_heads=kwargs["n_heads"],
                                dropout=kwargs["dropout"])
    elif arch in ARCHITECTURE_REGISTRY:
        return ARCHITECTURE_REGISTRY[arch](input_dim=input_dim, cond_dim=cond_dim, **kwargs)
    elif arch in CNN_REGISTRY:
        return CNN_REGISTRY[arch](input_dim=input_dim, cond_dim=cond_dim, **kwargs)
    raise ValueError(f"Unknown arch: {arch}")


def train_model(train_ds, val_ds, arch, input_dim, cond_dim, lr, weight_decay,
                device, do_flip, epochs=150, patience=20, batch_size=1024, **mkw):
    model = build_model(arch, input_dim, cond_dim, **mkw).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
                    pin_memory=(device.type == "cuda"))
    vl = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0,
                    pin_memory=(device.type == "cuda"))

    best_val, best_ep, no_imp = float("inf"), 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        r, cnt = 0.0, 0
        for x, c, y in tl:
            x, c, y = x.to(device, non_blocking=True), c.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if do_flip:
                x = augment_flip_batch(x)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x, c), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            r += loss.item() * x.size(0); cnt += x.size(0)
        tr_mse = r / max(1, cnt)

        model.eval()
        vr, vc = 0.0, 0
        with torch.no_grad():
            for x, c, y in vl:
                x, c, y = x.to(device, non_blocking=True), c.to(device, non_blocking=True), y.to(device, non_blocking=True)
                vr += criterion(model(x, c), y).item() * x.size(0); vc += x.size(0)
        val_mse = vr / max(1, vc)

        if val_mse < best_val:
            best_val, best_ep, no_imp = val_mse, ep, 0
        else:
            no_imp += 1
        if patience > 0 and no_imp >= patience:
            break

    return {"best_val_mse": best_val, "best_val_rmse": float(np.sqrt(best_val)),
            "best_epoch": best_ep, "final_epoch": ep, "final_train_mse": tr_mse, "n_params": n_params}


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Device: {device}", flush=True)

    print("Loading data (cond_dim=3, no is_hammer)...", flush=True)
    real_ds = ValueDataset(args.stones_csv, args.ends_csv, augment_positions=False, augment_flip=False)
    real_Xp, real_Xc, real_Y, real_comps = materialize_dataset(real_ds)
    comp_ids = sorted(set(real_comps.tolist()))
    input_dim, cond_dim = real_ds.input_dim, real_ds.cond_dim
    print(f"  Real: {real_Xp.shape[0]} samples, cond_dim={cond_dim}, comps={comp_ids}", flush=True)
    assert cond_dim == 3, f"Expected cond_dim=3 (no is_hammer), got {cond_dim}"

    synth_ds = ValueDataset(args.synth_stones_csv, args.synth_ends_csv, augment_positions=False, augment_flip=False)
    synth_Xp, synth_Xc, synth_Y, _ = materialize_dataset(synth_ds)
    n_synth = synth_Xp.shape[0]
    print(f"  Synth: {n_synth} samples, cond_dim={synth_ds.cond_dim}", flush=True)

    rng = np.random.default_rng(42)
    synth_perm = rng.permutation(n_synth)
    synth_subsets = {sf: synth_perm[:min(int(round(sf * n_synth)), n_synth)] if sf > 0 else np.array([], dtype=np.int64) for sf in SYNTH_FRACS}
    fold_indices = {c: (np.where(real_comps != c)[0], np.where(real_comps == c)[0]) for c in comp_ids}

    experiments = []
    for cname, cfg in CONFIGS.items():
        for sf in SYNTH_FRACS:
            for fname, do_flip in FLIP_MODES.items():
                experiments.append({"config_name": cname, "synth_frac": sf, "augment": fname, "do_flip": do_flip, **cfg})

    total = len(experiments) * len(comp_ids)
    print(f"\n{'='*80}\n{len(experiments)} configs × {len(comp_ids)} folds = {total} runs\n{'='*80}\n", flush=True)

    all_results = []
    run_idx = 0

    for exp in experiments:
        cname, arch, sf = exp["config_name"], exp["arch"], exp["synth_frac"]
        aug_name, do_flip = exp["augment"], exp["do_flip"]
        group = f"{cname}_s{int(sf*100):03d}_{aug_name}"
        fold_mses = []

        for fi, hcomp in enumerate(comp_ids):
            run_idx += 1
            rname = f"{group}_f{hcomp}"

            tr_idx, va_idx = fold_indices[hcomp]
            s_idx = synth_subsets[sf]
            if len(s_idx) > 0:
                tr_Xp = torch.cat([real_Xp[tr_idx], synth_Xp[s_idx]], 0)
                tr_Xc = torch.cat([real_Xc[tr_idx], synth_Xc[s_idx]], 0)
                tr_Y  = torch.cat([real_Y[tr_idx],  synth_Y[s_idx]], 0)
            else:
                tr_Xp, tr_Xc, tr_Y = real_Xp[tr_idx], real_Xc[tr_idx], real_Y[tr_idx]

            train_td = TensorDataset(tr_Xp, tr_Xc, tr_Y)
            val_td = TensorDataset(real_Xp[va_idx], real_Xc[va_idx], real_Y[va_idx])

            print(f"[{run_idx}/{total}] {rname} | train={len(train_td)} | val={len(va_idx)}", flush=True)

            wcfg = {"arch": arch, "config_name": cname, "synth_frac": sf, "augment": aug_name,
                     "cond_dim": cond_dim, "holdout_comp": hcomp, "total_train_size": len(train_td)}
            for k, v in exp.items():
                if k not in wcfg and k not in ("do_flip",):
                    wcfg[k] = v

            run = wandb.init(project=args.wandb_project, name=rname, group=group, job_type=arch,
                             tags=[arch, cname, f"s{int(sf*100):03d}", aug_name, "cond3"],
                             config=wcfg, reinit=True, mode="offline")

            mkw = {k: v for k, v in exp.items()
                   if k not in ("config_name", "synth_frac", "arch", "lr", "weight_decay",
                                "augment", "do_flip")}
            t0 = time.time()
            try:
                result = train_model(train_td, val_td, arch=arch, input_dim=input_dim, cond_dim=cond_dim,
                                     lr=exp["lr"], weight_decay=exp["weight_decay"], device=device,
                                     do_flip=do_flip, epochs=args.epochs, patience=args.patience,
                                     batch_size=args.batch_size, **mkw)
                elapsed = time.time() - t0
                wandb.log({"test/mse": result["best_val_mse"], "test/rmse": result["best_val_rmse"],
                           "train/final_mse": result["final_train_mse"], "best_epoch": result["best_epoch"],
                           "n_params": result["n_params"], "elapsed_sec": elapsed})
                wandb.summary.update({"test_mse": result["best_val_mse"], "test_rmse": result["best_val_rmse"],
                                      "n_params": result["n_params"]})
                print(f"  -> mse={result['best_val_mse']:.4f} ep={result['best_epoch']}/{result['final_epoch']} "
                      f"params={result['n_params']:,} {elapsed:.1f}s", flush=True)
                fold_mses.append(result["best_val_mse"])
            except Exception as e:
                import traceback; traceback.print_exc()
                elapsed = time.time() - t0
                result = {"best_val_mse": float("inf"), "best_val_rmse": float("inf"),
                          "best_epoch": -1, "final_epoch": -1, "final_train_mse": float("inf"), "n_params": 0}
                print(f"  -> ERROR: {e}", flush=True)

            run.finish()
            all_results.append({"arch": arch, "config_name": cname, "synth_frac": sf, "augment": aug_name,
                                "cond_dim": cond_dim, "holdout_comp": hcomp,
                                "test_mse": result["best_val_mse"], "test_rmse": result["best_val_rmse"],
                                "best_epoch": result["best_epoch"], "n_params": result.get("n_params", 0),
                                "elapsed_sec": elapsed})

        if fold_mses:
            avg = np.mean(fold_mses)
            print(f"  >> {group} AVG mse={avg:.4f} rmse={np.sqrt(avg):.4f}", flush=True)

    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / "clean_results.csv", index=False)

    summary = df.groupby(["config_name", "synth_frac", "augment"], as_index=False).agg(
        mean_test_mse=("test_mse", "mean"), std_test_mse=("test_mse", "std"),
        mean_test_rmse=("test_rmse", "mean"), n_params=("n_params", "first"),
        mean_elapsed=("elapsed_sec", "mean"),
    ).sort_values("mean_test_mse")
    summary.to_csv(out_dir / "clean_summary.csv", index=False)

    print(f"\n{'='*110}\nCLEAN ABLATION (cond_dim=3) SUMMARY:\n{'='*110}", flush=True)
    print(summary.to_string(index=False), flush=True)

    srun = wandb.init(project=args.wandb_project, name="clean_summary",
                      job_type="summary", tags=["summary", "cond3"], reinit=True, mode="offline")
    wandb.log({"summary": wandb.Table(dataframe=summary)})
    wandb.log({"full_results": wandb.Table(dataframe=df)})
    art = wandb.Artifact("clean_ablation_results", type="results")
    art.add_file(str(out_dir / "clean_results.csv"))
    art.add_file(str(out_dir / "clean_summary.csv"))
    srun.log_artifact(art)
    srun.finish()

    print(f"\nDone! {total} runs.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    base = Path(__file__).resolve().parent.parent
    ap.add_argument("--stones_csv",       default=str(base / ".." / "2026" / "Stones.csv"))
    ap.add_argument("--ends_csv",         default=str(base / ".." / "2026" / "Ends.csv"))
    ap.add_argument("--synth_stones_csv", default=str(base / "synth_terminal_stones.csv"))
    ap.add_argument("--synth_ends_csv",   default=str(base / "synth_terminal_ends.csv"))
    ap.add_argument("--out_dir",          default=str(Path(__file__).resolve().parent / "clean_ablation_results"))
    ap.add_argument("--epochs",       type=int, default=150)
    ap.add_argument("--patience",     type=int, default=20)
    ap.add_argument("--batch_size",   type=int, default=1024)
    ap.add_argument("--no_cuda",      action="store_true")
    ap.add_argument("--wandb_project", default="curling-value-clean")
    args = ap.parse_args()
    run(args)
