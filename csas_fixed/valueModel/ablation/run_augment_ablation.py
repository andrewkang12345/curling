#!/usr/bin/env python3
"""
Test top configs with data augmentation (horizontal flip + position shuffle).
Uses BATCH-LEVEL augmentation for speed.
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


# ─────────────────────────────────────────────────────────────
# Batch-level augmentation (vectorized, fast)
# ─────────────────────────────────────────────────────────────

def augment_batch(x, do_flip=True, do_shuffle=False):
    """Apply flip augmentation to a batch. Fully vectorized, fast on GPU. x: (B, 24)"""
    B = x.size(0)
    stones = x.view(B, NUM_STONES, 2).clone()

    if do_flip:
        flip_mask = (torch.rand(B, device=x.device) < 0.5).view(B, 1, 1)
        ip = ((stones.sum(dim=-1) > 0.001) & (stones.max(dim=-1).values < 0.999))
        ip = ip.unsqueeze(-1)
        flipped_x = FLIP_CENTER_X - stones[:, :, 0:1]
        new_x = torch.where(flip_mask & ip, flipped_x, stones[:, :, 0:1])
        stones = torch.cat([new_x, stones[:, :, 1:2]], dim=-1)

    return stones.view(B, -1)


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

SYNTH_FRACS = [0.0, 0.50, 1.0]

CONFIGS = {
    "settf_medium": dict(arch="set_transformer", hidden_dim=256, n_layers=4, n_heads=4, lr=2e-4, dropout=0.05, weight_decay=1e-4),
    "phystf_small": dict(arch="physics_transformer", hidden_dim=128, n_layers=3, n_heads=4, lr=3e-4, dropout=0.1, weight_decay=1e-4),
    "tf_medium_ld": dict(arch="original_transformer", hidden_dim=256, n_layers=4, n_heads=4, lr=2e-4, dropout=0.05, weight_decay=1e-4),
    "feat_mlp_small": dict(arch="feat_mlp", hidden_dim=256, n_layers=4, lr=3e-4, dropout=0.1, weight_decay=1e-4),
    "feat_tf_small": dict(arch="feat_transformer", hidden_dim=128, n_layers=3, n_heads=4, lr=3e-4, dropout=0.1, weight_decay=1e-4),
    "deepsets_small": dict(arch="deepsets", hidden_dim=128, n_layers=3, lr=3e-4, dropout=0.1, weight_decay=1e-4),
}

# None vs flip (the key test: curling sheet left-right symmetry)
AUGMENT_MODES = {
    "none": dict(do_flip=False, do_shuffle=False),
    "flip": dict(do_flip=True, do_shuffle=False),
}


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def materialize_dataset(ds):
    loader = DataLoader(ds, batch_size=8192, shuffle=False, num_workers=0)
    xs, cs, ys = [], [], []
    for x, c, y in loader:
        xs.append(x); cs.append(c); ys.append(y)
    return torch.cat(xs, 0), torch.cat(cs, 0), torch.cat(ys, 0), \
           pd.to_numeric(ds.df["CompetitionID"], errors="coerce").to_numpy(dtype="int64")


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
                device, do_flip, do_shuffle,
                epochs=150, patience=20, batch_size=1024, **mkw):
    model = build_model(arch, input_dim, cond_dim, **mkw).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
                    pin_memory=(device.type == "cuda"))
    vl = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0,
                    pin_memory=(device.type == "cuda"))

    use_aug = do_flip or do_shuffle
    best_val, best_ep, no_imp = float("inf"), 0, 0

    for ep in range(1, epochs + 1):
        model.train()
        r, cnt = 0.0, 0
        for x, c, y in tl:
            x, c, y = x.to(device, non_blocking=True), c.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if use_aug:
                x = augment_batch(x, do_flip=do_flip, do_shuffle=do_shuffle)
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


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Device: {device}", flush=True)

    print("Loading data...", flush=True)
    real_ds = ValueDataset(args.stones_csv, args.ends_csv, augment_positions=False, augment_flip=False)
    real_Xp, real_Xc, real_Y, real_comps = materialize_dataset(real_ds)
    comp_ids = sorted(set(real_comps.tolist()))
    print(f"  Real: {real_Xp.shape[0]}, comps={comp_ids}", flush=True)

    synth_ds = ValueDataset(args.synth_stones_csv, args.synth_ends_csv, augment_positions=False, augment_flip=False)
    synth_Xp, synth_Xc, synth_Y, _ = materialize_dataset(synth_ds)
    n_synth = synth_Xp.shape[0]
    print(f"  Synth: {n_synth}", flush=True)

    input_dim, cond_dim = real_ds.input_dim, real_ds.cond_dim

    rng = np.random.default_rng(42)
    synth_perm = rng.permutation(n_synth)
    synth_subsets = {sf: synth_perm[:min(int(round(sf * n_synth)), n_synth)] if sf > 0 else np.array([], dtype=np.int64) for sf in SYNTH_FRACS}
    fold_indices = {c: (np.where(real_comps != c)[0], np.where(real_comps == c)[0]) for c in comp_ids}

    experiments = []
    for cname, cfg in CONFIGS.items():
        for sf in SYNTH_FRACS:
            for aname, amode in AUGMENT_MODES.items():
                experiments.append({"config_name": cname, "synth_frac": sf, "augment": aname, **cfg, **amode})

    total = len(experiments) * len(comp_ids)
    print(f"\n{'='*80}\n{len(experiments)} configs × {len(comp_ids)} folds = {total} runs\n{'='*80}\n", flush=True)

    all_results = []
    run_idx = 0

    for exp in experiments:
        cname, arch, sf = exp["config_name"], exp["arch"], exp["synth_frac"]
        aug_name = exp["augment"]
        group = f"{cname}_synth{int(sf*100):03d}_aug{aug_name}"
        fold_mses = []

        for fi, hcomp in enumerate(comp_ids):
            run_idx += 1
            rname = f"{group}_fold{hcomp}"

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
                     "holdout_comp": hcomp, "total_train_size": len(train_td)}
            run = wandb.init(project=args.wandb_project, name=rname, group=group, job_type=arch,
                             tags=[arch, cname, f"synth{int(sf*100):03d}", f"aug_{aug_name}", "augment_ablation"],
                             config=wcfg, reinit=True)

            mkw = {k: v for k, v in exp.items()
                   if k not in ("config_name", "synth_frac", "arch", "lr", "weight_decay",
                                "augment", "do_flip", "do_shuffle")}

            t0 = time.time()
            try:
                result = train_model(train_td, val_td, arch=arch, input_dim=input_dim, cond_dim=cond_dim,
                                     lr=exp["lr"], weight_decay=exp["weight_decay"], device=device,
                                     do_flip=exp["do_flip"], do_shuffle=exp["do_shuffle"],
                                     epochs=args.epochs, patience=args.patience, batch_size=args.batch_size, **mkw)
                elapsed = time.time() - t0
                wandb.log({"test/mse": result["best_val_mse"], "test/rmse": result["best_val_rmse"],
                           "best_epoch": result["best_epoch"], "elapsed_sec": elapsed})
                wandb.summary.update({"test_mse": result["best_val_mse"], "test_rmse": result["best_val_rmse"]})
                print(f"  -> mse={result['best_val_mse']:.4f} ep={result['best_epoch']}/{result['final_epoch']} {elapsed:.1f}s", flush=True)
                fold_mses.append(result["best_val_mse"])
            except Exception as e:
                import traceback; traceback.print_exc()
                elapsed = time.time() - t0
                result = {"best_val_mse": float("inf"), "best_val_rmse": float("inf"),
                          "best_epoch": -1, "final_epoch": -1, "final_train_mse": float("inf"), "n_params": 0}
                print(f"  -> ERROR: {e}", flush=True)

            run.finish()
            all_results.append({"arch": arch, "config_name": cname, "synth_frac": sf, "augment": aug_name,
                                "holdout_comp": hcomp, "test_mse": result["best_val_mse"],
                                "test_rmse": result["best_val_rmse"], "best_epoch": result["best_epoch"],
                                "elapsed_sec": elapsed})

        if fold_mses:
            avg = np.mean(fold_mses)
            print(f"  >> {group} AVG mse={avg:.4f} rmse={np.sqrt(avg):.4f}", flush=True)

    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / "augment_results.csv", index=False)

    summary = df.groupby(["config_name", "synth_frac", "augment"], as_index=False).agg(
        mean_test_mse=("test_mse", "mean"), std_test_mse=("test_mse", "std"),
        mean_test_rmse=("test_rmse", "mean"), mean_elapsed=("elapsed_sec", "mean"),
    ).sort_values("mean_test_mse")
    summary.to_csv(out_dir / "augment_summary.csv", index=False)

    print(f"\n{'='*110}\nAUGMENTATION SUMMARY:\n{'='*110}", flush=True)
    print(summary.to_string(index=False), flush=True)

    # Augmentation delta per config
    print(f"\n{'='*110}\nAUGMENT EFFECT (none vs both, at each synth_frac):\n{'='*110}", flush=True)
    for cname in CONFIGS:
        for sf in SYNTH_FRACS:
            sub = summary[(summary["config_name"] == cname) & (summary["synth_frac"] == sf)]
            none_mse = sub[sub["augment"] == "none"]["mean_test_mse"].values
            both_mse = sub[sub["augment"] == "both"]["mean_test_mse"].values
            if len(none_mse) and len(both_mse):
                d = none_mse[0] - both_mse[0]
                print(f"  {cname:20s} synth={sf:.0%}: none={none_mse[0]:.4f} both={both_mse[0]:.4f} delta={d:+.4f}", flush=True)

    srun = wandb.init(project=args.wandb_project, name="augment_summary_v2",
                      job_type="summary", tags=["summary", "augment_ablation"], reinit=True)
    wandb.log({"augment_summary": wandb.Table(dataframe=summary)})
    wandb.log({"augment_full": wandb.Table(dataframe=df)})
    art = wandb.Artifact("augment_results_v2", type="results")
    art.add_file(str(out_dir / "augment_results.csv"))
    art.add_file(str(out_dir / "augment_summary.csv"))
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
    ap.add_argument("--out_dir",          default=str(Path(__file__).resolve().parent / "augment_results"))
    ap.add_argument("--epochs",       type=int, default=150)
    ap.add_argument("--patience",     type=int, default=20)
    ap.add_argument("--batch_size",   type=int, default=1024)
    ap.add_argument("--no_cuda",      action="store_true")
    ap.add_argument("--wandb_project", default="curling-value-ablation")
    args = ap.parse_args()
    run(args)
