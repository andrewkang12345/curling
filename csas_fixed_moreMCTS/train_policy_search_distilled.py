#!/usr/bin/env python3
"""Train pi_{k+1} from weighted continuous search-policy targets."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from common import (
    ACTION_ANGLE_MAX,
    ACTION_ANGLE_MIN,
    ACTION_COLS,
    ACTION_SPEED_MAX,
    ACTION_SPEED_MIN,
    ACTION_SPIN_MAX,
    ACTION_SPIN_MIN,
    ACTION_Y0_MAX,
    ACTION_Y0_MIN,
    FIXED_ROOT,
    KEY_COLS,
    NUM_STONES,
    POS_MAX,
    STONE_COLS,
    log,
    next_condition,
    random_flip_state_action_z,
    random_team_swap_state_cond,
    set_seed,
)
from dataset import ValueDataset
from generate_horizon_curriculum_targets import score_end_value
from kr_uct_search import _simulate_candidates, evaluate_states, load_policy, load_value_model
from policy_dataset import load_inverse_estimates
from policy_model import PolicySetTransformerMDN
from preplaced_value_data import load_preplaced_mcts_roots
from train_holdout_models_cond3 import make_holdout_split


def _safe_metric_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "reference"


def _parse_reference_policy(spec: str) -> tuple[str, str]:
    if "=" in spec:
        name, path = spec.split("=", 1)
        return _safe_metric_name(name), path
    path = spec
    return _safe_metric_name(Path(path).stem), path


def _load_policy_ckpt(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    arch = str(ckpt.get("arch", "policy_set_transformer_mdn"))
    if arch in {"policy_graph_transformer_mdn", "policy_graph_transformer_fullcov_mdn"}:
        graph_env = ckpt.get("graph_feature_env") or {}
        for key, value in graph_env.items():
            if value is not None:
                os.environ[str(key)] = str(value)
        sys.path.insert(0, str(FIXED_ROOT / "valueModel"))
        sys.path.insert(0, str(FIXED_ROOT / "valueModel" / "ablation"))
        from policy_graph_model import PolicyGraphTransformerFullCovMDN, PolicyGraphTransformerMDN

        model_cls = (
            PolicyGraphTransformerFullCovMDN
            if arch == "policy_graph_transformer_fullcov_mdn"
            else PolicyGraphTransformerMDN
        )
        model = model_cls(
            input_dim=ckpt.get("input_dim", 24),
            cond_dim=ckpt.get("cond_dim", 3),
            action_dim=ckpt.get("action_dim", 4),
            hidden_dim=args.get("hidden_dim", 256),
            n_layers=args.get("n_layers", 4),
            n_heads=args.get("n_heads", 4),
            dropout=args.get("dropout", 0.10),
            n_mixtures=args.get("n_mixtures", 16),
        ).to(device)
    else:
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


def _mdn_nll_from_outputs(pi_logits: torch.Tensor, mu: torch.Tensor, scale: torch.Tensor, action_z: torch.Tensor) -> torch.Tensor:
    if scale.ndim == 4:
        delta = action_z.unsqueeze(1) - mu
        whitened = torch.linalg.solve_triangular(scale, delta.unsqueeze(-1), upper=False).squeeze(-1)
        log_det = torch.log(torch.diagonal(scale, dim1=-2, dim2=-1)).sum(-1)
        comp_logp = -0.5 * whitened.square().sum(-1) - log_det - 0.5 * mu.shape[-1] * np.log(2.0 * np.pi)
        return -torch.logsumexp(torch.log_softmax(pi_logits, dim=-1) + comp_logp, dim=-1)
    log_std = scale
    target = action_z.unsqueeze(1)
    inv_var = torch.exp(-2.0 * log_std)
    comp_logp = -0.5 * ((target - mu).pow(2) * inv_var).sum(-1)
    comp_logp = comp_logp - log_std.sum(-1) - 0.5 * mu.shape[-1] * np.log(2.0 * np.pi)
    return -torch.logsumexp(torch.log_softmax(pi_logits, dim=-1) + comp_logp, dim=-1)


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


@torch.no_grad()
def _sample_actions_from_model(model, action_mean, action_std, x, c, n, device, temperature, std_scale, global_frac=0.0):
    z = model.sample_z(
        torch.as_tensor(x[None], dtype=torch.float32, device=device),
        torch.as_tensor(c[None], dtype=torch.float32, device=device),
        n_samples=n,
        temperature=temperature,
        std_scale=std_scale,
    )[0]
    actions = z * action_std.to(device) + action_mean.to(device)
    if global_frac > 0.0:
        m = int(round(n * global_frac))
        if m > 0:
            global_actions = actions[:m].clone()
            global_actions[:, 0] = torch.empty(m, device=device).uniform_(ACTION_SPEED_MIN, ACTION_SPEED_MAX)
            global_actions[:, 1] = torch.empty(m, device=device).uniform_(ACTION_ANGLE_MIN, ACTION_ANGLE_MAX)
            global_actions[:, 2] = torch.empty(m, device=device).uniform_(ACTION_SPIN_MIN, ACTION_SPIN_MAX)
            global_actions[:, 3] = torch.empty(m, device=device).uniform_(ACTION_Y0_MIN, ACTION_Y0_MAX)
            actions[:m] = global_actions
    actions[:, 0] = actions[:, 0].clamp(ACTION_SPEED_MIN, ACTION_SPEED_MAX)
    actions[:, 1] = actions[:, 1].clamp(ACTION_ANGLE_MIN, ACTION_ANGLE_MAX)
    actions[:, 2] = actions[:, 2].clamp(ACTION_SPIN_MIN, ACTION_SPIN_MAX)
    actions[:, 3] = actions[:, 3].clamp(ACTION_Y0_MIN, ACTION_Y0_MAX)
    return np.asarray(actions.detach().cpu().tolist(), dtype=np.float32)


def _row_positions(row: pd.Series) -> np.ndarray:
    return row[STONE_COLS].to_numpy(dtype=np.float32).reshape(NUM_STONES, 2)


def _row_condition(row: pd.Series) -> np.ndarray:
    return np.asarray([row["shot_norm"], row["team_order"], row["stone_block"]], dtype=np.float32)


def _stage_roots(ds: ValueDataset, holdout: int, split: str, horizon: int, include_preplaced: bool, max_roots: int):
    train_idx, val_idx, test_idx, _ = make_holdout_split(ds.df, holdout, 0.10, 123)
    base_idx = {"train": train_idx, "val": val_idx, "test": test_idx}[split]
    df = ds.df.copy()
    group_cols = ["CompetitionID", "SessionID", "GameID", "EndID"]
    df["_prev_ds_idx"] = pd.Series(df.index, index=df.index).groupby([df[c] for c in group_cols], sort=False).shift(1)
    subset = df.iloc[np.asarray(base_idx, dtype=np.int64)].copy()
    subset = subset[subset["_prev_ds_idx"].notna()].copy()
    subset = subset[(subset["ShotsInEnd"].astype(int) - subset["ShotIndex"].astype(int)) == int(horizon)].copy()
    roots = []
    for _, row in subset.iterrows():
        prev = ds.df.iloc[int(row["_prev_ds_idx"])]
        roots.append(
            {
                "state_norm": (_row_positions(prev).reshape(-1) / POS_MAX).astype(np.float32),
                "cond": _row_condition(row),
                "ShotIndex": int(row["ShotIndex"]),
                "ShotsInEnd": int(row["ShotsInEnd"]),
                "thrower_block": int(round(float(row["stone_block"]))),
            }
        )
        if max_roots > 0 and len(roots) >= max_roots:
            break
    if include_preplaced:
        train_comps = set(int(x) for x in pd.unique(ds.df.iloc[train_idx]["CompetitionID"]).tolist())
        for root in load_preplaced_mcts_roots(train_comps, ds.df):
            if int(root["ShotsInEnd"]) - int(root["ShotIndex"]) == int(horizon):
                roots.append({**root, "thrower_block": int(round(float(root["cond"][2])))})
                if max_roots > 0 and len(roots) >= max_roots:
                    break
    return roots


def _human_roots(ds: ValueDataset, inv: pd.DataFrame, holdout: int, split: str, horizon: int, max_roots: int):
    """Roots at this horizon that have a usable inverse-estimated human action."""
    train_idx, val_idx, test_idx, _ = make_holdout_split(ds.df, holdout, 0.10, 123)
    base_idx = {"train": train_idx, "val": val_idx, "test": test_idx}[split]
    df = ds.df.copy()
    group_cols = ["CompetitionID", "SessionID", "GameID", "EndID"]
    df["_prev_ds_idx"] = pd.Series(df.index, index=df.index).groupby([df[c] for c in group_cols], sort=False).shift(1)
    subset = df.iloc[np.asarray(base_idx, dtype=np.int64)].copy()
    subset = subset[subset["_prev_ds_idx"].notna()].copy()
    subset = subset[(subset["ShotsInEnd"].astype(int) - subset["ShotIndex"].astype(int)) == int(horizon)].copy()
    merged = subset.merge(inv, on=KEY_COLS, how="inner")
    roots = []
    for _, row in merged.iterrows():
        prev = ds.df.iloc[int(row["_prev_ds_idx"])]
        human_action = np.asarray([float(row[c]) for c in ACTION_COLS], dtype=np.float32)
        roots.append(
            {
                "state_norm": (_row_positions(prev).reshape(-1) / POS_MAX).astype(np.float32),
                "cond": _row_condition(row),
                "ShotIndex": int(row["ShotIndex"]),
                "ShotsInEnd": int(row["ShotsInEnd"]),
                "thrower_block": int(round(float(row["stone_block"]))),
                "human_action": human_action,
            }
        )
        if max_roots > 0 and len(roots) >= max_roots:
            break
    return roots


@torch.no_grad()
def _human_eval_obj(policy_model, action_mean, action_std, value_model, roots, horizon: int, device: torch.device, n_candidates: int, batch_size: int, temperature: float, std_scale: float, noise=None, noise_samples: int = 0):
    """Per-root (policy_q, human_q).

    If noise_samples<=0 or noise is None: policy_q = argmax over n_candidates deterministic sims.
    If noise_samples>0 and noise is not None: policy_q = argmax over n_candidates *expected*
    values, each averaging Q over noise_samples noisy executions of the candidate (Student-t
    LocalNoise from v1_bowling.json — the same noise model used in target generation).
    Human side always uses the inverse-estimated action with NO execution noise added.
    """
    action_std_np = action_std.detach().cpu().numpy().astype(np.float32).reshape(-1)
    denom = np.where(action_std_np > 1e-6, action_std_np, 1.0)
    use_noise = noise is not None and int(noise_samples) > 0
    deltas, action_l2s, blocks, pol_qs, hum_qs = [], [], [], [], []
    for root in roots:
        x = root["state_norm"]
        c = root["cond"]
        actions = _sample_actions_from_model(policy_model, action_mean, action_std, x, c, n_candidates, device, temperature, std_scale, 0.0)
        if use_noise:
            noisy = noise.sample_batch(actions, int(noise_samples))                 # (n_candidates, K, 4)
            K = int(noisy.shape[1])
            flat = noisy.reshape(n_candidates * K, 4).astype(np.float32)
            posts_flat = _simulate_candidates(x, c, flat)
            if horizon <= 1:
                q_flat = np.asarray([score_end_value(p, int(round(float(c[2])))) for p in posts_flat], dtype=np.float32)
            else:
                q_flat = -evaluate_states(value_model, posts_flat, next_condition(c, shots_in_end=int(root["ShotsInEnd"])), device, batch_size).astype(np.float32)
            q_per_candidate = q_flat.reshape(n_candidates, K).mean(axis=1)
            best_idx = int(np.argmax(q_per_candidate))
            pol_q = float(q_per_candidate[best_idx])
            pol_action = actions[best_idx].astype(np.float32)
        else:
            posts = _simulate_candidates(x, c, actions)
            if horizon <= 1:
                q = np.asarray([score_end_value(p, int(round(float(c[2])))) for p in posts], dtype=np.float32)
            else:
                q = -evaluate_states(value_model, posts, next_condition(c, shots_in_end=int(root["ShotsInEnd"])), device, batch_size).astype(np.float32)
            best_idx = int(np.argmax(q))
            pol_q = float(q[best_idx])
            pol_action = actions[best_idx].astype(np.float32)
        hum_posts = _simulate_candidates(x, c, root["human_action"][None, :])
        if horizon <= 1:
            hum_q = float(score_end_value(hum_posts[0], int(round(float(c[2])))))
        else:
            hum_q = float(-evaluate_states(value_model, hum_posts, next_condition(c, shots_in_end=int(root["ShotsInEnd"])), device, batch_size)[0])
        deltas.append(pol_q - hum_q)
        action_l2s.append(float(np.linalg.norm((pol_action - root["human_action"]) / denom)))
        blocks.append(int(root["thrower_block"]))
        pol_qs.append(pol_q)
        hum_qs.append(hum_q)
    return (np.asarray(deltas, dtype=np.float32), np.asarray(action_l2s, dtype=np.float32), np.asarray(blocks, dtype=np.int32),
            np.asarray(pol_qs, dtype=np.float32), np.asarray(hum_qs, dtype=np.float32))


def _human_bucketed_metrics(deltas: np.ndarray, action_l2s: np.ndarray, blocks: np.ndarray, pol_qs: np.ndarray, hum_qs: np.ndarray, prefix: str = "human_eval") -> dict[str, float]:
    out: dict[str, float] = {}
    for name, mask in [
        ("overall", np.ones(len(deltas), dtype=bool)),
        ("thrower_block_0", blocks == 0),
        ("thrower_block_1", blocks == 1),
    ]:
        if not np.any(mask):
            continue
        d = deltas[mask]
        out[f"{prefix}/{name}/n_roots"] = float(int(mask.sum()))
        out[f"{prefix}/{name}/mean_delta_q"] = float(d.mean())
        out[f"{prefix}/{name}/policy_mean_q"] = float(pol_qs[mask].mean())
        out[f"{prefix}/{name}/human_mean_q"] = float(hum_qs[mask].mean())
        out[f"{prefix}/{name}/new_win_rate"] = float((d > 1e-6).mean())
        out[f"{prefix}/{name}/old_win_rate"] = float((d < -1e-6).mean())
        out[f"{prefix}/{name}/mean_action_l2"] = float(action_l2s[mask].mean())
    return out


def _preplaced_roots(ds: ValueDataset, holdout: int, max_per_block: int):
    train_idx, _, _, _ = make_holdout_split(ds.df, holdout, 0.10, 123)
    train_comps = set(int(x) for x in pd.unique(ds.df.iloc[train_idx]["CompetitionID"]).tolist())
    roots = load_preplaced_mcts_roots(train_comps, ds.df)
    buckets = {0: [], 1: []}
    for root in roots:
        block = int(round(float(root["cond"][2])))
        if len(buckets[block]) < max_per_block:
            buckets[block].append(root)
    return buckets[0] + buckets[1]


@torch.no_grad()
def _best_q_for_policy_obj(policy_model, action_mean, action_std, value_model, roots, horizon: int, device: torch.device, n_candidates: int, batch_size: int, temperature: float, std_scale: float):
    vals = []
    for root in roots:
        x = root["state_norm"]
        c = root["cond"]
        actions = _sample_actions_from_model(policy_model, action_mean, action_std, x, c, n_candidates, device, temperature, std_scale, 0.0)
        posts = _simulate_candidates(x, c, actions)
        if horizon <= 1:
            q = np.asarray([score_end_value(p, int(round(float(c[2])))) for p in posts], dtype=np.float32)
        else:
            q = -evaluate_states(value_model, posts, next_condition(c, shots_in_end=int(root["ShotsInEnd"])), device, batch_size).astype(np.float32)
        vals.append(float(np.max(q)))
    return np.asarray(vals, dtype=np.float32)


@torch.no_grad()
def _rollout_policy_obj(policy_model, action_mean, action_std, value_model, roots, device: torch.device, n_candidates: int, batch_size: int, temperature: float, std_scale: float):
    final_vals = []
    for root in roots:
        state = np.asarray(root["state_norm"], dtype=np.float32).copy()
        cond = np.asarray(root["cond"], dtype=np.float32).copy()
        root_block = int(round(float(cond[2])))
        shot_index = int(root["ShotIndex"])
        shots_in_end = int(root["ShotsInEnd"])
        while True:
            throws_remaining = int(shots_in_end - shot_index)
            actions = _sample_actions_from_model(policy_model, action_mean, action_std, state, cond, n_candidates, device, temperature, std_scale, 0.0)
            posts = _simulate_candidates(state, cond, actions)
            if throws_remaining <= 1:
                q = np.asarray([score_end_value(p, int(round(float(cond[2])))) for p in posts], dtype=np.float32)
            else:
                q = -evaluate_states(value_model, posts, next_condition(cond, shots_in_end=shots_in_end), device, batch_size).astype(np.float32)
            best = int(np.argmax(q))
            state = posts[best].astype(np.float32)
            shot_index += 1
            if shot_index >= shots_in_end:
                break
            cond = next_condition(cond, shots_in_end=shots_in_end)
        final_vals.append(float(score_end_value(state, root_block)))
    return np.asarray(final_vals, dtype=np.float32)


def _bucketed_metrics(old_vals: np.ndarray, new_vals: np.ndarray, blocks: np.ndarray, prefix: str) -> dict[str, float]:
    out = {}
    for name, mask in [
        ("overall", np.ones(len(old_vals), dtype=bool)),
        ("thrower_block_0", blocks == 0),
        ("thrower_block_1", blocks == 1),
    ]:
        if not np.any(mask):
            continue
        ov = old_vals[mask]
        nv = new_vals[mask]
        out[f"{prefix}/{name}/old_mean"] = float(np.mean(ov))
        out[f"{prefix}/{name}/new_mean"] = float(np.mean(nv))
        out[f"{prefix}/{name}/delta_mean"] = float(np.mean(nv - ov))
        out[f"{prefix}/{name}/new_win_rate"] = float(np.mean(nv > ov))
        out[f"{prefix}/{name}/old_win_rate"] = float(np.mean(nv < ov))
        out[f"{prefix}/{name}/tie_rate"] = float(np.mean(np.isclose(nv, ov, atol=1e-6)))
    return out


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
    ap.add_argument("--data_parallel", action="store_true")
    ap.add_argument("--holdout", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--eval_value", default=None)
    ap.add_argument("--original_policy", default=None)
    ap.add_argument(
        "--reference_policy",
        action="append",
        default=[],
        help="Extra fixed reference policy for W&B eval, as name=/path/to/model.pt. May be repeated.",
    )
    ap.add_argument("--include_preplaced_eval", action="store_true")
    ap.add_argument("--wandb_project", default=None)
    ap.add_argument("--wandb_run_name", default=None)
    ap.add_argument("--wandb_eval_interval", type=int, default=5)
    ap.add_argument("--wandb_max_stage_roots", type=int, default=48)
    ap.add_argument("--wandb_max_preplaced_per_block", type=int, default=8)
    ap.add_argument("--eval_policy_candidates", type=int, default=64)
    ap.add_argument("--eval_batch_size", type=int, default=2048)
    ap.add_argument("--eval_temperature", type=float, default=1.0)
    ap.add_argument("--eval_std_scale", type=float, default=1.25)
    ap.add_argument(
        "--stop_on_metric",
        choices=["none", "val_nll", "stage_win_rate", "stage_delta_mean", "preplaced_win_rate", "preplaced_delta_mean"],
        default="val_nll",
        help="Primary early-stop metric. Win-rate/delta metrics are maximized; val_nll is minimized.",
    )
    ap.add_argument(
        "--metric_patience_evals",
        type=int,
        default=0,
        help="If > 0 and stop_on_metric != val_nll, stop after this many eval checkpoints without improvement.",
    )
    ap.add_argument("--no-augment-flip", action="store_true", help="Disable horizontal flip augmentation for weighted policy targets.")
    ap.add_argument("--no-augment-team-swap", action="store_true", help="Disable team slot-block swap augmentation for weighted policy targets.")
    ap.add_argument("--human_eval_split", default="test", choices=["train", "val", "test"], help="Split to draw human-comparison roots from. Off if max_human_roots<=0.")
    ap.add_argument("--max_human_roots", type=int, default=64, help="Max human-action roots per eval (set 0 to disable the human_eval/* metrics).")
    ap.add_argument("--human_inverse_glob", default=str(FIXED_ROOT / "inverse_current" / "stones_with_estimates.chunk*.csv"))
    ap.add_argument("--human_max_inverse_loss", type=float, default=0.08)
    ap.add_argument("--human_noise_config", default=str(FIXED_ROOT / "noise_versions" / "v1_bowling.json"))
    ap.add_argument("--human_noise_samples", type=int, default=0, help="If > 0, log a second human_eval_noise/* block where policy_q averages Q over this many noisy executions per candidate. 0 = deterministic only.")
    ap.add_argument("--human_noise_seed", type=int, default=4242)
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

    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        train_model = torch.nn.DataParallel(model)
    else:
        train_model = model
    opt = torch.optim.AdamW(train_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    wandb_run = None
    eval_ctx = None
    if args.wandb_project:
        try:
            import wandb  # type: ignore
            wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args), reinit=True)
        except Exception as e:
            log(f"wandb disabled: {e}", log_path)
            wandb_run = None
    reference_specs = list(args.reference_policy or [])
    if args.original_policy:
        reference_specs.insert(0, f"original={args.original_policy}")
    references = []
    seen_reference_names = set()
    for spec in reference_specs:
        name, path = _parse_reference_policy(spec)
        base_name = name
        suffix = 2
        while name in seen_reference_names:
            name = f"{base_name}_{suffix}"
            suffix += 1
        seen_reference_names.add(name)
        references.append((name, path))

    if references and args.eval_value and args.holdout is not None and args.horizon is not None:
        ds_eval = ValueDataset(str(FIXED_ROOT / "2026" / "Stones.csv"), str(FIXED_ROOT / "2026" / "Ends.csv"), augment_positions=False, augment_flip=False)
        stage_roots = _stage_roots(ds_eval, int(args.holdout), "val", int(args.horizon), bool(args.include_preplaced_eval), int(args.wandb_max_stage_roots))
        preplaced_roots = _preplaced_roots(ds_eval, int(args.holdout), int(args.wandb_max_preplaced_per_block))
        ref_models = []
        for ref_name, ref_path in references:
            ref_policy, ref_mean, ref_std = load_policy(ref_path, device)
            ref_models.append(
                {
                    "name": ref_name,
                    "path": ref_path,
                    "policy": ref_policy,
                    "mean": ref_mean,
                    "std": ref_std,
                }
            )
        log(f"reference_policies={[r['name'] for r in ref_models]}", log_path)
        value_model = load_value_model(args.eval_value, device)
        human_roots = []
        if int(args.max_human_roots) > 0:
            try:
                inv = load_inverse_estimates(args.human_inverse_glob, args.human_max_inverse_loss)
                human_roots = _human_roots(ds_eval, inv, int(args.holdout), str(args.human_eval_split), int(args.horizon), int(args.max_human_roots))
                log(f"human_eval roots={len(human_roots)} split={args.human_eval_split} horizon={args.horizon}", log_path)
            except Exception as e:
                log(f"human_eval disabled (could not load inverse estimates): {e}", log_path)
                human_roots = []
        human_noise = None
        if int(args.human_noise_samples) > 0 and human_roots:
            try:
                from generate_horizon_curriculum_targets_pipeline import LocalNoise as _LN
                human_noise = _LN(args.human_noise_config, int(args.human_noise_seed))
                log(f"human_eval_noise enabled: samples={args.human_noise_samples} noise_config={args.human_noise_config}", log_path)
            except Exception as e:
                log(f"human_eval_noise disabled: {e}", log_path)
                human_noise = None
        eval_ctx = {
            "stage_roots": stage_roots,
            "preplaced_roots": preplaced_roots,
            "human_roots": human_roots,
            "human_noise": human_noise,
            "human_noise_samples": int(args.human_noise_samples),
            "references": ref_models,
            "value_model": value_model,
        }
    best = float("inf")
    best_state = None
    best_epoch = 0
    no_imp = 0
    best_metric = None
    best_metric_epoch = 0
    metric_no_imp = 0
    last_state = None
    log(
        f"rows={len(x)} train={len(train_idx)} val={len(val_idx)} device={device} "
        f"augment_flip={not args.no_augment_flip} augment_team_swap={not args.no_augment_team_swap} "
        f"data_parallel={bool(args.data_parallel and device.type == 'cuda' and torch.cuda.device_count() > 1)} "
        f"stop_on_metric={args.stop_on_metric} metric_patience_evals={args.metric_patience_evals}",
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
            pi_logits, mu, log_std = train_model(xb, cb)
            nll = _mdn_nll_from_outputs(pi_logits, mu, log_std, zb)
            loss = (nll * wb).sum() / wb.sum().clamp(min=1e-9)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float((nll.detach() * wb).sum().item())
            wsum += float(wb.sum().item())
        last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": int(ep),
                    "train/weighted_nll": float(total / max(wsum, 1e-9)),
                    "val/weighted_nll": float(val),
                    "best/weighted_nll": float(best),
                    "best/epoch": int(best_epoch),
                },
                step=int(ep),
            )
        current_metric = None
        if eval_ctx is not None and (ep == 1 or ep % max(int(args.wandb_eval_interval), 1) == 0 or no_imp == 0):
            new_q = _best_q_for_policy_obj(
                model, action_mean, action_std, eval_ctx["value_model"],
                eval_ctx["stage_roots"], int(args.horizon), device, int(args.eval_policy_candidates), int(args.eval_batch_size),
                float(args.eval_temperature), float(args.eval_std_scale),
            )
            blocks = np.asarray([int(r["thrower_block"]) for r in eval_ctx["stage_roots"]], dtype=np.int32)
            payload = {"epoch": int(ep)}
            first_stage_metrics = {}
            if len(eval_ctx["preplaced_roots"]):
                new_v = _rollout_policy_obj(
                    model, action_mean, action_std, eval_ctx["value_model"],
                    eval_ctx["preplaced_roots"], device, int(args.eval_policy_candidates), int(args.eval_batch_size),
                    float(args.eval_temperature), float(args.eval_std_scale),
                )
                pblocks = np.asarray([int(round(float(r["cond"][2]))) for r in eval_ctx["preplaced_roots"]], dtype=np.int32)
            else:
                new_v = None
                pblocks = None
            for ref_i, ref in enumerate(eval_ctx["references"]):
                prefix = f"vs_{ref['name']}"
                old_q = _best_q_for_policy_obj(
                    ref["policy"], ref["mean"], ref["std"], eval_ctx["value_model"],
                    eval_ctx["stage_roots"], int(args.horizon), device, int(args.eval_policy_candidates), int(args.eval_batch_size),
                    float(args.eval_temperature), float(args.eval_std_scale),
                )
                stage_metrics = _bucketed_metrics(old_q, new_q, blocks, f"{prefix}/stage_win")
                payload.update(stage_metrics)
                if ref_i == 0:
                    first_stage_metrics = stage_metrics
                if new_v is not None and pblocks is not None:
                    old_v = _rollout_policy_obj(
                        ref["policy"], ref["mean"], ref["std"], eval_ctx["value_model"],
                        eval_ctx["preplaced_roots"], device, int(args.eval_policy_candidates), int(args.eval_batch_size),
                        float(args.eval_temperature), float(args.eval_std_scale),
                    )
                    payload.update(_bucketed_metrics(old_v, new_v, pblocks, f"{prefix}/preplaced_rollout"))
            if eval_ctx.get("human_roots"):
                h_deltas, h_a_l2, h_blocks, h_pq, h_hq = _human_eval_obj(
                    model, action_mean, action_std, eval_ctx["value_model"],
                    eval_ctx["human_roots"], int(args.horizon), device, int(args.eval_policy_candidates), int(args.eval_batch_size),
                    float(args.eval_temperature), float(args.eval_std_scale),
                )
                payload.update(_human_bucketed_metrics(h_deltas, h_a_l2, h_blocks, h_pq, h_hq, prefix="human_eval"))
                if eval_ctx.get("human_noise") is not None and eval_ctx.get("human_noise_samples", 0) > 0:
                    n_deltas, n_a_l2, n_blocks, n_pq, n_hq = _human_eval_obj(
                        model, action_mean, action_std, eval_ctx["value_model"],
                        eval_ctx["human_roots"], int(args.horizon), device, int(args.eval_policy_candidates), int(args.eval_batch_size),
                        float(args.eval_temperature), float(args.eval_std_scale),
                        noise=eval_ctx["human_noise"], noise_samples=int(eval_ctx["human_noise_samples"]),
                    )
                    payload.update(_human_bucketed_metrics(n_deltas, n_a_l2, n_blocks, n_pq, n_hq, prefix="human_eval_noise"))
            if args.stop_on_metric == "stage_win_rate":
                current_metric = float(first_stage_metrics.get(f"vs_{eval_ctx['references'][0]['name']}/stage_win/overall/new_win_rate", 0.0))
            elif args.stop_on_metric == "stage_delta_mean":
                current_metric = float(first_stage_metrics.get(f"vs_{eval_ctx['references'][0]['name']}/stage_win/overall/delta_mean", 0.0))
            elif args.stop_on_metric == "preplaced_win_rate":
                current_metric = float(payload.get(f"vs_{eval_ctx['references'][0]['name']}/preplaced_rollout/overall/new_win_rate", float("-inf")))
            elif args.stop_on_metric == "preplaced_delta_mean":
                current_metric = float(payload.get(f"vs_{eval_ctx['references'][0]['name']}/preplaced_rollout/overall/delta_mean", float("-inf")))
            if current_metric is not None and current_metric != float("-inf"):
                if best_metric is None or current_metric > best_metric + 1e-9:
                    best_metric = current_metric
                    best_metric_epoch = int(ep)
                    metric_no_imp = 0
                else:
                    metric_no_imp += 1
                payload["best/stop_metric"] = float(best_metric)
                payload["best/stop_metric_epoch"] = int(best_metric_epoch)
                payload["stop_metric/current"] = float(current_metric)
                payload["stop_metric/no_improve_evals"] = int(metric_no_imp)
            if wandb_run is not None:
                wandb_run.log(payload, step=int(ep))
        if args.stop_on_metric == "val_nll":
            if args.patience > 0 and no_imp >= args.patience:
                break
        elif args.stop_on_metric != "none":
            if args.metric_patience_evals > 0 and metric_no_imp >= args.metric_patience_evals:
                log(
                    f"early_stop_metric epoch={ep:03d} metric={args.stop_on_metric} current={current_metric} "
                    f"best={best_metric} best_epoch={best_metric_epoch} no_improve_evals={metric_no_imp}",
                    log_path,
                )
                break
        elif args.patience > 0 and no_imp >= args.patience:
            break
    best_out = dict(ckpt)
    best_out["model_state_dict"] = best_state
    best_out["distilled_from"] = str(args.init_policy)
    best_out["search_targets"] = str(args.targets)
    best_out["best_val_weighted_nll"] = float(best)
    best_out["best_epoch"] = int(best_epoch)
    best_out["last_epoch"] = int(ep)
    last_out = dict(best_out)
    last_out["model_state_dict"] = last_state if last_state is not None else best_state
    torch.save(best_out, out_dir / "best.pt")
    torch.save(last_out, out_dir / "last.pt")
    torch.save(best_out, out_dir / "model.pt")
    (out_dir / "summary.json").write_text(json.dumps({k: v for k, v in best_out.items() if k != "model_state_dict"}, indent=2, sort_keys=True) + "\n")
    log(f"saved {out_dir / 'best.pt'} and {out_dir / 'last.pt'}", log_path)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
