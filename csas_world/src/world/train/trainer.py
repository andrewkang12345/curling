"""Joint multi-head trainer (4-GPU DDP).

Collection (JAX sim) and training (torch) are separate phases, so this module
never imports JAX: it consumes pre-collected ``.npz`` replay shards plus the
csas human/value datasets, and trains every head jointly from mixed replay.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from ..config import Config
from ..data.human import HumanPolicyDataset, ValueStateDataset
from ..heads.policy_head import fullcov_mdn_nll
from ..losses import WorldLossModule, compute_losses
from ..model import WorldModel
from ..replay import schema
from ..replay.augment import augment_batch
from ..replay.buffers import MixedReplay, load_shards, make_loader
from ..utils.distributed import all_reduce_mean, ddp_cleanup, ddp_setup, is_main
from ..utils.logging import JsonlLogger, fmt_metrics
from ..utils.seed import set_seed


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #
def save_world_checkpoint(path: str, model: WorldModel, cfg: Config, epoch: int,
                          metrics: Dict[str, float], optimizer=None,
                          global_step: int = 0) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "arch": "world_model",
        "model_state_dict": model.state_dict(),
        "model_cfg": asdict(cfg.model),
        "action_mean": model.action_mean.detach().cpu(),
        "action_std": model.action_std.detach().cpu(),
        "epoch": epoch,
        "metrics": metrics,
        "global_step": int(global_step),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
    }, path)


def load_world_checkpoint(model: WorldModel, path: str, map_location="cpu") -> dict:
    ck = torch.load(path, map_location=map_location, weights_only=False)
    sd = dict(ck["model_state_dict"])
    # Backward compatibility: anchor_noisy used a combined reward head whose
    # categorical branch is the current standalone outcome head.
    old_prefix = "reward_head.outcome_head."
    new_prefix = "outcome_head.outcome_head."
    if model.outcome_head is not None:
        for key, value in list(sd.items()):
            if key.startswith(old_prefix):
                sd[new_prefix + key[len(old_prefix):]] = value
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return {"epoch": ck.get("epoch"), "metrics": ck.get("metrics"),
            "global_step": int(ck.get("global_step", 0)),
            "optimizer_state_dict": ck.get("optimizer_state_dict"),
            "missing": len(missing), "unexpected": len(unexpected),
            "missing_keys": list(missing), "unexpected_keys": list(unexpected)}


def export_csas_policy(model: WorldModel, path: str, cfg: Config) -> None:
    """Export the WorldModel's policy head as a csas-loadable full-cov MDN checkpoint.

    ``WorldModel.trunk.prior`` *is* a ``PolicyGraphTransformerFullCovMDN``, so this
    yields a checkpoint that ``csas.search.load_policy`` can consume -- used both
    to drive MCTS collection and to head-to-head against the mcts_horizon baselines.
    """
    from .._bootstrap import GNN_FEATURE_ENV

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "arch": "policy_graph_transformer_fullcov_mdn",
        "model_state_dict": {k: v.detach().cpu() for k, v in model.trunk.prior.state_dict().items()},
        "input_dim": cfg.model.input_dim, "cond_dim": cfg.model.cond_dim,
        "action_dim": cfg.model.action_dim,
        "args": {"hidden_dim": cfg.model.hidden_dim, "n_layers": cfg.model.n_layers,
                 "n_heads": cfg.model.n_heads, "n_mixtures": cfg.model.n_mixtures,
                 "covariance": "full"},
        "action_mean": model.action_mean.detach().cpu().tolist(),
        "action_std": model.action_std.detach().cpu().tolist(),
        "graph_feature_env": dict(GNN_FEATURE_ENV),
    }, path)


def build_model(cfg: Config, device, init_ckpt: Optional[str]):
    model = WorldModel(cfg.model).to(device)
    resume = {}
    if init_ckpt and Path(init_ckpt).exists():
        resume = load_world_checkpoint(model, init_ckpt, map_location=device)
        printable = {k: v for k, v in resume.items() if k != "optimizer_state_dict"}
        print(f"[init] resumed from {init_ckpt}: {printable}", flush=True)
    else:
        try:
            rep = model.warm_start(cfg.csas_path(cfg.paths.prior_policy_ckpt).as_posix(),
                                   cfg.csas_path(cfg.paths.prior_value_ckpt).as_posix())
            print(f"[init] warm-started from prior: "
                  f"{ {k: (v.get('loaded') if isinstance(v, dict) else v) for k, v in rep.items()} }",
                  flush=True)
        except RuntimeError as e:
            # the canonical prior is 256-dim; a scaled model (az_v14) can't absorb it —
            # train truly from scratch (the human-BC replay slice bootstraps the policy).
            print(f"[init] prior warm-start skipped (dim mismatch — training FROM SCRATCH): "
                  f"{str(e).splitlines()[0]}", flush=True)
    if model.target_trunk is not None and not (init_ckpt and Path(init_ckpt).exists()):
        model.target_trunk.load_state_dict(model.trunk.state_dict())
    return model, resume


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #
def build_sources(cfg: Config, mcts_shard_dir: Optional[str], sim_shard_dir: Optional[str],
                  split: str = "train") -> Dict[str, tuple]:
    K, M = cfg.replay.unroll_steps, cfg.search.soft_topk
    root = cfg.paths.csas_v3_root
    sources: Dict[str, tuple] = {}
    if cfg.loss.policy_bc > 0:
        sources["human"] = (HumanPolicyDataset(root, K, M, holdout=0, split=split), cfg.replay.mix_human)
    if cfg.loss.value > 0:
        from torch.utils.data import ConcatDataset

        val_parts = [ValueStateDataset(
            cfg.csas_path(cfg.paths.value_data_stones).as_posix(),
            cfg.csas_path(cfg.paths.value_data_ends).as_posix(),
            K, M, holdout=0, split=split)]
        if split == "train" and cfg.replay.value_use_synthetic:
            val_parts.append(ValueStateDataset(
                cfg.csas_path(cfg.paths.value_synth_stones).as_posix(),
                cfg.csas_path(cfg.paths.value_synth_ends).as_posix(),
                K, M, split="all", max_rows=65_536))
        value_ds = val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts)
        sources["value"] = (value_ds, cfg.replay.mix_value)
    if sim_shard_dir:
        sim = load_shards(sim_shard_dir, K, M)
        if sim is not None:
            sources["sim"] = (sim, cfg.replay.mix_sim)
    if mcts_shard_dir:
        mcts = load_shards(mcts_shard_dir, K, M)
        if mcts is not None:
            sources["mcts"] = (mcts, cfg.replay.mix_mcts)
    return sources


# --------------------------------------------------------------------------- #
# evaluation (baseline-comparable val NLL/MSE)
# --------------------------------------------------------------------------- #
@torch.no_grad()
@torch.no_grad()
def evaluate_mcts_losses(model: WorldModel, cfg: Config, device,
                         mcts_val_ds, max_batches: int = 60) -> Dict[str, float]:
    """Per-loss val metrics on a held-out MCTS partition. Iterates the val dataset, calls
    ``compute_losses`` per batch with the LIVE cfg.loss (so every active head contributes), and
    aggregates each metric. Returns a dict with keys ``val_<name>_mcts`` for every metric the
    loss reports (policy_distill, value_mse, value_nll, consistency, step_reward, etc.). Used to
    detect overfitting of the value/policy/dynamics heads on their actual training distribution,
    not just on the external human-data benchmarks."""
    from .. import losses as _losses_mod
    out: Dict[str, float] = {}
    if mcts_val_ds is None or len(mcts_val_ds) == 0:
        return out
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    bs = 128
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    n_batches = 0
    for start in range(0, len(mcts_val_ds), bs):
        end = min(len(mcts_val_ds), start + bs)
        rows = [mcts_val_ds[i] for i in range(start, end)]
        batch: Dict[str, torch.Tensor] = {}
        for k in rows[0].keys():
            arr = np.stack([r[k] for r in rows], axis=0)
            batch[k] = torch.from_numpy(arr).to(device)
        try:
            _total, m = _losses_mod.compute_losses(model, batch, cfg.loss)
        except Exception:                                # pragma: no cover (best-effort eval)
            continue
        for kk, vv in m.items():
            sums[kk] = sums.get(kk, 0.0) + float(vv)
            counts[kk] = counts.get(kk, 0) + 1
        n_batches += 1
        if n_batches >= int(max_batches):
            break
    for kk, s in sums.items():
        out[f"val_{kk}_mcts"] = s / max(counts[kk], 1)
    model.train()
    return out


def evaluate(model: WorldModel, cfg: Config, device, max_batches: int = 60,
             mcts_val_ds=None) -> Dict[str, float]:
    model.eval()
    # Free training's cached-but-unallocated blocks before the val eval allocates — rank-0
    # GPU 0 carries ~18 GB of DDP training, and over a long multi-stage curriculum the val
    # eval's extra ~1.5 GB stopped fitting (EXP-017 OOM'd at h08). empty_cache + per-batch
    # device moves + a smaller eval batch keep GPU 0 under the ceiling. Numerics unchanged.
    if device.type == "cuda":
        torch.cuda.empty_cache()
    bs = 256
    out: Dict[str, float] = {}
    # policy val NLL on human val split (val tensors stay on CPU; move per-batch)
    try:
        hp = HumanPolicyDataset(cfg.paths.csas_v3_root, cfg.replay.unroll_steps,
                                cfg.search.soft_topk, holdout=0, split="val")
        x = torch.from_numpy(hp.x); c = torch.from_numpy(hp.c); a = torch.from_numpy(hp.a)
        nlls = []
        for i in range(0, len(x), bs):
            h = model.encode(x[i:i+bs].to(device), c[i:i+bs].to(device))
            pi, mu, tril = model.policy(h)
            az = model.raw_to_z(a[i:i+bs].to(device))
            nlls.append(fullcov_mdn_nll(pi, mu, tril, az, reduce=False).cpu())
            if i // bs >= max_batches:
                break
        out["val_policy_nll"] = float(torch.cat(nlls).mean())
    except Exception as e:  # noqa: BLE001
        out["val_policy_nll"] = float("nan")
        print(f"[eval] policy eval skipped: {e}", flush=True)
    # value val MSE/NLL on value val split
    if model.value_head is not None:
        try:
            vs = ValueStateDataset(cfg.csas_path(cfg.paths.value_data_stones).as_posix(),
                                   cfg.csas_path(cfg.paths.value_data_ends).as_posix(),
                                   cfg.replay.unroll_steps, cfg.search.soft_topk,
                                   holdout=0, split="val", max_rows=20_000)
            x = torch.from_numpy(vs.x); c = torch.from_numpy(vs.c); y = torch.from_numpy(vs.v)
            se, nll, n = 0.0, 0.0, 0
            for i in range(0, len(x), bs):
                h = model.encode(x[i:i+bs].to(device), c[i:i+bs].to(device))
                mean, logvar = model.value(h)
                tgt = y[i:i+bs].to(device)
                se += float(((mean - tgt) ** 2).sum())
                nll += float((0.5 * (torch.exp(-logvar) * (tgt - mean) ** 2 + logvar)).sum())
                n += len(tgt)
            out["val_value_mse"] = se / max(n, 1)
            out["val_value_nll"] = nll / max(n, 1)
        except Exception as e:  # noqa: BLE001
            print(f"[eval] value eval skipped: {e}", flush=True)
    # per-loss val on held-out MCTS partition (real training distribution; catches overfitting
    # of value/policy_distill/consistency/step_reward heads that the external benchmarks miss).
    if mcts_val_ds is not None:
        try:
            out.update(evaluate_mcts_losses(model, cfg, device, mcts_val_ds))
        except Exception as e:  # noqa: BLE001
            print(f"[eval] mcts-val eval skipped: {e}", flush=True)
    model.train()
    return out


# --------------------------------------------------------------------------- #
# training (one DDP rank)
# --------------------------------------------------------------------------- #
def train_world(cfg: Config, rank: int, world_size: int, *, mcts_shard_dir: Optional[str] = None,
                sim_shard_dir: Optional[str] = None, mcts_val_shard_dir: Optional[str] = None,
                init_ckpt: Optional[str] = None,
                out_dir: str = "checkpoints/csas_world", epochs: Optional[int] = None,
                results_path: Optional[str] = None) -> None:
    device = ddp_setup(rank, world_size, cfg.train.gpus, cfg.train.ddp_backend)
    set_seed(cfg.train.seed + rank)
    epochs = epochs or cfg.train.epochs
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    model, resume = build_model(cfg, device, init_ckpt)
    loss_mod = WorldLossModule(model, cfg.loss).to(device)
    # az_v15-VH freeze must happen BEFORE the DDP wrap (the reducer registers hooks from
    # requires_grad at construction; freezing afterwards makes DDP expect grads that never come)
    if getattr(cfg.train, "train_value_head_only", False):
        n_frozen = n_train = 0
        for name, p in loss_mod.named_parameters():
            trainable = "value_head" in name
            p.requires_grad = trainable
            n_train += int(trainable); n_frozen += int(not trainable)
        if is_main(rank):
            print(f"[freeze] value-head-only: {n_train} trainable tensors, {n_frozen} frozen", flush=True)
    if world_size > 1:
        ddp = DDP(loss_mod, device_ids=[device.index] if device.type == "cuda" else None,
                  output_device=device.index if device.type == "cuda" else None,
                  broadcast_buffers=True, find_unused_parameters=False)
    else:
        ddp = loss_mod

    sources = build_sources(cfg, mcts_shard_dir, sim_shard_dir, split="train")
    mixed = MixedReplay(sources, virtual_len=cfg.train.samples_per_epoch, seed=cfg.train.seed)
    loader, sampler = make_loader(mixed, cfg.train.batch_size, world_size > 1, rank, world_size,
                                  cfg.train.num_workers, shuffle=True, seed=cfg.train.seed)
    if is_main(rank):
        print(f"[data] sources={ {k: len(v[0]) for k, v in sources.items()} } "
              f"mix={mixed.composition()}", flush=True)
    # Optional held-out MCTS val partition for per-loss generalization metrics. Built once,
    # iterated rank-0-only during evaluate(); not used for training.
    mcts_val_ds = None
    if mcts_val_shard_dir and is_main(rank):
        K_, M_ = cfg.replay.unroll_steps, cfg.search.soft_topk
        mcts_val_ds = load_shards(mcts_val_shard_dir, K_, M_)
        if mcts_val_ds is not None:
            print(f"[data] mcts-val held-out: {len(mcts_val_ds)} records from {mcts_val_shard_dir}",
                  flush=True)

    # A resumed checkpoint is already trained end-to-end. Do not misclassify its
    # heads as fresh: all loaded parameters continue at lr_pretrained. Fresh-start
    # runs retain the original discriminative grouping.
    pretrained, fresh = [], []
    for name, p in loss_mod.named_parameters():
        if not p.requires_grad:
            continue
        n = name.replace("model.", "")
        if init_ckpt and Path(init_ckpt).exists():
            pretrained.append(p)
        elif n.startswith("trunk.prior"):
            pretrained.append(p)
        else:
            fresh.append(p)
    params = pretrained + fresh
    groups = []
    if pretrained:
        groups.append({"params": pretrained, "lr": cfg.train.lr_pretrained})
    if fresh:
        groups.append({"params": fresh, "lr": cfg.train.lr})
    opt = torch.optim.AdamW(groups, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    optimizer_state = resume.get("optimizer_state_dict")
    if optimizer_state is not None:
        # Resume intentionally regroups every param into one lr_pretrained group, so a
        # checkpoint saved with the 2-group (pretrained/fresh) split won't match. Skip
        # the moment restore on any mismatch rather than crash -- AdamW re-warms quickly
        # and the model weights are already loaded.
        try:
            opt.load_state_dict(optimizer_state)
            print(f"[opt] restored AdamW state from {init_ckpt}", flush=True)
        except (ValueError, KeyError) as e:
            print(f"[opt] WARNING: optimizer state not restored ({e}); fresh AdamW moments", flush=True)
    if is_main(rank):
        print(f"[opt] pretrained params={sum(p.numel() for p in pretrained)/1e6:.2f}M @lr={cfg.train.lr_pretrained} | "
              f"fresh params={sum(p.numel() for p in fresh)/1e6:.2f}M @lr={cfg.train.lr}", flush=True)
    amp = cfg.train.amp and device.type == "cuda"

    logger = JsonlLogger(os.path.join(cfg.train.ckpt_dir, "..", "artifacts", "logs",
                                      f"{cfg.train.run_name}.jsonl"),
                         enabled=is_main(rank))
    best_key = float("inf")
    best_sel_epoch = -1
    last_metrics: Dict[str, float] = {}
    step = int(resume.get("global_step", 0))
    start_epoch = int(resume.get("epoch", -1)) + 1
    for epoch in range(epochs):
        mixed.set_epoch(epoch)
        if sampler is not None:
            sampler.set_epoch(epoch)
        loss_mod.train()
        running: Dict[str, float] = {}
        nb = 0
        for batch in loader:
            batch = schema.to_device(batch, device)
            if cfg.train.augment:
                batch = augment_batch(batch)
            opt.zero_grad(set_to_none=True)
            if amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    total, metrics = ddp(batch)
            else:
                total, metrics = ddp(batch)
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
            opt.step()
            model.update_ema()
            step += 1
            nb += 1
            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + v
            if is_main(rank) and step % cfg.train.log_every == 0:
                cur = {k: v / nb for k, v in running.items()}
                print(f"[e{epoch} s{step}] {fmt_metrics(cur)}", flush=True)
        # epoch end
        if world_size > 1:
            torch.distributed.barrier()
        if is_main(rank):
            train_metrics = {k: v / max(nb, 1) for k, v in running.items()}
            val_metrics = evaluate(model, cfg, device, mcts_val_ds=mcts_val_ds)
            last_metrics = {**{f"train_{k}": v for k, v in train_metrics.items()}, **val_metrics}
            absolute_epoch = start_epoch + epoch
            logger.log({"epoch": absolute_epoch, **last_metrics})
            print(f"[epoch {absolute_epoch}] {fmt_metrics(val_metrics)}", flush=True)
            save_world_checkpoint(os.path.join(out_dir, "last.pt"), model, cfg, absolute_epoch,
                                  last_metrics, optimizer=opt, global_step=step)
            if cfg.train.checkpoint_metric != "none":
                key = val_metrics.get(cfg.train.checkpoint_metric, float("inf"))
                guard = float(getattr(cfg.train, "select_value_guard", 0.0) or 0.0)
                guard_metric = str(getattr(cfg.train, "select_value_guard_metric", "val_value_mse"))
                guard_ok = guard <= 0.0 or val_metrics.get(guard_metric, 0.0) <= guard
                if not guard_ok:
                    print(f"[select] epoch {absolute_epoch} ineligible: {guard_metric}="
                          f"{val_metrics.get(guard_metric, float('nan')):.4f} > guard {guard:.4f}", flush=True)
                if np.isfinite(key) and key < best_key and guard_ok:
                    best_key = key
                    best_sel_epoch = epoch
                    save_world_checkpoint(os.path.join(out_dir, "best.pt"), model, cfg, absolute_epoch,
                                          last_metrics, optimizer=opt, global_step=step)

        # val-driven early stopping: abort (all ranks) when the selection metric hasn't
        # improved for `early_stop_patience` epochs. Rank 0 decides; broadcast the flag.
        patience = int(getattr(cfg.train, "early_stop_patience", 0) or 0)
        if patience > 0 and cfg.train.checkpoint_metric != "none":
            stop = 0
            if is_main(rank):
                anchor = best_sel_epoch if best_sel_epoch >= 0 else 0
                stop = 1 if (epoch - anchor) >= patience else 0
            if world_size > 1:
                t = torch.tensor([stop], dtype=torch.int64)
                torch.distributed.broadcast(t, src=0)
                stop = int(t.item())
            if stop:
                if is_main(rank):
                    print(f"[early-stop] no eligible improvement for {patience} epochs "
                          f"(best at epoch {start_epoch + best_sel_epoch if best_sel_epoch >= 0 else 'none'}) "
                          f"-> abort after epoch {start_epoch + epoch}", flush=True)
                break

    if is_main(rank):
        save_world_checkpoint(os.path.join(out_dir, "model.pt"), model, cfg,
                              start_epoch + epochs - 1, last_metrics,
                              optimizer=opt, global_step=step)
        if results_path:
            with open(results_path, "w") as fh:
                json.dump({"out_dir": out_dir, "metrics": last_metrics, "best_val": best_key}, fh, indent=2)
    ddp_cleanup(world_size)


def _worker(rank: int, world_size: int, cfg: Config, kwargs: dict) -> None:
    train_world(cfg, rank, world_size, **kwargs)


def launch(cfg: Config, **kwargs) -> dict:
    """Spawn DDP training across cfg.train.gpus; returns rank-0 results dict."""
    world_size = max(1, len(cfg.train.gpus))
    results_path = kwargs.get("results_path") or os.path.join(
        kwargs.get("out_dir", "checkpoints/csas_world"), "results.json")
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    kwargs["results_path"] = results_path
    if world_size == 1:
        train_world(cfg, 0, 1, **kwargs)
    else:
        mp.spawn(_worker, args=(world_size, cfg, kwargs), nprocs=world_size, join=True)
    if Path(results_path).exists():
        with open(results_path) as fh:
            return json.load(fh)
    return {"out_dir": kwargs.get("out_dir"), "metrics": {}}


__all__ = ["train_world", "launch", "build_model", "build_sources", "evaluate",
           "save_world_checkpoint", "load_world_checkpoint"]
