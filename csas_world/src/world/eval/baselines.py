"""Compare csas_world's value/policy heads (and game strength) to prior bests.

Produces, for the held-out split:
  * policy val/test NLL   : csas_world vs human_prior_fullcov vs mcts_horizon
  * value  val/test MSE/NLL: csas_world vs the canonical Gaussian value model
  * head-to-head winrate  : csas_world vs each baseline policy, both orders

Run:
  JAX_PLATFORMS=cpu PYTHONPATH=src python -m world.eval.baselines \
      --world checkpoints/csas_world/anchor/model.pt --horizons 2,5,8,10 --h2h-roots 150
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional

import numpy as np
import torch

from ..config import Config
from ..heads.policy_head import fullcov_mdn_nll


# --------------------------------------------------------------------------- #
# NLL helpers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _csas_policy_nll(policy_ckpt: str, x, c, a, device) -> float:
    from csas.search import load_policy

    model, amean, astd = load_policy(policy_ckpt, device)
    az = (torch.from_numpy(a).to(device) - amean) / astd
    xt = torch.from_numpy(x).to(device); ct = torch.from_numpy(c).to(device)
    nlls = []
    for i in range(0, len(xt), 256):
        nlls.append(model.nll_per_sample(xt[i:i+256], ct[i:i+256], az[i:i+256]).cpu())
    return float(torch.cat(nlls).mean())


@torch.no_grad()
def _world_policy_nll(model, x, c, a, device) -> float:
    xt = torch.from_numpy(x).to(device); ct = torch.from_numpy(c).to(device)
    at = torch.from_numpy(a).to(device)
    nlls = []
    for i in range(0, len(xt), 256):
        h = model.encode(xt[i:i+256], ct[i:i+256])
        pi, mu, tril = model.policy(h)
        nlls.append(fullcov_mdn_nll(pi, mu, tril, model.raw_to_z(at[i:i+256]), reduce=False).cpu())
    return float(torch.cat(nlls).mean())


@torch.no_grad()
def _world_value_metrics(model, x, c, y, device) -> Dict[str, float]:
    xt = torch.from_numpy(x).to(device); ct = torch.from_numpy(c).to(device)
    yt = torch.from_numpy(y).to(device)
    se = nll = n = 0.0
    for i in range(0, len(xt), 256):
        h = model.encode(xt[i:i+256], ct[i:i+256])
        mean, logvar = model.value(h)
        tgt = yt[i:i+256]
        se += float(((mean - tgt) ** 2).sum())
        nll += float((0.5 * (torch.exp(-logvar) * (tgt - mean) ** 2 + logvar)).sum())
        n += len(tgt)
    return {"mse": se / n, "nll": nll / n}


@torch.no_grad()
def _csas_value_metrics(value_ckpt: str, x, c, y, device) -> Dict[str, float]:
    from .. import env_bridge

    vm = env_bridge.load_csas_value(value_ckpt, device)
    xt = torch.from_numpy(x).to(device); ct = torch.from_numpy(c).to(device)
    yt = torch.from_numpy(y).to(device)
    se = nll = n = 0.0
    for i in range(0, len(xt), 256):
        res = vm(xt[i:i+256], ct[i:i+256])
        mean = res[0].squeeze(-1); logvar = res[1].squeeze(-1)
        tgt = yt[i:i+256]
        se += float(((mean - tgt) ** 2).sum())
        nll += float((0.5 * (torch.exp(-logvar) * (tgt - mean) ** 2 + logvar)).sum())
        n += len(tgt)
    return {"mse": se / n, "nll": nll / n}


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def report(world_ckpt: str, cfg: Config, device, horizons: List[int], h2h_roots: int = 150,
           n_candidates: int = 32) -> Dict:
    from ..config import model_cfg_from_dict
    from ..data.human import HumanPolicyDataset, ValueStateDataset
    from ..eval.head_to_head import CsasPlayer, WorldPlayer, build_h2h_roots, head_to_head
    from ..model import WorldModel
    from ..train.trainer import load_world_checkpoint

    dev = torch.device(device)
    ck = torch.load(world_ckpt, map_location=dev, weights_only=False)
    mcfg = model_cfg_from_dict(ck["model_cfg"]) if "model_cfg" in ck else cfg.model
    model = WorldModel(mcfg).to(dev)
    load_world_checkpoint(model, world_ckpt, map_location=dev)
    model.eval()

    out: Dict = {"world_ckpt": world_ckpt, "policy_nll": {}, "value": {}, "head_to_head": {}}

    # ---- policy NLL (val + test) ----
    for split in ("val", "test"):
        hp = HumanPolicyDataset(cfg.paths.csas_v3_root, cfg.replay.unroll_steps,
                                cfg.search.soft_topk, holdout=0, split=split)
        wp = _world_policy_nll(model, hp.x, hp.c, hp.a, dev)
        prior = _csas_policy_nll(cfg.csas_path(cfg.paths.prior_policy_ckpt).as_posix(),
                                 hp.x, hp.c, hp.a, dev)
        out["policy_nll"][split] = {"csas_world": wp, "human_prior_fullcov": prior}

    # ---- value MSE/NLL (val + test) ----
    for split in ("val", "test"):
        vs = ValueStateDataset(cfg.csas_path(cfg.paths.value_data_stones).as_posix(),
                               cfg.csas_path(cfg.paths.value_data_ends).as_posix(),
                               cfg.replay.unroll_steps, cfg.search.soft_topk,
                               holdout=0, split=split)
        wv = _world_value_metrics(model, vs.x, vs.c, vs.v, dev)
        cv = _csas_value_metrics(cfg.csas_path(cfg.paths.baseline_value_ckpt).as_posix(),
                                 vs.x, vs.c, vs.v, dev)
        out["value"][split] = {"csas_world": wv, "gaussian_value": cv}

    # ---- head-to-head vs baseline policies ----
    A = WorldPlayer(world_ckpt, dev, n_candidates=n_candidates, name="csas_world")
    bval = cfg.csas_path(cfg.paths.baseline_value_ckpt).as_posix()
    for pol in cfg.paths.baseline_policy_ckpts:
        polp = cfg.csas_path(pol).as_posix()
        B = CsasPlayer(polp, bval, dev, n_candidates=n_candidates, name=pol)
        per_h = {}
        for h in horizons:
            roots = build_h2h_roots(cfg.paths.csas_v3_root, h, h2h_roots, split="val", seed=h)
            per_h[f"h{h:02d}"] = head_to_head(A, B, roots)
        out["head_to_head"][pol] = per_h

    return out


def _fmt(out: Dict) -> str:
    L = []
    L.append("=== POLICY NLL (lower better; baseline prior val/test = 2.975 / 3.068) ===")
    for sp, d in out["policy_nll"].items():
        L.append(f"  {sp}: csas_world={d['csas_world']:.4f}  prior={d['human_prior_fullcov']:.4f}")
    L.append("=== VALUE (baseline val MSE/NLL ~ 2.13 / 0.73) ===")
    for sp, d in out["value"].items():
        L.append(f"  {sp}: csas_world mse={d['csas_world']['mse']:.4f} nll={d['csas_world']['nll']:.4f} | "
                 f"gaussian mse={d['gaussian_value']['mse']:.4f} nll={d['gaussian_value']['nll']:.4f}")
    L.append("=== HEAD-TO-HEAD winrate (csas_world vs baseline; >0.5 = world better) ===")
    for pol, per_h in out["head_to_head"].items():
        L.append(f"  vs {pol}:")
        for h, wr in per_h.items():
            L.append(f"    {h}: overall={wr['winrate']:.3f} "
                     f"(order0={wr['winrate_order0']:.2f}, order1={wr['winrate_order1']:.2f}) "
                     f"margin={wr['mean_margin']:+.2f}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--horizons", default="2,5,8,10")
    ap.add_argument("--h2h-roots", type=int, default=150)
    ap.add_argument("--n-candidates", type=int, default=32)
    ap.add_argument("--baselines", default=None, help="comma list overriding cfg baseline policies")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from ..config import load_config

    cfg = load_config(args.config) if args.config else Config()
    if args.baselines:
        cfg.paths.baseline_policy_ckpts = [b for b in args.baselines.split(",") if b]
    horizons = [int(h) for h in args.horizons.split(",") if h]
    out = report(args.world, cfg, args.device, horizons, args.h2h_roots, args.n_candidates)
    print(_fmt(out))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"[baselines] wrote {args.out}")


if __name__ == "__main__":
    main()
