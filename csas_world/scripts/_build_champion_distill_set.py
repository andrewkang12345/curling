#!/usr/bin/env python3
"""Build the champion→big-model DISTILLATION set (az_v14b phase 1).

For every state in the az_v14 corpus: 24 actions sampled from the CHAMPION's policy
(temperature 1.0 — the true distribution) as a uniform-weight distillation target, and
the champion's value estimate as the step-0 value target. k_eff=0 (no unroll), so
consistency/reward come from the usual sim/replay slices during training.

Teacher: checkpoints/csas_world/az_v9_selfplay/iter2/best.pt (7.3M champion).
Output: artifacts/replay/az_v14b_distill_{train,val} (~90/10 split by shard).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import world  # noqa: E402,F401
from world.replay.episode import build_unroll_record  # noqa: E402
from world.replay.schema import SOURCE_MCTS  # noqa: E402
from world.search.collect import _live_np  # noqa: E402

CHAMPION = ROOT / "checkpoints/csas_world/az_v9_selfplay/iter2/best.pt"
K, M = 5, 24
N_SAMPLES = 24


def main():
    import torch
    from world.config import ModelCfg, model_cfg_from_dict
    from world.heads.policy_head import sample_actions_z
    from world.model import WorldModel
    from world.train.trainer import load_world_checkpoint

    dev = torch.device("cuda:0")
    ck = torch.load(CHAMPION, map_location=dev, weights_only=False)
    teacher = WorldModel(model_cfg_from_dict(ck["model_cfg"])).to(dev)
    load_world_checkpoint(teacher, str(CHAMPION), map_location=dev)
    teacher.eval()

    # gather (x0, c0, horizon) from the az_v14 corpus (train side only)
    xs, cs, hs = [], [], []
    for fp in sorted((ROOT / "artifacts/replay/az_v14_train").glob("*.npz")):
        d = np.load(fp, allow_pickle=True)
        xs.append(d["x0"]); cs.append(d["c0"]); hs.append(d["horizon"])
    X = np.concatenate(xs); C = np.concatenate(cs); H = np.concatenate(hs)
    print(f"[distill-set] {len(X)} teacher states")

    out_t = ROOT / "artifacts/replay/az_v14b_distill_train"
    out_v = ROOT / "artifacts/replay/az_v14b_distill_val"
    for d in (out_t, out_v):
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*.npz"):
            f.unlink()

    from world.actions import clip_raw
    from world.replay.buffers import save_shard

    recs, shard_id, per_shard = [], 0, 1600
    w_uniform = np.full(N_SAMPLES, 1.0 / N_SAMPLES, dtype=np.float32)
    bs = 512
    with torch.no_grad():
        for s0 in range(0, len(X), bs):
            xb = torch.as_tensor(X[s0:s0 + bs], dtype=torch.float32, device=dev)
            cb = torch.as_tensor(C[s0:s0 + bs], dtype=torch.float32, device=dev)
            h_enc = teacher.encode(xb, cb)
            pi, mu, tril = teacher.policy(h_enc)
            z = sample_actions_z(pi, mu, tril, n_samples=N_SAMPLES, temperature=1.0, std_scale=1.0)
            a = (z * teacher.action_std + teacher.action_mean).cpu().numpy().astype(np.float32)
            v = teacher.value_head.value(h_enc).cpu().numpy().astype(np.float32).reshape(-1)
            for i in range(len(xb)):
                da = clip_raw(a[i])
                rec = build_unroll_record(
                    K, M, x0=X[s0 + i], c0=C[s0 + i],
                    actions_raw=np.zeros((0, 4), np.float32),
                    next_states=np.zeros((0, 24), np.float32),
                    next_conds=np.zeros((0, 3), np.float32),
                    value_targets=np.array([v[i]], np.float32),
                    rewards=np.zeros((0,), np.float32),
                    outcome_margin=float(v[i]), source=SOURCE_MCTS, horizon=int(H[s0 + i]),
                    dist_actions_raw=da, dist_weights=w_uniform, live_mask_fn=_live_np)
                recs.append(rec)
            while len(recs) >= per_shard:
                dst = out_v if shard_id % 10 == 9 else out_t   # every 10th shard -> val
                save_shard(str(dst / f"shard{shard_id:03d}.npz"), recs[:per_shard])
                recs = recs[per_shard:]
                shard_id += 1
            if (s0 // bs) % 20 == 0:
                print(f"[distill-set] {s0 + len(xb)}/{len(X)}", flush=True)
    if recs:
        dst = out_v if shard_id % 10 == 9 else out_t
        save_shard(str(dst / f"shard{shard_id:03d}.npz"), recs)
    n_t = sum(1 for _ in out_t.glob("*.npz")); n_v = sum(1 for _ in out_v.glob("*.npz"))
    print(f"[distill-set] wrote {n_t} train shards, {n_v} val shards")
    print("DISTILL_SET_DONE")


if __name__ == "__main__":
    main()
