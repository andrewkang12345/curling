"""Render policy multi-action-sample figures for csas_world, with the thrower's
HAMMER status added to each title (next to "thrower team").

Reuses the csas plotting pipeline verbatim by monkeypatching ``heatmap_real_cases``
so each case's title gains a "(hammer)" / "(no hammer)" tag derived from the
thrower's team_order (csas dropped is_hammer because it is identical to
team_order; team_order==1 == has hammer). Nothing in csas_v3 is modified on disk.

    cd /mnt/data/curling2/csas_world && source scripts/setup_gpu.sh
    python -m world.eval.policy_figs --policy <csas_policy.pt> --label csas_world_anchor \
        --out-root artifacts/figures/policy_world --device cuda:0
"""
from __future__ import annotations

import argparse
import sys

import world  # noqa: F401  (bootstrap: csas on path + GNN env)


def hammer_word(cond) -> str:
    return "hammer" if int(round(float(cond[1]))) == 1 else "no hammer"


def _install_hammer_titles() -> None:
    import csas.visualize_policy_multi_action_samples as M
    from csas.visualize_policy_prior_samples import _team_name_from_block

    _orig = M.heatmap_real_cases

    def wrapped(n_real, seed, holdout, split, horizon):
        cases = _orig(n_real, seed, holdout, split, horizon)
        for c in cases:
            cond = c.get("post_cond", c.get("cond"))
            title = c.get("title")
            if cond is None or not title:
                continue
            team = _team_name_from_block(cond)
            tag = f"thrower team: {team}"
            if tag in title:
                c["title"] = title.replace(tag, f"{tag} ({hammer_word(cond)})", 1)
            else:
                c["title"] = f"{title} | thrower hammer: {hammer_word(cond)}"
        return cases

    M.heatmap_real_cases = wrapped
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="csas-format policy checkpoint")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--holdout", type=int, default=0)
    ap.add_argument("--split", default="test")
    ap.add_argument("--start-horizon", type=int, default=1)
    ap.add_argument("--max-horizon", type=int, default=10)
    ap.add_argument("--n-real", type=int, default=4)
    ap.add_argument("--n-samples", type=int, default=96)
    ap.add_argument("--n-trajectories", type=int, default=12)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    M = _install_hammer_titles()
    sys.argv = [
        "visualize_policy_multi_action_samples",
        "--policy-template", args.policy, "--policy-label", args.label,
        "--out-root", args.out_root, "--holdout", str(args.holdout), "--split", args.split,
        "--start-horizon", str(args.start_horizon), "--max-horizon", str(args.max_horizon),
        "--n-real", str(args.n_real), "--n-samples", str(args.n_samples),
        "--n-trajectories", str(args.n_trajectories), "--device", args.device,
    ]
    M.main()


if __name__ == "__main__":
    main()
