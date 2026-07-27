#!/usr/bin/env python3
"""Render value-head and best-decision figures from a trained csas_world WorldModel.

Two figure kinds, mirroring the csas_v3 style:

  * ``value`` -- value-delta heatmaps around the button. For each held-out real
    test state we slide the thrown stone over a grid and plot
    ``V(post) - V(pre)`` using the WorldModel value head. Reuses csas's
    ``make_value_heatmaps._candidate_heatmap`` / ``_predict_value`` unchanged by
    wrapping the WorldModel in ``WorldValueAdapter`` (a tiny ``nn.Module`` whose
    ``forward(x, c)`` returns ``(mean, logvar)``).

  * ``best_decision`` -- best policy-sampled shot under execution noise. We sample
    policy candidate actions from the WorldModel policy head, score each by the
    *mean noisy decision-value surplus* over ``noise_samples`` executions
    (simulate -> apply legality -> terminal ``score_end`` if horizon<=1 else
    ``-V(post, next_cond)``), pick the argmax surplus, and plot the best intended
    shot's noisy trajectories/endpoints plus a histogram of candidate surplus.

This module is the ONLY world.eval figure entry point and must be run under
``scripts/setup_gpu.sh`` so the JAX simulator (csas) and the vendored deps are on
the path. ``import world`` happens first so the GNN feature env is consistent.
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import the world package FIRST so the GNN_* feature env is exported before any
# csas.gnn_models / csas.policy_graph_model import (setup_gpu.sh also exports it).
import world  # noqa: F401
from world.config import ModelCfg, model_cfg_from_dict
from world.model import WorldModel
from world.train.trainer import load_world_checkpoint
from world import env_bridge
from world.actions import clip_raw
from world.heads.policy_head import sample_actions_z
from world.search.noise import make_noise

# csas figure helpers (read-only reuse). These must be imported after `world`.
from csas import make_value_heatmaps as hv  # value-heatmap grid + predict
from csas.common import POS_MAX, NUM_STONES, STONE_RADIUS_M, in_play_raw, resolve_stone_overlaps_raw
from csas.common import ACTION_SPEED_MIN, ACTION_SPEED_MAX, ACTION_SPIN_MIN, ACTION_SPIN_MAX  # noqa: F401
from csas.visualize_policy_prior_samples import (  # plotting + case loading
    BUTTON_RAW,
    M_PER_RAW,
    heatmap_real_cases,
    _draw_house,
    _plot_stones,
    _plot_highlighted_slot,
    _new_slot,
    _exact_observed_throw_slot,
    _team_name_from_block,
    _endpoint_m_from_states,
    _trajectory_m_for_action,
)
from csas.visualize_policy_multi_action_samples import _trajectory_m_for_actions

DEFAULT_NOISE_CONFIG = "/mnt/data/curling2/csas_v3/configs/noise/v1_bowling.json"


# --------------------------------------------------------------------------- #
# WorldModel adapter so csas's value-grid code works unchanged
# --------------------------------------------------------------------------- #
class WorldValueAdapter(nn.Module):
    """Wrap a WorldModel so it presents the csas value-model interface.

    ``forward(x, c)`` returns ``(mean, logvar)`` exactly like the csas Gaussian
    value model, so ``make_value_heatmaps._candidate_heatmap`` / ``_predict_value``
    (which only use ``out[0]``) operate on it without modification.
    """

    def __init__(self, model: WorldModel):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        h = self.model.encode(x, c)
        return self.model.value(h)  # (mean[B], logvar[B])


class CsasValueAdapter(nn.Module):
    """Stand-in for WorldValueAdapter that runs the standalone csas Gaussian value
    model (the human-prior baseline) directly. Used by --prior-value-ckpt in main()
    so we can render heatmaps for the prior without needing a WorldModel checkpoint."""

    def __init__(self, csas_value):
        super().__init__()
        self.csas_value = csas_value

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        out = self.csas_value(x, c)
        if isinstance(out, tuple):
            return out
        return out, torch.zeros_like(out)


def load_world_model(world_ckpt: str, device: torch.device) -> WorldModel:
    ck = torch.load(world_ckpt, map_location=device, weights_only=False)
    model = WorldModel(model_cfg_from_dict(ck["model_cfg"])).to(device)
    load_world_checkpoint(model, world_ckpt, map_location=device)
    model.eval()
    # restore action standardiser (z <-> raw) saved alongside the checkpoint
    if "action_mean" in ck and "action_std" in ck:
        model.set_action_normaliser(ck["action_mean"], ck["action_std"])
    return model


# --------------------------------------------------------------------------- #
# hammer annotation: team_order==1 == has hammer (csas dropped is_hammer because
# it is always identical to team_order).
# --------------------------------------------------------------------------- #
def _hammer_word(cond) -> str:
    return "hammer" if int(round(float(np.asarray(cond)[1]))) == 1 else "no hammer"


def _inject_hammer(case: dict) -> None:
    """Append a (hammer)/(no hammer) tag next to 'thrower team: X' in case['title']."""
    from csas.visualize_policy_prior_samples import _team_name_from_block

    cond = case.get("post_cond", case.get("cond"))
    title = case.get("title")
    if cond is None or not title:
        return
    tag = f"thrower team: {_team_name_from_block(cond)}"
    if tag in title and "hammer" not in title:
        case["title"] = title.replace(tag, f"{tag} ({_hammer_word(cond)})", 1)


# --------------------------------------------------------------------------- #
# Kind 1: value-delta heatmaps  (reuses csas _candidate_heatmap / _predict_value)
# --------------------------------------------------------------------------- #
def value_heatmaps(
    world_ckpt: str,
    out_dir: str,
    device: torch.device,
    horizons: List[int],
    n_real: int,
    *,
    holdout: int = 0,
    split: str = "test",
    grid_n: int = 71,
    extent_m: float = 2.2,
    seed: int = 20260608,
    batch_size: int = 512,
    vlim: Optional[float] = 3.0,
    model: Optional[WorldModel] = None,
) -> List[dict]:
    """Render value-delta heatmaps for ``n_real`` real test states per horizon.

    ``batch_size`` is kept modest because the WorldModel trunk's edge-feature
    computation allocates O(batch * stones^2) tensors (much heavier than the
    plain csas value GraphTF), so the grid (grid_n^2 candidates) is chunked.

    ``vlim`` (default ``3.0``) sets a SHARED symmetric color scale
    (``vmin=-vlim, vmax=+vlim``) so draw + collision heatmaps are comparable. Pass
    ``vlim=None`` to fall back to the per-figure auto-limit.
    """
    if model is None:
        model = load_world_model(world_ckpt, device)
    if isinstance(model, nn.Module) and not isinstance(model, WorldModel):
        adapter = model.to(device).eval()                     # pre-built adapter (e.g. CsasValueAdapter)
    else:
        adapter = WorldValueAdapter(model).to(device).eval()

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[dict] = []
    value_rows: List[dict] = []

    for horizon in horizons:
        cases = heatmap_real_cases(int(n_real), int(seed) + int(horizon), int(holdout), str(split), int(horizon))
        for k, case in enumerate(cases, start=1):
            # sparse horizons (e.g. h10 in the test split) can yield incomplete cases
            if not (("pre_stones" in case or "stones_raw" in case)
                    and "pre_cond" in case and "post_cond" in case and "slot" in case):
                continue
            pre_stones_raw = np.asarray(case.get("pre_stones", case.get("stones_raw")), dtype=np.float32)
            slot = int(case.get("slot", 0))
            _inject_hammer(case)
            xs_m, ys_m, value_delta, pre_value = hv._candidate_heatmap(
                adapter,
                pre_stones_raw,
                case["pre_cond"],
                case["post_cond"],
                slot,
                device,
                int(grid_n),
                float(extent_m),
                int(batch_size),
            )

            # Mask grid cells that fall OUTSIDE the curling sheet, so the heatmap shows
            # only positions where a stone can legally come to rest. Sheet half-width is
            # 2.286 m laterally (raw 750 each side of centerline, * M_PER_RAW). The back
            # line is conventionally 1.829 m behind the tee; stones beyond it are out of
            # play. The forward (release) bound is well beyond our visible range, so no
            # mask is needed there.
            SHEET_HALF_W_M = 750.0 * 0.003048   # 2.286 m
            BACK_LINE_M = -1.829                # back boundary behind the tee
            xx, yy = np.meshgrid(xs_m, ys_m)
            off_sheet = (np.abs(xx) > SHEET_HALF_W_M) | (yy < BACK_LINE_M)
            value_delta = np.where(off_sheet, np.nan, value_delta)

            fig, ax = plt.subplots(figsize=(6.2, 6.8), dpi=180)
            if vlim is not None and float(vlim) > 0:
                lim = float(vlim)  # shared scale across draw + collision heatmaps
            else:
                lim = float(np.nanmax(np.abs(value_delta)))
                if not np.isfinite(lim) or lim <= 1e-6:
                    lim = 1.0
            im = ax.imshow(
                value_delta,
                origin="lower",
                extent=[xs_m.min(), xs_m.max(), ys_m.min(), ys_m.max()],
                cmap="coolwarm",
                vmin=-lim,
                vmax=lim,
                alpha=0.88,
                aspect="equal",
            )
            hv._draw_house(ax)
            hv._plot_stones(ax, pre_stones_raw, thrown_slot=-1)
            # Draw sheet boundary lines so the in-play region is visually unambiguous.
            ax.axvline(-SHEET_HALF_W_M, color="0.35", lw=0.9, ls="--", zorder=2)
            ax.axvline(+SHEET_HALF_W_M, color="0.35", lw=0.9, ls="--", zorder=2)
            ax.axhline(BACK_LINE_M, color="0.35", lw=0.9, ls="--", zorder=2)
            # Clip the visible window to the sheet plus a small margin.
            ax.set_xlim(-SHEET_HALF_W_M - 0.15, SHEET_HALF_W_M + 0.15)
            ax.set_ylim(BACK_LINE_M - 0.15, float(ys_m.max()))
            ax.set_xlabel("lateral from button (m)")
            ax.set_ylabel("along-sheet from button (m)")
            # Break the title across two lines so it never overflows: the human-readable test-state
            # description goes on line 1, and the (horizon, V_pre) annotation goes on line 2.
            ax.set_title(f"{case['title']}\nhorizon $h={horizon}$, $V_{{\\mathrm{{pre}}}}={pre_value:+.2f}$",
                         fontsize=9)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("Predicted draw value differential")
            fig.tight_layout()
            out_path = out_root / f"value_heatmap_{horizon:02d}_{k:02d}_{case['label']}.png"
            fig.savefig(out_path)
            plt.close(fig)
            print(out_path, flush=True)

            manifest_rows.append(
                {"path": str(out_path), "label": case["label"], "title": case["title"],
                 "horizon": int(horizon), "pre_value": float(pre_value)}
            )
            value_rows.append(
                {"label": case["label"], "horizon": int(horizon), "pre_value": float(pre_value),
                 "max_value_delta": float(np.nanmax(value_delta)),
                 "min_value_delta": float(np.nanmin(value_delta))}
            )

    pd.DataFrame(manifest_rows).to_csv(out_root / "manifest.csv", index=False)
    pd.DataFrame(value_rows).to_csv(out_root / "state_values.csv", index=False)
    print(out_root / "manifest.csv", flush=True)
    return manifest_rows


# --------------------------------------------------------------------------- #
# Kind 1b: collision-shot value heatmaps
#
# Instead of sliding the thrown stone over a draw grid (value_heatmaps), here we
# fire a large diverse set of shots that are *intended to collide* with the
# pre-existing stones, then place a colored marker on each struck stone's outline
# at the contact point, colored by the aggregated value-delta of the shots that
# made that contact. This visualises "where on each stone is it valuable to hit?".
# --------------------------------------------------------------------------- #
def _live_stone_centers_plot_m(pre_stones_raw: np.ndarray):
    """Replicate ``_plot_stones``'s raw->plot-meter conversion for LIVE stone centers.

    Returns ``(centers_m[L,2], live_idx[L])`` in the SAME plot frame the thrown
    trajectory uses: x = lateral-from-button, y = (raw_y - button_y) in metres.
    """
    raw = np.asarray(pre_stones_raw, dtype=np.float32).reshape(NUM_STONES, 2)
    live = in_play_raw(raw)
    xy_m = (raw - BUTTON_RAW[None]) * M_PER_RAW  # [NUM_STONES, 2] plot frame
    live_idx = np.flatnonzero(live).astype(np.int64)
    return xy_m[live_idx].astype(np.float32), live_idx


def _build_collision_candidates(pre_state_norm: np.ndarray, n_shots: int,
                                rng: np.random.Generator) -> np.ndarray:
    """Diverse candidate raw [N,4] shots intended to collide with existing stones.

    Combines csas ``diverse_grid_actions`` + ``structured_actions`` with uniform
    random takeout-speed shots aimed across the sheet, then clips to legal bounds.
    """
    parts = []
    grid = env_bridge.diverse_grid_actions(pre_state_norm, int(n_shots))
    if len(grid):
        parts.append(np.asarray(grid, dtype=np.float32))
    struct = env_bridge.structured_actions(pre_state_norm, int(n_shots))
    if len(struct):
        parts.append(np.asarray(struct, dtype=np.float32))

    # Uniform random takeout-speed shots aimed across the sheet (the collision pool):
    # speed in the takeout range ~1.5-2.35, angle/spin/y0 spread over their bounds.
    n_rand = max(int(n_shots), 0)
    if n_rand > 0:
        rand = np.empty((n_rand, 4), dtype=np.float32)
        rand[:, 0] = rng.uniform(1.5, ACTION_SPEED_MAX, size=n_rand)        # takeout speed
        rand[:, 1] = rng.uniform(-0.25, 0.25, size=n_rand)                  # angle
        rand[:, 2] = rng.uniform(ACTION_SPIN_MIN, ACTION_SPIN_MAX, size=n_rand)  # spin
        rand[:, 3] = rng.uniform(-0.23, 0.23, size=n_rand)                  # y0 (lateral aim)
        parts.append(rand)

    actions = np.concatenate(parts, axis=0).astype(np.float32)
    if len(actions) > int(n_shots):
        sel = rng.choice(len(actions), size=int(n_shots), replace=False)
        actions = actions[sel]
    return clip_raw(actions)


def _trajectories_chunked(pre_stones_raw: np.ndarray, actions: np.ndarray,
                          chunk: int = 256):
    """Batch ``_trajectory_m_for_actions`` in chunks of ``chunk`` actions.

    The csas helper returns ONLY finite trajectories (drops any with non-finite
    points), so it can return fewer paths than inputs. We re-run per-chunk and
    keep an index map back to the originating action so contacts stay aligned.
    Returns ``(trajs[list of [T,2]], src_idx[list of int])``.
    """
    trajs: list = []
    src_idx: list = []
    n = len(actions)
    for start in range(0, n, int(chunk)):
        sub = actions[start:start + int(chunk)]
        out = _trajectory_m_for_actions(pre_stones_raw, sub)
        # csas helper preserves order but drops non-finite entries; when counts
        # match it is 1:1, otherwise we cannot disambiguate so fall back per-action.
        if len(out) == len(sub):
            for j, t in enumerate(out):
                trajs.append(t)
                src_idx.append(start + j)
        else:
            for j in range(len(sub)):
                one = _trajectory_m_for_actions(pre_stones_raw, sub[j:j + 1])
                if len(one) == 1 and one[0] is not None and len(one[0]):
                    trajs.append(one[0])
                    src_idx.append(start + j)
    return trajs, src_idx


def collision_heatmaps(
    world_ckpt: str,
    out_dir: str,
    device: torch.device,
    horizons: List[int],
    n_real: int,
    *,
    n_shots: int = 1500,
    ang_bins: int = 24,
    min_samples: int = 8,
    agg: str = "p75",
    vlim: float = 3.0,
    holdout: int = 0,
    split: str = "test",
    seed: int = 20260608,
    traj_chunk: int = 256,
    grid_n: int = 71,
    extent_m: float = 2.2,
    batch_size: int = 512,
    model: Optional[WorldModel] = None,
) -> List[dict]:
    """Render collision-shot value heatmaps for ``n_real`` real states per horizon.

    For each case we fire ~``n_shots`` diverse collision-intent shots, find the
    first contact each makes with a pre-existing live stone, bin contacts by
    (struck stone, angular sector), aggregate the value-delta of shots in each bin
    via ``agg`` ("p75" or "max"), and plot one colored marker per populated bin on
    the struck stone's outline. Color uses the SHARED scale ``[-vlim, +vlim]``.
    """
    import time

    if model is None:
        model = load_world_model(world_ckpt, device)
    adapter = WorldValueAdapter(model).to(device).eval()

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    two_r = 2.0 * float(STONE_RADIUS_M)
    agg = str(agg).lower()
    if agg not in ("p75", "max"):
        raise ValueError(f"agg must be 'p75' or 'max', got {agg!r}")

    timing = {"traj": 0.0, "value": 0.0, "bin_plot": 0.0}
    manifest_rows: List[dict] = []

    for horizon in horizons:
        cases = heatmap_real_cases(int(n_real), int(seed) + int(horizon), int(holdout), str(split), int(horizon))
        for k, case in enumerate(cases, start=1):
            if not (("pre_stones" in case or "stones_raw" in case)
                    and ("post_cond" in case or "cond" in case) and "label" in case):
                continue
            pre_stones_raw = resolve_stone_overlaps_raw(
                np.asarray(case.get("pre_stones", case.get("stones_raw")), dtype=np.float32))
            cond = np.asarray(case.get("post_cond", case.get("cond")), dtype=np.float32)
            pre_cond = np.asarray(case.get("pre_cond", cond), dtype=np.float32)
            centers_m, live_idx = _live_stone_centers_plot_m(pre_stones_raw)
            if len(centers_m) == 0:
                continue

            pre_state_norm = (pre_stones_raw.reshape(-1) / POS_MAX).astype(np.float32)
            rng = np.random.default_rng(int(seed) + int(horizon) * 1000 + int(k))
            actions = _build_collision_candidates(pre_state_norm, int(n_shots), rng)
            n_built = int(len(actions))

            # --- value-delta per shot (same value call the draw heatmaps use) --- #
            t0 = time.perf_counter()
            posts = env_bridge.simulate(pre_state_norm, cond, actions)            # [N,24]
            posts, illegal = env_bridge.apply_legality(pre_state_norm, posts, int(horizon), cond)
            next_cond = env_bridge.next_condition(cond, int(case.get("ShotsInEnd", 10)))
            pre_value = float(_world_value(model, pre_state_norm[None], pre_cond, device)[0])
            post_value = _world_value(model, posts, next_cond, device)
            value_delta = (post_value - pre_value).astype(np.float32)            # thrower perspective
            timing["value"] += time.perf_counter() - t0

            # --- collision contact points (expensive: dynamic trajectories) --- #
            t0 = time.perf_counter()
            trajs, src_idx = _trajectories_chunked(pre_stones_raw, actions, chunk=int(traj_chunk))
            timing["traj"] += time.perf_counter() - t0

            # --- contact detection + binning --- #
            t0 = time.perf_counter()
            # accumulate per (live-stone position in centers_m, ang_bin) -> list of deltas
            bin_deltas: dict = {}
            n_collisions = 0
            edge = np.linspace(-np.pi, np.pi, int(ang_bins) + 1, dtype=np.float64)
            for traj, ai in zip(trajs, src_idx):
                if illegal[ai]:
                    continue
                pts = np.asarray(traj, dtype=np.float32)
                if pts.ndim != 2 or pts.shape[0] == 0:
                    continue
                # first trajectory point within 2r of any live pre-stone center
                hit_stone = -1
                hit_pt = None
                for p in pts:
                    d = np.linalg.norm(centers_m - p[None, :], axis=1)
                    j = int(np.argmin(d))
                    if d[j] <= two_r:
                        hit_stone = j
                        hit_pt = p
                        break
                if hit_stone < 0:
                    continue
                n_collisions += 1
                ctr = centers_m[hit_stone]
                vec = hit_pt - ctr
                if np.linalg.norm(vec) < 1e-9:
                    continue
                ang = float(np.arctan2(vec[1], vec[0]))
                b = int(np.clip(np.searchsorted(edge, ang, side="right") - 1, 0, int(ang_bins) - 1))
                bin_deltas.setdefault((hit_stone, b), []).append(float(value_delta[ai]))
            timing_bin_start = t0

            # aggregate + place one marker per populated bin with enough samples
            marker_xy = []
            marker_val = []
            for (hit_stone, b), deltas in bin_deltas.items():
                if len(deltas) < int(min_samples):
                    continue
                arr = np.asarray(deltas, dtype=np.float32)
                val = float(np.percentile(arr, 75.0)) if agg == "p75" else float(np.max(arr))
                mean_ang = (edge[b] + edge[b + 1]) * 0.5
                u = np.array([np.cos(mean_ang), np.sin(mean_ang)], dtype=np.float32)
                pt = centers_m[hit_stone] + float(STONE_RADIUS_M) * u
                marker_xy.append(pt)
                marker_val.append(val)
            marker_xy = np.asarray(marker_xy, dtype=np.float32).reshape(-1, 2)
            marker_val = np.asarray(marker_val, dtype=np.float32)

            # --- value-delta DRAW heatmap as background (same grid as value_heatmaps) --- #
            slot = int(case.get("slot", 0))
            xs_m, ys_m, value_grid, _pv = hv._candidate_heatmap(
                adapter, pre_stones_raw, pre_cond, cond, slot, device,
                int(grid_n), float(extent_m), int(batch_size))

            # --- plot: value heatmap background + small collision markers on top --- #
            _inject_hammer(case)
            fig, ax = plt.subplots(figsize=(6.6, 7.4), dpi=180)
            im = ax.imshow(
                value_grid, origin="lower",
                extent=[float(xs_m.min()), float(xs_m.max()), float(ys_m.min()), float(ys_m.max())],
                cmap="coolwarm", vmin=-float(vlim), vmax=float(vlim), alpha=0.85,
                aspect="equal", zorder=0,
            )
            _draw_house(ax)
            _plot_stones(ax, pre_stones_raw)
            if len(marker_xy):
                ax.scatter(
                    marker_xy[:, 0], marker_xy[:, 1], c=marker_val, cmap="coolwarm",
                    vmin=-float(vlim), vmax=float(vlim), s=14, edgecolors="black",
                    linewidths=0.3, zorder=8,
                )
            # view: encompass the value-grid region AND all collision markers (e.g. guards)
            ys = [float(extent_m), -float(extent_m)]
            if len(marker_xy):
                ys += [float(marker_xy[:, 1].min()), float(marker_xy[:, 1].max())]
            ax.set_xlim(-2.375, 2.375)
            ax.set_ylim(min(ys) - 0.4, max(ys) + 0.4)
            ax.set_aspect("equal")
            ax.set_xlabel("lateral from button (m)")
            ax.set_ylabel("along-sheet from button (m)")
            tag = f"thrower team: {_team_name_from_block(cond)} ({_hammer_word(cond)})"
            title = str(case.get("title", case["label"]))
            if "hammer" not in title:
                title = f"{title} | {tag}"
            ax.set_title("\n".join(textwrap.wrap(f"h={horizon:02d} | {title}", 62)), fontsize=7)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(f"value Δ = V(post) - V(pre)   (heatmap: draw shot; points: collision {agg})",
                         fontsize=8)
            if not len(marker_xy):
                ax.text(0.5, 0.02, "no collision bins >= min_samples", transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=9, color="0.3")
            fig.tight_layout()
            out_path = out_root / f"collision_{horizon:02d}_{k:02d}_{case['label']}.png"
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            timing["bin_plot"] += time.perf_counter() - timing_bin_start
            print(out_path, flush=True)

            manifest_rows.append({
                "path": str(out_path), "label": case["label"], "title": title,
                "horizon": int(horizon), "n_shots": int(n_built),
                "n_collisions": int(n_collisions), "n_bins_shown": int(len(marker_xy)),
            })

    pd.DataFrame(manifest_rows).to_csv(out_root / "manifest.csv", index=False)
    print(out_root / "manifest.csv", flush=True)
    print(
        f"[collision timing] traj_sim={timing['traj']:.2f}s "
        f"value_eval={timing['value']:.2f}s bin_plot={timing['bin_plot']:.2f}s "
        f"total={sum(timing.values()):.2f}s",
        flush=True,
    )
    return manifest_rows


# --------------------------------------------------------------------------- #
# Kind 2: best decision under execution noise
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _world_value(model: WorldModel, states_norm: np.ndarray, cond: np.ndarray,
                 device: torch.device, batch_size: int = 512) -> np.ndarray:
    """Mean predicted value per state from the WorldModel value head."""
    states = np.atleast_2d(np.asarray(states_norm, dtype=np.float32))
    cond = np.asarray(cond, dtype=np.float32)
    c = np.broadcast_to(cond.reshape(1, -1), (states.shape[0], cond.shape[-1])).astype(np.float32)
    out = np.empty(states.shape[0], dtype=np.float32)
    for i in range(0, states.shape[0], batch_size):
        xb = torch.as_tensor(states[i:i + batch_size], device=device)
        cb = torch.as_tensor(c[i:i + batch_size], device=device)
        h = model.encode(xb, cb)
        out[i:i + batch_size] = model.value_head.value(h).float().cpu().numpy()
    return out


@torch.no_grad()
def _sample_policy_actions(model: WorldModel, x: np.ndarray, c: np.ndarray, n: int,
                           device: torch.device, temperature: float, std_scale: float) -> np.ndarray:
    """Sample raw [N,4] candidate actions from the WorldModel policy head."""
    xb = torch.as_tensor(x.reshape(1, -1), dtype=torch.float32, device=device)
    cb = torch.as_tensor(c.reshape(1, -1), dtype=torch.float32, device=device)
    h = model.encode(xb, cb)
    pi, mu, tril = model.policy(h)
    z = sample_actions_z(pi, mu, tril, n_samples=int(n), temperature=float(temperature),
                         std_scale=float(std_scale))[0]                   # [N,4] z space
    a = z * model.action_std + model.action_mean                          # raw
    a = a.detach().cpu().numpy().astype(np.float32)
    return clip_raw(a)


def _state_value_scalar(model: WorldModel, x: np.ndarray, c: np.ndarray, horizon: int,
                        device: torch.device) -> float:
    if int(horizon) <= 0:
        return env_bridge.score_end(x, int(round(float(c[2]))))
    return float(_world_value(model, x[None], c, device)[0])


def _score_noisy_actions(model: WorldModel, x: np.ndarray, c: np.ndarray, actions: np.ndarray,
                         horizon: int, shots_in_end: int, noise, noise_samples: int,
                         device: torch.device):
    """Mean noisy decision value per candidate action (csas-style)."""
    noisy = noise.sample_batch(actions, int(noise_samples)).reshape(-1, 4)
    posts = env_bridge.simulate(x, c, noisy)
    posts, illegal = env_bridge.apply_legality(x, posts, int(horizon), c)
    if int(horizon) <= 1:
        block = int(round(float(c[2])))
        q_flat = np.asarray([env_bridge.score_end(p, block) for p in posts], dtype=np.float32)
    else:
        nc = env_bridge.next_condition(c, int(shots_in_end))
        q_flat = -_world_value(model, posts, nc, device).astype(np.float32)
    q = q_flat.reshape(len(actions), int(noise_samples))
    illegal = illegal.reshape(len(actions), int(noise_samples))
    mean_q = q.mean(axis=1)
    action_illegal = illegal.any(axis=1)
    mean_q = env_bridge.mask_illegal_scores(mean_q.astype(np.float64), action_illegal).astype(np.float32)
    return mean_q, q, posts.reshape(len(actions), int(noise_samples), -1), illegal


def _plot_noise_endpoints(ax, posts_for_action: np.ndarray, pre_stones: np.ndarray, c: np.ndarray):
    endpoints = _endpoint_m_from_states(posts_for_action, pre_stones, c)
    if len(endpoints):
        ax.scatter(endpoints[:, 0], endpoints[:, 1], s=28, alpha=0.70, color="#0f766e",
                   edgecolors="white", linewidths=0.35, zorder=4)


def _plot_best_decision(case, model, device, out_path: Path, horizon: int, *,
                        n_candidates: int, noise_config: str, noise_samples: int,
                        temperature: float, std_scale: float, seed: int, idx: int,
                        shots_in_end: int = 10) -> dict:
    pre_stones = resolve_stone_overlaps_raw(np.asarray(case.get("pre_stones", case.get("stones_raw")), dtype=np.float32))
    post_stones = case.get("observed_stones")
    cond = np.asarray(case.get("post_cond", case.get("cond")), dtype=np.float32)
    sim_slot = int(case.get("slot", _new_slot(pre_stones, cond)))
    obs_slot = None if post_stones is None else _exact_observed_throw_slot(pre_stones, post_stones)
    team_name = _team_name_from_block(cond)
    x = (pre_stones.reshape(-1) / POS_MAX).astype(np.float32)
    c = cond.astype(np.float32)

    run_seed = int(seed) + int(horizon) * 1000 + int(idx)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)
    noise = make_noise(noise_config, run_seed + 7919)

    pre_v = _state_value_scalar(model, x, c, int(horizon), device)
    actions = _sample_policy_actions(model, x, c, int(n_candidates), device, temperature, std_scale)
    mean_q, q_samples, posts, illegal_samples = _score_noisy_actions(
        model, x, c, actions, int(horizon), int(shots_in_end), noise, int(noise_samples), device)

    surplus = mean_q - pre_v
    best = int(np.argmax(surplus))
    best_action = actions[best]
    best_posts = posts[best]
    best_illegal = illegal_samples[best]
    best_post_raw = best_posts[int(np.argmax(q_samples[best]))].reshape(NUM_STONES, 2) * POS_MAX
    best_slot = _exact_observed_throw_slot(pre_stones, best_post_raw)
    if best_slot is None:
        best_slot = sim_slot
    best_traj = _trajectory_m_for_action(pre_stones, cond, best_action)
    noisy_trajs = _trajectory_m_for_actions(
        pre_stones, noise.sample_batch(best_action[None], min(8, int(noise_samples))).reshape(-1, 4))
    obs_traj = _trajectory_m_for_action(pre_stones, cond, case.get("inverse_action"))

    fig, axes = plt.subplots(1, 5, figsize=(20.8, 5.8), dpi=170,
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.08]})

    ax = axes[0]
    _draw_house(ax)
    _plot_stones(ax, pre_stones)
    if obs_traj is not None and len(obs_traj):
        ax.plot(obs_traj[:, 0], obs_traj[:, 1], linestyle=":", color="#dc2626", lw=1.6, alpha=0.9)
    ax.set_title("pre-throw state")

    ax = axes[1]
    _draw_house(ax)
    if post_stones is not None:
        _plot_stones(ax, post_stones)
        if obs_slot is not None:
            _plot_highlighted_slot(ax, post_stones, obs_slot, color="#f2c14e")
    else:
        _plot_stones(ax, pre_stones)
    ax.set_title("observed post-throw")

    ax = axes[2]
    _draw_house(ax)
    _plot_stones(ax, best_post_raw)
    _plot_highlighted_slot(ax, best_post_raw, best_slot, color="#8ecae6")
    if best_traj is not None and len(best_traj):
        ax.plot(best_traj[:, 0], best_traj[:, 1], linestyle=":", color="#1d4ed8", lw=1.8, alpha=0.95)
    ax.set_title(f"best noisy-sample outcome\nillegal replaced: {int(best_illegal.sum())}/{len(best_illegal)}")

    ax = axes[3]
    _draw_house(ax)
    _plot_stones(ax, pre_stones)
    for traj in noisy_trajs:
        ax.plot(traj[:, 0], traj[:, 1], color="#2563eb", lw=1.0, alpha=0.28, zorder=2)
    if best_traj is not None and len(best_traj):
        ax.plot(best_traj[:, 0], best_traj[:, 1], linestyle=":", color="#1d4ed8", lw=1.5, alpha=0.85)
    _plot_noise_endpoints(ax, best_posts, pre_stones, c)
    ax.set_title(f"best intended shot\n{len(best_illegal)} local-noise executions")

    axes[4].hist(surplus, bins=36, color="#64748b", alpha=0.7)
    axes[4].axvline(float(surplus[best]), color="#dc2626", lw=1.6, label="best")
    axes[4].axvline(0.0, color="0.2", lw=1.0, linestyle="--")
    axes[4].set_title("candidate mean value surplus")
    axes[4].set_xlabel("mean noisy Q(post) - V(pre)")
    axes[4].set_ylabel("candidate count")
    axes[4].legend(fontsize=8)

    for ax in axes[:4]:
        ax.set_xlim(-2.375, 2.375)
        ax.set_ylim(-6.40, 6.40)
        ax.set_aspect("equal")
        ax.set_xlabel("lateral from button (m)")
        ax.set_ylabel("along-sheet from button (m)")

    thrower = f"{team_name} stone {obs_slot + 1}" if obs_slot is not None else f"{team_name} team, slot ambiguous"
    fig.suptitle(
        f"world | h={horizon:02d} | best decision shot | "
        f"mean surplus={surplus[best]:+.3f} | mean Q={mean_q[best]:+.3f} | pre V={pre_v:+.3f} | "
        f"candidates={len(actions)} | thrower team: {team_name} ({_hammer_word(cond)}) | thrower: {thrower}",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "path": str(out_path),
        "label": case["label"],
        "title": case["title"],
        "horizon": int(horizon),
        "pre_value": float(pre_v),
        "best_mean_q": float(mean_q[best]),
        "best_mean_surplus": float(surplus[best]),
        "best_q_std": float(q_samples[best].std()),
        "best_illegal_count": int(best_illegal.sum()),
        "n_candidates": int(len(actions)),
        "best_action_speed": float(best_action[0]),
        "best_action_angle": float(best_action[1]),
        "best_action_spin": float(best_action[2]),
        "best_action_y0": float(best_action[3]),
    }


def best_decision_noisy(
    world_ckpt: str,
    out_dir: str,
    device: torch.device,
    horizons: List[int],
    n_real: int,
    noise_config: str,
    *,
    noise_samples: int = 16,
    n_candidates: int = 96,
    temperature: float = 0.9,
    std_scale: float = 1.1,
    holdout: int = 0,
    split: str = "test",
    seed: int = 20260608,
    model: Optional[WorldModel] = None,
) -> List[dict]:
    """Render the best-decision-with-execution-noise figure per held-out state."""
    if model is None:
        model = load_world_model(world_ckpt, device)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[dict] = []
    for horizon in horizons:
        cases = heatmap_real_cases(int(n_real), int(seed) + int(horizon), int(holdout), str(split), int(horizon))
        for i, case in enumerate(cases, start=1):
            if not (("pre_stones" in case or "stones_raw" in case)
                    and ("post_cond" in case or "cond" in case) and "label" in case):
                continue
            out_path = out_root / f"best_decision_{horizon:02d}_{i:02d}_{case['label']}.png"
            row = _plot_best_decision(
                case, model, device, out_path, int(horizon),
                n_candidates=int(n_candidates), noise_config=noise_config,
                noise_samples=int(noise_samples), temperature=float(temperature),
                std_scale=float(std_scale), seed=int(seed), idx=i,
                shots_in_end=int(case.get("ShotsInEnd", 10)),
            )
            manifest_rows.append(row)
            print(out_path, flush=True)

    pd.DataFrame(manifest_rows).to_csv(out_root / "manifest.csv", index=False)
    print(out_root / "manifest.csv", flush=True)
    return manifest_rows


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", required=False, default=None,
                    help="WorldModel checkpoint (.pt). Required unless --prior-value-ckpt is given.")
    ap.add_argument("--prior-value-ckpt", default=None,
                    help="Standalone csas Gaussian value model (the human-prior baseline). When set, "
                         "value heatmaps use this directly via CsasValueAdapter (no WorldModel needed); "
                         "other kinds (best_decision/collision) still need --world.")
    ap.add_argument("--value-subdir", default="value_heatmaps_world",
                    help="Subdirectory under --out-root for value heatmaps. Use 'value_heatmaps' for the prior.")
    ap.add_argument("--out-root", default="artifacts/figures")
    ap.add_argument("--horizons", type=int, nargs="+", default=[3, 6, 9])
    ap.add_argument("--n-real", type=int, default=4)
    ap.add_argument("--noise-config", default=DEFAULT_NOISE_CONFIG)
    ap.add_argument("--noise-samples", type=int, default=16)
    ap.add_argument("--n-candidates", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--std-scale", type=float, default=1.1)
    ap.add_argument("--holdout", type=int, default=0)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--grid", type=int, default=71)
    ap.add_argument("--extent-m", type=float, default=2.2)
    ap.add_argument("--eval-batch-size", type=int, default=512,
                    help="Chunk size for WorldModel value evals (edge features are memory-heavy).")
    ap.add_argument("--seed", type=int, default=20260608)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--kind", choices=["value", "best_decision", "collision", "both", "all"], default="both")
    # collision-heatmap args
    ap.add_argument("--n-shots", type=int, default=1500, help="Diverse collision-intent shots per case.")
    ap.add_argument("--ang-bins", type=int, default=24, help="Angular sectors per stone for collision binning.")
    ap.add_argument("--min-samples", type=int, default=8, help="Min contacts in a bin to render a marker.")
    ap.add_argument("--agg", choices=["p75", "max"], default="p75", help="Per-bin value-delta aggregator.")
    ap.add_argument("--vlim", type=float, default=3.0,
                    help="Shared symmetric color limit for value AND collision heatmaps.")
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    # Warm the JAX simulator once (compile cost) before timing the figures.
    if args.kind in ("best_decision", "collision", "both", "all"):
        env_bridge.warm_jax()

    # value-only mode with --prior-value-ckpt: skip loading any WorldModel; use the standalone
    # csas Gaussian value model directly via CsasValueAdapter.
    use_prior = (args.kind == "value" and args.prior_value_ckpt)
    if use_prior:
        model = CsasValueAdapter(env_bridge.load_csas_value(args.prior_value_ckpt, device))
    else:
        if args.world is None:
            raise SystemExit("--world is required unless --kind=value with --prior-value-ckpt")
        model = load_world_model(args.world, device)
    out_root = Path(args.out_root)

    if args.kind in ("value", "both", "all"):
        value_heatmaps(
            args.world or "", str(out_root / args.value_subdir), device,
            list(args.horizons), int(args.n_real),
            holdout=int(args.holdout), split=args.split,
            grid_n=int(args.grid), extent_m=float(args.extent_m), seed=int(args.seed),
            batch_size=int(args.eval_batch_size), vlim=float(args.vlim),
            model=model,
        )
    if args.kind in ("best_decision", "both", "all"):
        best_decision_noisy(
            args.world, str(out_root / "best_decision_world"), device,
            list(args.horizons), int(args.n_real), args.noise_config,
            noise_samples=int(args.noise_samples), n_candidates=int(args.n_candidates),
            temperature=float(args.temperature), std_scale=float(args.std_scale),
            holdout=int(args.holdout), split=args.split, seed=int(args.seed),
            model=model,
        )
    if args.kind in ("collision", "all"):
        collision_heatmaps(
            args.world, str(out_root / "collision_heatmaps_world"), device,
            list(args.horizons), int(args.n_real),
            n_shots=int(args.n_shots), ang_bins=int(args.ang_bins),
            min_samples=int(args.min_samples), agg=str(args.agg), vlim=float(args.vlim),
            holdout=int(args.holdout), split=args.split, seed=int(args.seed),
            model=model,
        )


if __name__ == "__main__":
    main()
