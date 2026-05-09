# make_bc_dataset_parallel.py
# Parallel dataset builder for Stones.csv
#
# - Spawns one worker per visible GPU.
# - Processes the dataset in CHUNKS of 5000 shots (configurable).
# - Each worker writes periodic per-GPU CSV parts (default every 500 rows).
# - After each 5000-shot chunk completes, the parent merges parts into
#   a single chunk CSV: <out_prefix>.chunkXXXX.csv.
#
# Usage:
#   python make_bc_dataset_parallel.py \
#       --csv /path/to/Stones.csv \
#       --out-prefix stones_with_estimates \
#       [--chunk-size 5000] [--flush-every 500] [--limit N] [--seed 0]
#
# Requirements (same repo):
#   - curling_sim_jax.py
#   - curling_inverse.py
#
# Notes:
# - We intentionally import JAX and curling_* inside each worker process
#   AFTER pinning CUDA_VISIBLE_DEVICES, to ensure one GPU per worker.
# - We avoid cross-process writes to a single file; workers emit parts
#   and the parent merges per chunk.

import os
import sys
import ctypes
import math
import re
import glob
import shutil
import random
import argparse
import tempfile
import pathlib
from dataclasses import dataclass, replace as dataclass_replace
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
from sim_presets import CONTACT_MILD_SIM_KWARGS

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs): return x


def _preload_nvidia_cuda_libs():
    if os.environ.get("CSAS_PRELOAD_NVIDIA_LIBS", "1").lower() not in {"1", "true", "yes"}:
        return
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    roots = [
        pathlib.Path(sys.prefix) / "lib" / pyver / "site-packages" / "nvidia",
        pathlib.Path("/opt/pytorch/lib") / pyver / "site-packages" / "nvidia",
    ]
    rels = [
        "nvjitlink/lib/libnvJitLink.so.12",
        "cuda_runtime/lib/libcudart.so.12",
        "cuda_nvrtc/lib/libnvrtc.so.12",
        "cublas/lib/libcublasLt.so.12",
        "cublas/lib/libcublas.so.12",
        "cusparse/lib/libcusparse.so.12",
        "cusolver/lib/libcusolver.so.11",
        "cufft/lib/libcufft.so.11",
        "cudnn/lib/libcudnn.so.9",
    ]
    seen = set()
    for root in roots:
        for rel in rels:
            lib = root / rel
            if not lib.exists():
                continue
            key = str(lib)
            if key in seen:
                continue
            seen.add(key)
            try:
                ctypes.CDLL(key, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue

# ---------------- CSV / unit conversion constants ----------------
SENTINEL_OFF = 4095
CSV_STONE_COUNT = 12
SHOT_KEY_COLS = ["CompetitionID", "SessionID", "GameID", "EndID", "ShotID"]
PARAM_COLS = ["est_speed", "est_angle", "est_spin", "est_y0"]

# CSV → meters for the simulator button frame
CSV_BUTTON_Y = 800.0
CSV_CENTER_X = 750.0
CSV_TO_M = 0.003048  # (800 - y)*0.003048 → along-sheet meters; (x - 750)*0.003048 → lateral meters

# Small separation for overlapped previous stones (sanitization)
STONE_RADIUS_M = 0.145
MIN_CLEAR = 2 * STONE_RADIUS_M + 1e-3
SEPARATE_PASSES = 6
PAD_POS_M = np.array([50.0, 50.0], dtype=np.float32)
# -----------------------------------------------------------------


@dataclass(frozen=True)
class ShotKey:
    comp: int
    sess: int
    game: int
    end: int
    shot: int  # ShotID for the "after" row (the one we estimate for)


@dataclass(frozen=True)
class SimConfig:
    refine_dt: float = 0.02
    refine_substeps: int = 2
    coarse_dt: float = 0.03
    coarse_substeps: int = 1
    coarse_max_steps: int = 900
    refine_k_penalty: float = 2.5e4
    coarse_k_penalty: float = 2.0e4
    c_damp: float = CONTACT_MILD_SIM_KWARGS["c_damp"]
    c_damp_sep_frac: float = CONTACT_MILD_SIM_KWARGS["c_damp_sep_frac"]
    c_tangent: float = CONTACT_MILD_SIM_KWARGS["c_tangent"]
    mu_tangent: float = CONTACT_MILD_SIM_KWARGS["mu_tangent"]
    spin_contact: float = CONTACT_MILD_SIM_KWARGS["spin_contact"]
    k_curl: float = CONTACT_MILD_SIM_KWARGS["k_curl"]
    a_linear: float = CONTACT_MILD_SIM_KWARGS["a_linear"]
    gamma_spin: float = CONTACT_MILD_SIM_KWARGS["gamma_spin"]


# --------- CSV helpers (no JAX imports here) ---------
def _csv_y_to_xm(y_csv: float) -> float:
    # Along-sheet, 0 at button; negative is toward the thrower-side hog line.
    return (CSV_BUTTON_Y - y_csv) * CSV_TO_M


def _csv_x_to_ym(x_csv: float) -> float:
    # Lateral, 0 at centerline
    return (x_csv - CSV_CENTER_X) * CSV_TO_M


def _valid_xy(x_csv: float, y_csv: float) -> bool:
    # 0 => not yet thrown, 4095 => off sheet (dead)
    if x_csv in (0, SENTINEL_OFF) or y_csv in (0, SENTINEL_OFF):
        return False
    return True


def _get_xy_from_row(row: pd.Series, i: int) -> Tuple[Optional[float], Optional[float]]:
    # Case-robust stone_i_x/y lookup
    for (kx, ky) in ((f"Stone_{i}_x", f"Stone_{i}_y"), (f"stone_{i}_x", f"stone_{i}_y")):
        if kx in row and ky in row:
            xv, yv = row[kx], row[ky]
            return (None if pd.isna(xv) else float(xv),
                    None if pd.isna(yv) else float(yv))
    return (None, None)


def _row_to_state_xy(row: pd.Series) -> Dict[int, Tuple[float, float]]:
    """Return dict: stone_index -> (x_m, y_m) for stones that are in play on this row."""
    out: Dict[int, Tuple[float, float]] = {}
    for i in range(1, CSV_STONE_COUNT + 1):
        xi, yi = _get_xy_from_row(row, i)
        if xi is None or yi is None:
            continue
        if not _valid_xy(xi, yi):
            continue
        xm = _csv_y_to_xm(yi)
        ym = _csv_x_to_ym(xi)
        out[i] = (xm, ym)
    return out


def _separate_overlaps(pts: np.ndarray, min_gap: float = MIN_CLEAR, passes: int = SEPARATE_PASSES) -> np.ndarray:
    """Deterministic relaxation: push apart any overlapping stones (prev state only)."""
    if pts.size == 0:
        return pts
    p = pts.copy()
    n = p.shape[0]
    for _ in range(passes):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = p[j, 0] - p[i, 0]
                dy = p[j, 1] - p[i, 1]
                d = math.hypot(dx, dy)
                if d < 1e-9:
                    dx, dy, d = 1e-6, 0.0, 1e-6
                if d < min_gap:
                    push = 0.5 * (min_gap - d)
                    nx, ny = dx / d, dy / d
                    p[i, 0] -= push * nx
                    p[i, 1] -= push * ny
                    p[j, 0] += push * nx
                    p[j, 1] += push * ny
                    moved = True
        if not moved:
            break
    return p


def _iter_shots_with_prev(df: pd.DataFrame):
    """Yield ShotKey for every row that has a previous row in the same (comp,sess,game,end)."""
    group_cols = ["CompetitionID", "SessionID", "GameID", "EndID"]
    for (comp, sess, game, end), df_end in df.groupby(group_cols, sort=False):
        df_end = df_end.sort_values("ShotID", ascending=True)
        shots = df_end["ShotID"].tolist()
        for idx in range(1, len(shots)):
            yield ShotKey(int(comp), int(sess), int(game), int(end), int(shots[idx]))


def _count_shots_with_prev(df: pd.DataFrame) -> int:
    group_cols = ["CompetitionID", "SessionID", "GameID", "EndID"]
    total = 0
    for _, df_end in df.groupby(group_cols, sort=False):
        df_end = df_end.sort_values("ShotID", ascending=True)
        n = df_end.shape[0]
        if n >= 2:
            total += (n - 1)
    return total


def _load_prev_next_states(df: pd.DataFrame, key: ShotKey):
    """Return prev/next dicts keyed by stone_id and the current row metadata."""
    df_end = df[
        (df["CompetitionID"] == key.comp) &
        (df["SessionID"] == key.sess) &
        (df["GameID"] == key.game) &
        (df["EndID"] == key.end)
    ].sort_values("ShotID", ascending=True)
    shots = df_end["ShotID"].tolist()
    idx = shots.index(key.shot)

    prev_row = df_end.iloc[idx - 1]
    next_row = df_end.iloc[idx]
    prev_state_m = _row_to_state_xy(prev_row)
    next_state_m = _row_to_state_xy(next_row)
    return prev_state_m, next_state_m, next_row


def _fill_flat_arrays(state_m: Dict[int, Tuple[float, float]]):
    """Return flat arrays shaped (12,2) with NaN where absent."""
    arr = np.full((CSV_STONE_COUNT, 2), np.nan, dtype=np.float32)
    for k, (xm, ym) in state_m.items():
        if 1 <= k <= CSV_STONE_COUNT:
            arr[k - 1, 0] = xm
            arr[k - 1, 1] = ym
    return arr


def _state_to_fixed_slot_arrays(state_m: Dict[int, Tuple[float, float]]):
    arr = np.repeat(PAD_POS_M[None, :], CSV_STONE_COUNT, axis=0).astype(np.float32, copy=True)
    mask = np.zeros((CSV_STONE_COUNT,), dtype=bool)
    for k, (xm, ym) in state_m.items():
        if 1 <= k <= CSV_STONE_COUNT:
            arr[k - 1, 0] = float(xm)
            arr[k - 1, 1] = float(ym)
            mask[k - 1] = True
    return arr, mask


def _targets_for_block(state_m: Dict[int, Tuple[float, float]], start_sid: int, end_sid: int):
    n = end_sid - start_sid + 1
    arr = np.repeat(PAD_POS_M[None, :], n, axis=0).astype(np.float32, copy=True)
    mask = np.zeros((n,), dtype=bool)
    for offset, sid in enumerate(range(start_sid, end_sid + 1)):
        if sid in state_m:
            xm, ym = state_m[sid]
            arr[offset, 0] = float(xm)
            arr[offset, 1] = float(ym)
            mask[offset] = True
    return arr, mask


def _slot_present_mask_from_row(row: pd.Series) -> np.ndarray:
    mask = np.zeros((CSV_STONE_COUNT,), dtype=bool)
    for i in range(1, CSV_STONE_COUNT + 1):
        xi, yi = _get_xy_from_row(row, i)
        if xi is None or yi is None:
            continue
        mask[i - 1] = _valid_xy(xi, yi)
    return mask


def _mode_or_round_mean(vals: pd.Series) -> float:
    mode_vals = vals.mode(dropna=True)
    if not mode_vals.empty:
        return float(mode_vals.iloc[0])
    mean_val = vals.mean()
    if pd.isna(mean_val):
        return np.nan
    return float(np.round(float(mean_val)))


def _attach_throw_slot_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds per-row throw-slot hints from observed transitions:
      - obs_throw_slot_id: exact added slot id when exactly one slot is added
      - team_slot_block: per-(end,TeamID) mode block (0 => slots 1..6, 1 => 7..12)
    """
    out = df.sort_values(["CompetitionID", "SessionID", "GameID", "EndID", "ShotID"]).reset_index(drop=True).copy()
    out["obs_throw_slot_id"] = np.nan

    end_group = ["CompetitionID", "SessionID", "GameID", "EndID"]
    for _, idx in out.groupby(end_group, sort=False).groups.items():
        idx_list = list(idx)
        prev_mask = None
        for ridx in idx_list:
            cur_mask = _slot_present_mask_from_row(out.loc[ridx])
            if prev_mask is not None:
                added = cur_mask & (~prev_mask)
                if int(np.sum(added)) == 1:
                    out.at[ridx, "obs_throw_slot_id"] = float(np.flatnonzero(added)[0] + 1)
            prev_mask = cur_mask

    obs_throw_slot = out["obs_throw_slot_id"].to_numpy(dtype=np.float32)
    out["obs_throw_block"] = np.where(np.isfinite(obs_throw_slot), (obs_throw_slot > 6).astype(np.float32), np.nan)

    group_cols = ["CompetitionID", "SessionID", "GameID", "EndID", "TeamID"]
    valid = out[np.isfinite(out["obs_throw_block"])].copy()
    if not valid.empty:
        mode_df = (
            valid.groupby(group_cols, dropna=False)["obs_throw_block"]
            .agg(_mode_or_round_mean)
            .reset_index()
            .rename(columns={"obs_throw_block": "team_slot_block"})
        )
        out = pd.merge(out, mode_df, on=group_cols, how="left")
    else:
        out["team_slot_block"] = np.nan

    # Fill the missing team in an end by complementing the known team block.
    for _, idx in out.groupby(end_group, sort=False).groups.items():
        idx_list = list(idx)
        end_rows = out.loc[idx_list, ["TeamID", "team_slot_block"]].drop_duplicates()
        team_ids = [int(v) for v in out.loc[idx_list, "TeamID"].dropna().unique().tolist()]
        known = end_rows[end_rows["team_slot_block"].notna()]
        if len(team_ids) == 2 and known["TeamID"].nunique() == 1:
            known_tid = int(known["TeamID"].iloc[0])
            known_block = float(known["team_slot_block"].iloc[0])
            other_tid = team_ids[0] if team_ids[1] == known_tid else team_ids[1]
            fill_mask = out.index.isin(idx_list) & (out["TeamID"] == other_tid) & (out["team_slot_block"].isna())
            out.loc[fill_mask, "team_slot_block"] = 1.0 - known_block

    if "obs_throw_block" in out.columns:
        out = out.drop(columns=["obs_throw_block"])
    return out


def _load_warm_start_df(warm_start_glob: str) -> Optional[pd.DataFrame]:
    if not warm_start_glob:
        return None

    files = sorted(glob.glob(warm_start_glob))
    if not files:
        print(f"[warn] no warm-start inverse files matched: {warm_start_glob}")
        return None

    prop_pat = re.compile(r"prop\d+_est_(speed|angle|spin|y0)$")
    keep_cols = SHOT_KEY_COLS + PARAM_COLS + ["solver_ok", "hard_loss_refine"]
    dfs = []
    for path in files:
        try:
            dfi = pd.read_csv(path, usecols=lambda c: (c in keep_cols) or bool(prop_pat.fullmatch(str(c))))
        except ValueError:
            dfi = pd.read_csv(path)
            dfi = dfi[[c for c in dfi.columns if (c in keep_cols) or bool(prop_pat.fullmatch(str(c)))]]
        dfs.append(dfi)

    warm = pd.concat(dfs, ignore_index=True)
    if warm.empty:
        return None

    warm = warm.drop_duplicates(subset=SHOT_KEY_COLS, keep="first").copy()
    rename_map = {
        "est_speed": "warm_est_speed",
        "est_angle": "warm_est_angle",
        "est_spin": "warm_est_spin",
        "est_y0": "warm_est_y0",
        "solver_ok": "warm_solver_ok",
        "hard_loss_refine": "warm_hard_loss_refine",
    }
    warm = warm.rename(columns=rename_map)
    return warm


def _proposal_bank_from_row(row: pd.Series) -> Optional[np.ndarray]:
    prop_pat = re.compile(r"prop(\d+)_est_(speed|angle|spin|y0)$")
    per_prop: Dict[int, Dict[str, float]] = {}
    for col in row.index:
        m = prop_pat.fullmatch(str(col))
        if not m:
            continue
        idx = int(m.group(1))
        field = m.group(2)
        per_prop.setdefault(idx, {})[field] = float(row.get(col, np.nan))
    props: List[np.ndarray] = []
    for idx in sorted(per_prop):
        fields = per_prop[idx]
        vals = np.asarray(
            [
                fields.get("speed", np.nan),
                fields.get("angle", np.nan),
                fields.get("spin", np.nan),
                fields.get("y0", np.nan),
            ],
            dtype=np.float32,
        )
        if np.all(np.isfinite(vals)):
            props.append(vals)
    if not props:
        return None
    arr = np.unique(np.round(np.asarray(props, dtype=np.float32), 6), axis=0).astype(np.float32)
    return arr


def infer_thrower_block(
    prev_ids: List[int],
    next_ids: List[int],
    obs_throw_slot_id: float,
    team_slot_block: float,
) -> int:
    if np.isfinite(obs_throw_slot_id):
        return 0 if int(round(float(obs_throw_slot_id))) <= 6 else 1

    added = [sid for sid in next_ids if sid not in prev_ids]
    if len(added) == 1:
        return 0 if added[0] <= 6 else 1

    if np.isfinite(team_slot_block):
        return int(np.clip(round(float(team_slot_block)), 0, 1))
    return 0


def choose_new_slot_id(
    prev_ids: List[int],
    next_ids: List[int],
    thrower_block: int,
    obs_throw_slot_id: float,
) -> int:
    if np.isfinite(obs_throw_slot_id):
        return int(round(float(obs_throw_slot_id)))

    added = [sid for sid in next_ids if sid not in prev_ids]
    if len(added) == 1:
        return int(added[0])

    missing = [sid for sid in range(1, 13) if sid not in prev_ids]
    if not missing:
        return prev_ids[-1] if prev_ids else 12

    if int(thrower_block) == 0:
        block_missing = [sid for sid in missing if sid <= 6]
    else:
        block_missing = [sid for sid in missing if sid >= 7]
    if block_missing:
        return block_missing[0]
    return missing[0]


# -------------------- Worker logic --------------------
def _worker_run(
    gpu_local_index: int,
    gpu_visible_id: str,
    keys: List[ShotKey],
    df_end_index: pd.DataFrame,  # pre-filtered to a chunk; we pass a whole df for simplicity
    out_dir: str,
    chunk_idx: int,
    base_seed: int,
    flush_every: int,
    solver_method: str,
    loss_variant: str,
    sim_cfg: SimConfig,
    verbose: bool = False,
):
    """
    Single-GPU worker. Pins to one GPU and processes its assigned keys,
    writing periodic part CSVs to out_dir.
    """
    # Pin this process to a single GPU before importing jax/curling libs.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_visible_id)
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")  # safer fragmentation behavior
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    nvidia_root = pathlib.Path(sys.prefix) / "lib" / pyver / "site-packages" / "nvidia"
    lib_dirs = [
        nvidia_root / "nvjitlink" / "lib",
        nvidia_root / "cuda_runtime" / "lib",
        nvidia_root / "cuda_nvrtc" / "lib",
        nvidia_root / "cublas" / "lib",
        nvidia_root / "cusparse" / "lib",
        nvidia_root / "cusolver" / "lib",
        nvidia_root / "cufft" / "lib",
        nvidia_root / "cudnn" / "lib",
    ]
    lib_dirs = [str(p) for p in lib_dirs if p.exists()]
    if lib_dirs:
        os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs + [os.environ.get("LD_LIBRARY_PATH", "")])
        os.environ["JAX_WHEEL_CUDNN_DIR"] = str(nvidia_root / "cudnn" / "lib")

    # Import JAX + sim libs AFTER pinning.
    _preload_nvidia_cuda_libs()
    import jax
    import jax.numpy as jnp
    from curling_sim_jax import CurlingParams
    from curling_inverse import (
        build_batched_hard_loss_by_block,
        solve_inverse_by_block,
        SolveBounds,
        MIN_X, MAX_X, MIN_Y, MAX_Y,
    )

    def _in_bounds_mask_np(pos_xy: np.ndarray) -> np.ndarray:
        if pos_xy.size == 0:
            return np.zeros((0,), dtype=bool)
        x = pos_xy[:, 0]
        y = pos_xy[:, 1]
        return (x > MIN_X) & (x < MAX_X) & (y > MIN_Y) & (y < MAX_Y)

    p_refine = CurlingParams(
        dt=float(sim_cfg.refine_dt),
        substeps=int(sim_cfg.refine_substeps),
        k_penalty=float(sim_cfg.refine_k_penalty),
        c_damp=float(sim_cfg.c_damp),
        c_damp_sep_frac=float(sim_cfg.c_damp_sep_frac),
        c_tangent=float(sim_cfg.c_tangent),
        mu_tangent=float(sim_cfg.mu_tangent),
        spin_contact=float(sim_cfg.spin_contact),
        k_curl=float(sim_cfg.k_curl),
        a_linear=float(sim_cfg.a_linear),
        gamma_spin=float(sim_cfg.gamma_spin),
    )
    p_coarse = dataclass_replace(
        p_refine,
        dt=float(sim_cfg.coarse_dt),
        substeps=int(sim_cfg.coarse_substeps),
        max_steps=int(sim_cfg.coarse_max_steps),
        k_penalty=float(sim_cfg.coarse_k_penalty),
    )
    bounds = SolveBounds()
    lo = np.array([bounds.speed_min, bounds.angle_min, bounds.spin_min, bounds.y0_min], dtype=np.float32)
    hi = np.array([bounds.speed_max, bounds.angle_max, bounds.spin_max, bounds.y0_max], dtype=np.float32)
    span = hi - lo
    batched_hard_refine = build_batched_hard_loss_by_block(p_refine, loss_variant=loss_variant)
    batched_hard_coarse = build_batched_hard_loss_by_block(p_coarse, loss_variant=loss_variant)
    if loss_variant == "current":
        batched_hard_refine_baseline = batched_hard_refine
        batched_hard_coarse_baseline = batched_hard_coarse
    else:
        batched_hard_refine_baseline = build_batched_hard_loss_by_block(p_refine, loss_variant="current")
        batched_hard_coarse_baseline = build_batched_hard_loss_by_block(p_coarse, loss_variant="current")
    batched_hard_cache = {
        ("refine", loss_variant): batched_hard_refine,
        ("coarse", loss_variant): batched_hard_coarse,
        ("refine", "current"): batched_hard_refine_baseline,
        ("coarse", "current"): batched_hard_coarse_baseline,
    }

    def _get_batched_hard(which: str, variant: str):
        key = (str(which), str(variant))
        fn = batched_hard_cache.get(key)
        if fn is None:
            params = p_refine if which == "refine" else p_coarse
            fn = build_batched_hard_loss_by_block(params, loss_variant=variant)
            batched_hard_cache[key] = fn
        return fn

    def _x01_from_phys_np(x_phys: np.ndarray) -> np.ndarray:
        arr = np.asarray(x_phys, dtype=np.float32)
        return np.clip((arr - lo) / (span + 1e-8), 0.0, 1.0).astype(np.float32)

    def _xphys_from_x01_np(x01: np.ndarray) -> np.ndarray:
        arr = np.asarray(x01, dtype=np.float32)
        return (lo + np.clip(arr, 0.0, 1.0) * span).astype(np.float32)

    def _pick_distinct_indices(
        X01: np.ndarray,
        losses: np.ndarray,
        max_keep: int,
        min_dist: float,
    ) -> List[int]:
        order = np.argsort(losses)
        chosen: List[int] = []
        for idx in order.tolist():
            cand = X01[idx]
            if all(np.linalg.norm(cand - X01[j]) >= min_dist for j in chosen):
                chosen.append(int(idx))
                if len(chosen) >= max_keep:
                    break
        if not chosen and len(order) > 0:
            chosen.append(int(order[0]))
        if len(chosen) < max_keep:
            for idx in order.tolist():
                if int(idx) not in chosen:
                    chosen.append(int(idx))
                    if len(chosen) >= max_keep:
                        break
        return chosen

    def _make_seed_bank(
        seed: int,
        init_x: Optional[np.ndarray],
        num_uniform: int = 160,
        num_local: int = 96,
        axis_step: float = 0.14,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        center = np.full((1, 4), 0.5, dtype=np.float32)
        base = center[0]
        seeds = [center]
        if init_x is not None:
            base = _x01_from_phys_np(init_x)
            seeds.append(base[None, :])
        axis = []
        for dim in range(4):
            e = np.zeros((4,), dtype=np.float32)
            e[dim] = axis_step
            axis.append(np.clip(base + e, 0.0, 1.0))
            axis.append(np.clip(base - e, 0.0, 1.0))
        if axis:
            seeds.append(np.asarray(axis, dtype=np.float32))
        seeds.append(rng.uniform(0.0, 1.0, size=(num_uniform, 4)).astype(np.float32))
        local_base = base[None, :]
        local_sigma = np.asarray([0.10, 0.10, 0.16, 0.10], dtype=np.float32)
        local = local_base + rng.normal(size=(num_local, 4)).astype(np.float32) * local_sigma[None, :]
        seeds.append(np.clip(local, 0.0, 1.0).astype(np.float32))
        X01 = np.concatenate(seeds, axis=0)
        X01 = np.unique(np.round(X01, 6), axis=0).astype(np.float32)
        return X01

    def _make_seed_bank_large(
        seed: int,
        init_x: Optional[np.ndarray],
        num_uniform: int = 224,
        num_local_narrow: int = 96,
        num_local_wide: int = 96,
        axis_step: float = 0.14,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        center = np.full((1, 4), 0.5, dtype=np.float32)
        base = center[0]
        seeds = [center]
        if init_x is not None:
            base = _x01_from_phys_np(init_x)
            seeds.append(base[None, :])

        axis = []
        for dim in range(4):
            e = np.zeros((4,), dtype=np.float32)
            e[dim] = axis_step
            axis.append(np.clip(base + e, 0.0, 1.0))
            axis.append(np.clip(base - e, 0.0, 1.0))
        if axis:
            seeds.append(np.asarray(axis, dtype=np.float32))

        if init_x is not None:
            mirror_sets = [(1,), (2,), (3,), (1, 2, 3)]
            mirrors = []
            for dims in mirror_sets:
                m = base.copy()
                for dim in dims:
                    m[dim] = 1.0 - m[dim]
                mirrors.append(np.clip(m, 0.0, 1.0))
            seeds.append(np.asarray(mirrors, dtype=np.float32))

        seeds.append(rng.uniform(0.0, 1.0, size=(num_uniform, 4)).astype(np.float32))

        local_base = base[None, :]
        sigma_narrow = np.asarray([0.10, 0.10, 0.16, 0.10], dtype=np.float32)
        sigma_wide = np.asarray([0.18, 0.16, 0.24, 0.14], dtype=np.float32)
        local_narrow = local_base + rng.normal(size=(num_local_narrow, 4)).astype(np.float32) * sigma_narrow[None, :]
        local_wide = local_base + rng.normal(size=(num_local_wide, 4)).astype(np.float32) * sigma_wide[None, :]
        seeds.append(np.clip(local_narrow, 0.0, 1.0).astype(np.float32))
        seeds.append(np.clip(local_wide, 0.0, 1.0).astype(np.float32))

        X01 = np.concatenate(seeds, axis=0)
        X01 = np.unique(np.round(X01, 6), axis=0).astype(np.float32)
        return X01

    def _make_seed_bank_rescue(
        seed: int,
        init_x: Optional[np.ndarray],
        num_uniform: int = 256,
        num_local_narrow: int = 160,
        num_local_wide: int = 160,
        num_local_ultrawide: int = 96,
        axis_step: float = 0.16,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        center = np.full((1, 4), 0.5, dtype=np.float32)
        base = center[0]
        seeds = [center]
        if init_x is not None:
            base = _x01_from_phys_np(init_x)
            seeds.append(base[None, :])

        axis = []
        for dim in range(4):
            e = np.zeros((4,), dtype=np.float32)
            e[dim] = axis_step
            axis.append(np.clip(base + e, 0.0, 1.0))
            axis.append(np.clip(base - e, 0.0, 1.0))
        if axis:
            seeds.append(np.asarray(axis, dtype=np.float32))

        if init_x is not None:
            mirror_sets = [
                (1,),
                (2,),
                (3,),
                (1, 2),
                (1, 3),
                (2, 3),
                (1, 2, 3),
            ]
            mirrors = []
            for dims in mirror_sets:
                m = base.copy()
                for dim in dims:
                    m[dim] = 1.0 - m[dim]
                mirrors.append(np.clip(m, 0.0, 1.0))
            seeds.append(np.asarray(mirrors, dtype=np.float32))

        seeds.append(rng.uniform(0.0, 1.0, size=(num_uniform, 4)).astype(np.float32))

        local_base = base[None, :]
        sigma_narrow = np.asarray([0.08, 0.08, 0.12, 0.08], dtype=np.float32)
        sigma_wide = np.asarray([0.16, 0.14, 0.22, 0.14], dtype=np.float32)
        sigma_ultrawide = np.asarray([0.24, 0.20, 0.30, 0.18], dtype=np.float32)
        seeds.append(np.clip(local_base + rng.normal(size=(num_local_narrow, 4)).astype(np.float32) * sigma_narrow[None, :], 0.0, 1.0).astype(np.float32))
        seeds.append(np.clip(local_base + rng.normal(size=(num_local_wide, 4)).astype(np.float32) * sigma_wide[None, :], 0.0, 1.0).astype(np.float32))
        seeds.append(np.clip(local_base + rng.normal(size=(num_local_ultrawide, 4)).astype(np.float32) * sigma_ultrawide[None, :], 0.0, 1.0).astype(np.float32))

        X01 = np.concatenate(seeds, axis=0)
        X01 = np.unique(np.round(X01, 6), axis=0).astype(np.float32)
        return X01

    def _make_seed_bank_refine_local(
        seed: int,
        init_x: Optional[np.ndarray],
        num_tiny: int = 384,
        num_small: int = 384,
        num_medium: int = 256,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        center = np.full((1, 4), 0.5, dtype=np.float32)
        base = center[0]
        seeds = [center]
        if init_x is not None:
            base = _x01_from_phys_np(init_x)
            seeds.append(base[None, :])

        axis = []
        for step in (0.01, 0.02, 0.04):
            for dim in range(4):
                e = np.zeros((4,), dtype=np.float32)
                e[dim] = step
                axis.append(np.clip(base + e, 0.0, 1.0))
                axis.append(np.clip(base - e, 0.0, 1.0))
        if axis:
            seeds.append(np.asarray(axis, dtype=np.float32))

        local_base = base[None, :]
        sigma_tiny = np.asarray([0.012, 0.012, 0.018, 0.012], dtype=np.float32)
        sigma_small = np.asarray([0.025, 0.020, 0.035, 0.020], dtype=np.float32)
        sigma_medium = np.asarray([0.050, 0.040, 0.060, 0.030], dtype=np.float32)
        seeds.append(np.clip(local_base + rng.normal(size=(num_tiny, 4)).astype(np.float32) * sigma_tiny[None, :], 0.0, 1.0).astype(np.float32))
        seeds.append(np.clip(local_base + rng.normal(size=(num_small, 4)).astype(np.float32) * sigma_small[None, :], 0.0, 1.0).astype(np.float32))
        seeds.append(np.clip(local_base + rng.normal(size=(num_medium, 4)).astype(np.float32) * sigma_medium[None, :], 0.0, 1.0).astype(np.float32))

        X01 = np.concatenate(seeds, axis=0)
        X01 = np.unique(np.round(X01, 6), axis=0).astype(np.float32)
        return X01

    def _nearest_sqdist(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
        if points_a.size == 0:
            return np.empty((0,), dtype=np.float32)
        if points_b.size == 0:
            return np.full((points_a.shape[0],), np.inf, dtype=np.float32)
        diff = points_a[:, None, :] - points_b[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        return np.min(d2, axis=1).astype(np.float32)

    def _make_transition_seed_bank_targeted(
        seed: int,
        init_x: Optional[np.ndarray],
        prev_slots: np.ndarray,
        prev_slot_mask: np.ndarray,
        next_pts_inb: np.ndarray,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        prev_pts = np.asarray(prev_slots[prev_slot_mask], dtype=np.float32)
        if prev_pts.size:
            prev_pts = prev_pts[_in_bounds_mask_np(prev_pts)]
        next_pts = np.asarray(next_pts_inb, dtype=np.float32)

        prev_d2 = _nearest_sqdist(prev_pts, next_pts)
        next_d2 = _nearest_sqdist(next_pts, prev_pts)
        change_thresh = float(0.12 ** 2)

        changed_prev = prev_pts[prev_d2 > change_thresh] if prev_pts.size else np.empty((0, 2), dtype=np.float32)
        changed_next = next_pts[next_d2 > change_thresh] if next_pts.size else np.empty((0, 2), dtype=np.float32)

        if changed_prev.size:
            prev_order = np.argsort(-prev_d2[prev_d2 > change_thresh])
            hit_pts = changed_prev[prev_order][:4]
        else:
            hit_pts = prev_pts[np.argsort(np.sum(prev_pts * prev_pts, axis=1))[:3]] if prev_pts.size else np.empty((0, 2), dtype=np.float32)

        if changed_next.size:
            next_order = np.argsort(-next_d2[next_d2 > change_thresh])
            draw_pts = changed_next[next_order][:4]
        else:
            draw_pts = next_pts[np.argsort(np.sum(next_pts * next_pts, axis=1))[:4]] if next_pts.size else np.empty((0, 2), dtype=np.float32)

        takeout_like = int(next_pts.shape[0]) <= int(prev_pts.shape[0])
        seeds_phys: List[np.ndarray] = []

        def _emit_line_seeds(
            target_pts: np.ndarray,
            speed_vals: np.ndarray,
            spin_vals: np.ndarray,
            y_offsets: Tuple[float, ...],
            angle_offsets: Tuple[float, ...],
        ):
            for x_t, y_t in target_pts.tolist():
                for y_off in y_offsets:
                    y0 = float(np.clip(y_t + y_off, bounds.y0_min, bounds.y0_max))
                    base_angle = float(np.arctan2(y_t - y0, x_t + p_refine.hog_to_tee))
                    for ang_off in angle_offsets:
                        ang = float(np.clip(base_angle + ang_off, bounds.angle_min, bounds.angle_max))
                        for speed in speed_vals.tolist():
                            for spin in spin_vals.tolist():
                                seeds_phys.append(
                                    np.array(
                                        [
                                            float(np.clip(speed, bounds.speed_min, bounds.speed_max)),
                                            ang,
                                            float(np.clip(spin, bounds.spin_min, bounds.spin_max)),
                                            y0,
                                        ],
                                        dtype=np.float32,
                                    )
                                )

        if draw_pts.size:
            draw_speeds = np.asarray([0.65, 0.95, 1.25, 1.55] if not takeout_like else [1.00, 1.35, 1.70], dtype=np.float32)
            draw_spins = np.asarray([-1.8, -0.8, 0.8, 1.8], dtype=np.float32)
            _emit_line_seeds(
                draw_pts,
                draw_speeds,
                draw_spins,
                y_offsets=(0.0, -0.08, 0.08),
                angle_offsets=(0.0, -0.018, 0.018),
            )

        if hit_pts.size:
            hit_speeds = np.asarray([1.40, 1.80, 2.20, 2.60, 2.95], dtype=np.float32)
            hit_spins = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)
            _emit_line_seeds(
                hit_pts,
                hit_speeds,
                hit_spins,
                y_offsets=(0.0, -0.05, 0.05),
                angle_offsets=(0.0, -0.012, 0.012),
            )

        if init_x is not None:
            base = np.clip(np.asarray(init_x, dtype=np.float32).reshape(-1), lo, hi)
            local_sigma = np.asarray([0.08, 0.020, 0.25, 0.020], dtype=np.float32)
            local = base[None, :] + rng.normal(size=(96, 4)).astype(np.float32) * local_sigma[None, :]
            seeds_phys.append(np.clip(local, lo, hi).astype(np.float32))

        if not seeds_phys:
            return np.empty((0, 4), dtype=np.float32)

        X_phys = np.concatenate(
            [arr if arr.ndim == 2 else arr[None, :] for arr in seeds_phys],
            axis=0,
        ).astype(np.float32)
        X01 = _x01_from_phys_np(X_phys)
        X01 = np.unique(np.round(X01, 6), axis=0).astype(np.float32)
        return X01

    def _coordinate_pattern_polish(
        x_init_phys: np.ndarray,
        eval_hard_np,
        max_passes: int = 2,
    ) -> Tuple[np.ndarray, float]:
        best = np.clip(np.asarray(x_init_phys, dtype=np.float32).reshape(-1), lo, hi).astype(np.float32)
        best_h = float(eval_hard_np(batched_hard_refine, best)[0])

        step_schedule = [
            np.asarray([0.12, 0.025, 0.40, 0.030], dtype=np.float32),
            np.asarray([0.06, 0.012, 0.20, 0.015], dtype=np.float32),
            np.asarray([0.03, 0.006, 0.10, 0.008], dtype=np.float32),
            np.asarray([0.015, 0.003, 0.05, 0.004], dtype=np.float32),
        ]
        pair_dims = [(1, 3), (0, 1), (0, 2), (2, 3)]

        for steps in step_schedule:
            improved = True
            pass_count = 0
            while improved and pass_count < max_passes:
                improved = False
                pass_count += 1
                cand_list = [best]
                for dim, step in enumerate(steps.tolist()):
                    for sign in (-1.0, 1.0):
                        cand = best.copy()
                        cand[dim] = float(np.clip(cand[dim] + sign * step, lo[dim], hi[dim]))
                        cand_list.append(cand)
                for d0, d1 in pair_dims:
                    for s0 in (-1.0, 1.0):
                        for s1 in (-1.0, 1.0):
                            cand = best.copy()
                            cand[d0] = float(np.clip(cand[d0] + s0 * steps[d0], lo[d0], hi[d0]))
                            cand[d1] = float(np.clip(cand[d1] + s1 * steps[d1], lo[d1], hi[d1]))
                            cand_list.append(cand)
                cand_batch = np.unique(np.round(np.asarray(cand_list, dtype=np.float32), 6), axis=0).astype(np.float32)
                hard_batch = eval_hard_np(batched_hard_refine, cand_batch)
                idx = int(np.argmin(hard_batch))
                cand_h = float(hard_batch[idx])
                if cand_h + 1e-7 < best_h:
                    best_h = cand_h
                    best = cand_batch[idx].astype(np.float32)
                    improved = True

        return best, best_h

    def _van_der_corput_sequence(n: int, base: int, start_index: int) -> np.ndarray:
        out = np.empty((n,), dtype=np.float32)
        for i in range(n):
            idx = int(start_index + i)
            denom = 1.0
            value = 0.0
            while idx > 0:
                idx, rem = divmod(idx, base)
                denom /= float(base)
                value += float(rem) * denom
            out[i] = value
        return out

    def _halton_sequence(n: int, dim: int, start_index: int) -> np.ndarray:
        primes = [2, 3, 5, 7, 11, 13, 17, 19]
        cols = [_van_der_corput_sequence(n, primes[d], start_index=start_index) for d in range(dim)]
        return np.stack(cols, axis=1).astype(np.float32)

    def _make_seed_bank_halton_refine(
        seed: int,
        init_x: Optional[np.ndarray],
        prev_slots: np.ndarray,
        prev_slot_mask: np.ndarray,
        next_pts_inb: np.ndarray,
        num_halton_global: int = 1536,
        num_halton_local: int = 768,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        banks = [_make_seed_bank_large(seed=seed, init_x=init_x)]

        targeted_bank = _make_transition_seed_bank_targeted(
            seed=seed + 7,
            init_x=init_x,
            prev_slots=prev_slots,
            prev_slot_mask=prev_slot_mask,
            next_pts_inb=next_pts_inb,
        )
        if targeted_bank.size:
            banks.append(targeted_bank)

        if init_x is not None:
            banks.append(_make_seed_bank_refine_local(seed=seed + 13, init_x=init_x))

        start_index = 1 + (abs(int(seed)) % 997)
        halton_global = _halton_sequence(num_halton_global, 4, start_index=start_index)
        banks.append(halton_global)

        if init_x is not None:
            base = _x01_from_phys_np(init_x).reshape(1, 4)
            halton_local = _halton_sequence(num_halton_local, 4, start_index=start_index + num_halton_global + 17)
            local_centered = (halton_local * 2.0) - 1.0
            sigma_small = np.asarray([0.08, 0.035, 0.10, 0.030], dtype=np.float32)
            sigma_large = np.asarray([0.16, 0.060, 0.20, 0.045], dtype=np.float32)
            local_small = np.clip(base + local_centered * sigma_small[None, :], 0.0, 1.0).astype(np.float32)
            local_large = np.clip(base + local_centered * sigma_large[None, :], 0.0, 1.0).astype(np.float32)
            banks.append(local_small)
            banks.append(local_large)

            mirror_sets = [
                (1,),
                (2,),
                (3,),
                (1, 2),
                (1, 3),
                (2, 3),
                (1, 2, 3),
            ]
            mirrors = []
            base_vec = base[0]
            for dims in mirror_sets:
                m = base_vec.copy()
                for dim in dims:
                    m[dim] = 1.0 - m[dim]
                mirrors.append(np.clip(m, 0.0, 1.0))
            if mirrors:
                banks.append(np.asarray(mirrors, dtype=np.float32))
        else:
            jitter = rng.uniform(-0.08, 0.08, size=(192, 4)).astype(np.float32)
            banks.append(np.clip(0.5 + jitter, 0.0, 1.0).astype(np.float32))

        X01 = np.concatenate(banks, axis=0)
        X01 = np.unique(np.round(X01, 6), axis=0).astype(np.float32)
        return X01

    def _nelder_mead_polish(
        x_init_phys: np.ndarray,
        eval_hard_np,
        batched_hard_fn,
        max_iters: int = 28,
    ) -> Tuple[np.ndarray, float]:
        x0 = np.clip(np.asarray(x_init_phys, dtype=np.float32).reshape(-1), lo, hi).astype(np.float32)
        steps = np.asarray([0.050, 0.010, 0.16, 0.012], dtype=np.float32)
        simplex = [x0]
        for dim in range(4):
            cand = x0.copy()
            cand[dim] = float(np.clip(cand[dim] + steps[dim], lo[dim], hi[dim]))
            if abs(float(cand[dim]) - float(x0[dim])) < 1e-7:
                cand[dim] = float(np.clip(cand[dim] - 2.0 * steps[dim], lo[dim], hi[dim]))
            simplex.append(cand.astype(np.float32))
        simplex = np.asarray(simplex, dtype=np.float32)
        losses = eval_hard_np(batched_hard_fn, simplex)

        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        for _ in range(max_iters):
            order = np.argsort(losses)
            simplex = simplex[order]
            losses = losses[order]
            best = simplex[0].copy()
            best_h = float(losses[0])
            worst = simplex[-1].copy()
            centroid = np.mean(simplex[:-1], axis=0).astype(np.float32)

            reflect = np.clip(centroid + alpha * (centroid - worst), lo, hi).astype(np.float32)
            cand_batch = [reflect]
            expand = np.clip(centroid + gamma * (reflect - centroid), lo, hi).astype(np.float32)
            contract_out = np.clip(centroid + rho * (reflect - centroid), lo, hi).astype(np.float32)
            contract_in = np.clip(centroid - rho * (reflect - centroid), lo, hi).astype(np.float32)
            cand_batch.extend([expand, contract_out, contract_in])
            cand_batch = np.asarray(cand_batch, dtype=np.float32)
            cand_losses = eval_hard_np(batched_hard_fn, cand_batch)

            reflect_h = float(cand_losses[0])
            expand_h = float(cand_losses[1])
            contract_out_h = float(cand_losses[2])
            contract_in_h = float(cand_losses[3])

            if reflect_h < float(losses[0]):
                if expand_h < reflect_h:
                    simplex[-1] = expand
                    losses[-1] = expand_h
                else:
                    simplex[-1] = reflect
                    losses[-1] = reflect_h
                continue

            if reflect_h < float(losses[-2]):
                simplex[-1] = reflect
                losses[-1] = reflect_h
                continue

            if reflect_h < float(losses[-1]):
                if contract_out_h <= reflect_h:
                    simplex[-1] = contract_out
                    losses[-1] = contract_out_h
                    continue
            else:
                if contract_in_h < float(losses[-1]):
                    simplex[-1] = contract_in
                    losses[-1] = contract_in_h
                    continue

            shrink = [best]
            for i in range(1, simplex.shape[0]):
                shrunk = np.clip(best + sigma * (simplex[i] - best), lo, hi).astype(np.float32)
                shrink.append(shrunk)
            simplex = np.asarray(shrink, dtype=np.float32)
            losses = eval_hard_np(batched_hard_fn, simplex)
            if np.max(np.linalg.norm(simplex - simplex[0], axis=1)) < 5e-4 and (np.max(losses) - np.min(losses)) < 1e-5:
                break

        idx = int(np.argmin(losses))
        return simplex[idx].astype(np.float32), float(losses[idx])

    def _prepare_objective(
        prev_slots: np.ndarray,
        prev_slot_mask: np.ndarray,
        thrower_block: int,
        target_block0_full: np.ndarray,
        target_block0_mask: np.ndarray,
        target_block1_full: np.ndarray,
        target_block1_mask: np.ndarray,
    ):
        target_block0_mask_inb = target_block0_mask & _in_bounds_mask_np(target_block0_full)
        target_block1_mask_inb = target_block1_mask & _in_bounds_mask_np(target_block1_full)
        target_block0_inb = target_block0_full[target_block0_mask_inb]
        target_block1_inb = target_block1_full[target_block1_mask_inb]
        next_pts_inb = np.concatenate([target_block0_inb, target_block1_inb], axis=0)

        prev_j = jnp.asarray(prev_slots, dtype=jnp.float32)
        prev_slot_mask_j = jnp.asarray(prev_slot_mask, dtype=jnp.bool_)
        thrower_block_j = jnp.asarray(int(thrower_block), dtype=jnp.int32)
        target_block0_j = jnp.asarray(target_block0_full, dtype=jnp.float32)
        target_block1_j = jnp.asarray(target_block1_full, dtype=jnp.float32)
        target_block0_mask_j = jnp.asarray(target_block0_mask_inb, dtype=jnp.bool_)
        target_block1_mask_j = jnp.asarray(target_block1_mask_inb, dtype=jnp.bool_)

        def eval_hard_np(batched_hard_fn, x_phys_batch: np.ndarray) -> np.ndarray:
            arr = np.asarray(x_phys_batch, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[None, :]
            if arr.shape[0] == 0:
                return np.empty((0,), dtype=np.float32)
            out = batched_hard_fn(
                prev_j,
                prev_slot_mask_j,
                thrower_block_j,
                target_block0_j,
                target_block0_mask_j,
                target_block1_j,
                target_block1_mask_j,
                jnp.asarray(arr, dtype=jnp.float32),
            )
            return np.asarray(out, dtype=np.float32)

        return (
            next_pts_inb,
            prev_j,
            prev_slot_mask_j,
            thrower_block_j,
            target_block0_j,
            target_block0_mask_j,
            target_block1_j,
            target_block1_mask_j,
            eval_hard_np,
        )

    def run_two_stage_cem(prev_slots: np.ndarray,
                          prev_slot_mask: np.ndarray,
                          thrower_block: int,
                          target_block0_full: np.ndarray,
                          target_block0_mask: np.ndarray,
                          target_block1_full: np.ndarray,
                          target_block1_mask: np.ndarray,
                          warm_start_x: Optional[np.ndarray] = None,
                          seed: int = 0):
        """Run the same coarse + refine CEM with block-aware matching."""
        warm_start_coarse_threshold = 0.5
        (
            next_pts_inb,
            prev_j,
            prev_slot_mask_j,
            thrower_block_j,
            target_block0_j,
            target_block0_mask_j,
            target_block1_j,
            target_block1_mask_j,
            eval_hard_np,
        ) = _prepare_objective(
            prev_slots,
            prev_slot_mask,
            thrower_block,
            target_block0_full,
            target_block0_mask,
            target_block1_full,
            target_block1_mask,
        )

        warm_init = None
        warm_init_hard = np.nan
        used_coarse_fallback = False
        if warm_start_x is not None:
            warm_arr = np.asarray(warm_start_x, dtype=np.float32).reshape(-1)
            if warm_arr.shape[0] == 4 and np.all(np.isfinite(warm_arr)):
                warm_init = np.clip(warm_arr, lo, hi).astype(np.float32)
                warm_init_hard = float(eval_hard_np(batched_hard_refine, warm_init)[0])

        if warm_init is not None and warm_init_hard <= warm_start_coarse_threshold:
            x, hard = solve_inverse_by_block(
                p_refine,
                prev_j,
                prev_slot_mask_j,
                thrower_block_j,
                target_block0_j,
                target_block0_mask_j,
                target_block1_j,
                target_block1_mask_j,
                bounds=bounds,
                pop_size=800,
                generations=80,
                elite_frac=0.30,
                sigma_init=0.10,
                sigma_floor=0.005,
                ema_alpha=0.75,
                use_full_cov=False,
                mix_with_best_frac=0.40,
                jitter_anchor=0.0015,
                key=jax.random.PRNGKey(seed + 1),
                init_x=warm_init,
                loss_threshold=0.10,
                log_topk=3,
                eval_chunk_size=400,
                batched_hard_fn=batched_hard_refine,
            )
            return x, float(hard), np.nan, next_pts_inb, float(warm_init_hard), used_coarse_fallback

        if warm_init is not None:
            used_coarse_fallback = True

        # Stage A (coarse)
        x0, hard0 = solve_inverse_by_block(
            p_coarse,
            prev_j,
            prev_slot_mask_j,
            thrower_block_j,
            target_block0_j,
            target_block0_mask_j,
            target_block1_j,
            target_block1_mask_j,
            bounds=bounds,
            pop_size=800,
            generations=25,
            elite_frac=0.20,
            sigma_init=0.25,
            sigma_floor=0.01,
            ema_alpha=0.7,
            use_full_cov=False,
            mix_with_best_frac=0.35,
            jitter_anchor=0.006,
            key=jax.random.PRNGKey(seed),
            init_x=warm_init,
            loss_threshold=0.5,
            log_topk=3,
            eval_chunk_size=400,
            batched_hard_fn=batched_hard_coarse,
        )

        # Stage B (refine)
        x, hard = solve_inverse_by_block(
            p_refine,
            prev_j,
            prev_slot_mask_j,
            thrower_block_j,
            target_block0_j,
            target_block0_mask_j,
            target_block1_j,
            target_block1_mask_j,
            bounds=bounds,
            pop_size=800,
            generations=80,
            elite_frac=0.30,
            sigma_init=0.10,
            sigma_floor=0.005,
            ema_alpha=0.75,
            use_full_cov=False,
            mix_with_best_frac=0.40,
            jitter_anchor=0.0015,
            key=jax.random.PRNGKey(seed + 1),
            init_x=x0,
            loss_threshold=0.10,
            log_topk=3,
            eval_chunk_size=400,
            batched_hard_fn=batched_hard_refine,
        )

        return x, float(hard), float(hard0), next_pts_inb, float(warm_init_hard), used_coarse_fallback

    def run_portfolio_cem(prev_slots: np.ndarray,
                          prev_slot_mask: np.ndarray,
                          thrower_block: int,
                          target_block0_full: np.ndarray,
                          target_block0_mask: np.ndarray,
                          target_block1_full: np.ndarray,
                          target_block1_mask: np.ndarray,
                          warm_start_x: Optional[np.ndarray] = None,
                          seed: int = 0):
        warm_accept_threshold = 0.10
        warm_global_threshold = 0.50
        (
            next_pts_inb,
            prev_j,
            prev_slot_mask_j,
            thrower_block_j,
            target_block0_j,
            target_block0_mask_j,
            target_block1_j,
            target_block1_mask_j,
            eval_hard_np,
        ) = _prepare_objective(
            prev_slots,
            prev_slot_mask,
            thrower_block,
            target_block0_full,
            target_block0_mask,
            target_block1_full,
            target_block1_mask,
        )

        warm_init = None
        warm_init_hard = np.nan
        used_coarse_fallback = False
        if warm_start_x is not None:
            warm_arr = np.asarray(warm_start_x, dtype=np.float32).reshape(-1)
            if warm_arr.shape[0] == 4 and np.all(np.isfinite(warm_arr)):
                warm_init = np.clip(warm_arr, lo, hi).astype(np.float32)
                warm_init_hard = float(eval_hard_np(batched_hard_refine, warm_init)[0])
                if warm_init_hard > warm_global_threshold:
                    used_coarse_fallback = True
                if warm_init_hard <= warm_accept_threshold:
                    return warm_init, float(warm_init_hard), np.nan, next_pts_inb, float(warm_init_hard), used_coarse_fallback

        seed_bank_x01 = _make_seed_bank(seed=seed, init_x=warm_init)
        seed_bank_phys = _xphys_from_x01_np(seed_bank_x01)
        seed_bank_losses = eval_hard_np(batched_hard_coarse, seed_bank_phys)
        restart_seed_idx = _pick_distinct_indices(
            seed_bank_x01,
            seed_bank_losses,
            max_keep=4,
            min_dist=0.16,
        )

        short_results = []
        for restart_idx, seed_idx in enumerate(restart_seed_idx):
            seed_x01 = seed_bank_x01[seed_idx]
            seed_phys = _xphys_from_x01_np(seed_x01)
            x_short, hard_short = solve_inverse_by_block(
                p_coarse,
                prev_j,
                prev_slot_mask_j,
                thrower_block_j,
                target_block0_j,
                target_block0_mask_j,
                target_block1_j,
                target_block1_mask_j,
                bounds=bounds,
                pop_size=224,
                generations=8,
                elite_frac=0.25,
                sigma_init=0.16,
                sigma_floor=0.01,
                ema_alpha=0.68,
                use_full_cov=False,
                mix_with_best_frac=0.18,
                jitter_anchor=0.008,
                key=jax.random.PRNGKey(seed + 10 + restart_idx),
                init_x=seed_phys,
                loss_threshold=0.25,
                log_topk=2,
                eval_chunk_size=400,
                batched_hard_fn=batched_hard_coarse,
            )
            short_results.append(
                {
                    "x_phys": np.asarray(x_short, dtype=np.float32),
                    "x01": _x01_from_phys_np(np.asarray(x_short, dtype=np.float32)),
                    "hard": float(hard_short),
                }
            )

        if warm_init is not None:
            warm_init_coarse = float(eval_hard_np(batched_hard_coarse, warm_init)[0])
            short_results.append(
                {
                    "x_phys": warm_init.astype(np.float32),
                    "x01": _x01_from_phys_np(warm_init),
                    "hard": warm_init_coarse,
                }
            )

        short_X01 = np.asarray([r["x01"] for r in short_results], dtype=np.float32)
        short_losses = np.asarray([r["hard"] for r in short_results], dtype=np.float32)
        finisher_idx = _pick_distinct_indices(
            short_X01,
            short_losses,
            max_keep=2,
            min_dist=0.10,
        )

        best_phys = None
        best_hard = np.inf
        proposal_hard = float(np.min(short_losses)) if len(short_losses) else np.nan
        for fin_idx, idx in enumerate(finisher_idx):
            x_init = short_results[idx]["x_phys"]
            x_fin, hard_fin = solve_inverse_by_block(
                p_refine,
                prev_j,
                prev_slot_mask_j,
                thrower_block_j,
                target_block0_j,
                target_block0_mask_j,
                target_block1_j,
                target_block1_mask_j,
                bounds=bounds,
                pop_size=256,
                generations=20,
                elite_frac=0.24,
                sigma_init=0.07,
                sigma_floor=0.003,
                ema_alpha=0.72,
                use_full_cov=True,
                mix_with_best_frac=0.20,
                jitter_anchor=0.0015,
                key=jax.random.PRNGKey(seed + 100 + fin_idx),
                init_x=x_init,
                loss_threshold=0.10,
                log_topk=2,
                eval_chunk_size=400,
                batched_hard_fn=batched_hard_refine,
            )
            hard_fin = float(hard_fin)
            if hard_fin < best_hard:
                best_hard = hard_fin
                best_phys = np.asarray(x_fin, dtype=np.float32)

        if best_phys is None:
            best_row = short_results[int(np.argmin(short_losses))]
            best_phys = best_row["x_phys"]
            best_hard = float(eval_hard_np(batched_hard_refine, best_phys)[0])

        return best_phys, float(best_hard), proposal_hard, next_pts_inb, float(warm_init_hard), used_coarse_fallback

    def run_portfolio_large_cem(prev_slots: np.ndarray,
                                prev_slot_mask: np.ndarray,
                                thrower_block: int,
                                target_block0_full: np.ndarray,
                                target_block0_mask: np.ndarray,
                                target_block1_full: np.ndarray,
                                target_block1_mask: np.ndarray,
                                warm_start_x: Optional[np.ndarray] = None,
                                seed: int = 0):
        warm_accept_threshold = 0.10
        warm_global_threshold = 0.50
        (
            next_pts_inb,
            prev_j,
            prev_slot_mask_j,
            thrower_block_j,
            target_block0_j,
            target_block0_mask_j,
            target_block1_j,
            target_block1_mask_j,
            eval_hard_np,
        ) = _prepare_objective(
            prev_slots,
            prev_slot_mask,
            thrower_block,
            target_block0_full,
            target_block0_mask,
            target_block1_full,
            target_block1_mask,
        )

        warm_init = None
        warm_init_hard = np.nan
        used_coarse_fallback = False
        if warm_start_x is not None:
            warm_arr = np.asarray(warm_start_x, dtype=np.float32).reshape(-1)
            if warm_arr.shape[0] == 4 and np.all(np.isfinite(warm_arr)):
                warm_init = np.clip(warm_arr, lo, hi).astype(np.float32)
                warm_init_hard = float(eval_hard_np(batched_hard_refine, warm_init)[0])
                if warm_init_hard > warm_global_threshold:
                    used_coarse_fallback = True
                if warm_init_hard <= warm_accept_threshold:
                    return warm_init, float(warm_init_hard), np.nan, next_pts_inb, float(warm_init_hard), used_coarse_fallback

        seed_bank_x01 = _make_seed_bank_large(seed=seed, init_x=warm_init)
        seed_bank_phys = _xphys_from_x01_np(seed_bank_x01)
        seed_bank_losses = eval_hard_np(batched_hard_coarse, seed_bank_phys)
        restart_seed_idx = _pick_distinct_indices(
            seed_bank_x01,
            seed_bank_losses,
            max_keep=4,
            min_dist=0.16,
        )

        scout_results = []
        for restart_idx, seed_idx in enumerate(restart_seed_idx):
            seed_x01 = seed_bank_x01[seed_idx]
            seed_phys = _xphys_from_x01_np(seed_x01)
            x_scout, hard_scout = solve_inverse_by_block(
                p_coarse,
                prev_j,
                prev_slot_mask_j,
                thrower_block_j,
                target_block0_j,
                target_block0_mask_j,
                target_block1_j,
                target_block1_mask_j,
                bounds=bounds,
                pop_size=384,
                generations=12,
                elite_frac=0.24,
                sigma_init=0.17,
                sigma_floor=0.01,
                ema_alpha=0.68,
                use_full_cov=False,
                mix_with_best_frac=0.16,
                jitter_anchor=0.008,
                key=jax.random.PRNGKey(seed + 20 + restart_idx),
                init_x=seed_phys,
                loss_threshold=0.25,
                log_topk=2,
                eval_chunk_size=384,
                batched_hard_fn=batched_hard_coarse,
            )
            scout_results.append(
                {
                    "x_phys": np.asarray(x_scout, dtype=np.float32),
                    "x01": _x01_from_phys_np(np.asarray(x_scout, dtype=np.float32)),
                    "coarse_hard": float(hard_scout),
                }
            )

        shortlist = list(scout_results)
        if warm_init is not None:
            shortlist.append(
                {
                    "x_phys": warm_init.astype(np.float32),
                    "x01": _x01_from_phys_np(warm_init),
                    "coarse_hard": float(eval_hard_np(batched_hard_coarse, warm_init)[0]),
                }
            )

        shortlist_phys = np.asarray([r["x_phys"] for r in shortlist], dtype=np.float32)
        shortlist_refine_losses = eval_hard_np(batched_hard_refine, shortlist_phys)
        for idx, refine_hard in enumerate(shortlist_refine_losses.tolist()):
            shortlist[idx]["refine_hard"] = float(refine_hard)

        shortlist_x01 = np.asarray([r["x01"] for r in shortlist], dtype=np.float32)
        finisher_idx = _pick_distinct_indices(
            shortlist_x01,
            np.asarray([r["refine_hard"] for r in shortlist], dtype=np.float32),
            max_keep=2,
            min_dist=0.10,
        )

        best_phys = None
        best_hard = np.inf
        proposal_hard = float(np.min([r["coarse_hard"] for r in shortlist])) if shortlist else np.nan
        for fin_idx, idx in enumerate(finisher_idx):
            x_init = shortlist[idx]["x_phys"]
            x_fin, hard_fin = solve_inverse_by_block(
                p_refine,
                prev_j,
                prev_slot_mask_j,
                thrower_block_j,
                target_block0_j,
                target_block0_mask_j,
                target_block1_j,
                target_block1_mask_j,
                bounds=bounds,
                pop_size=512,
                generations=32,
                elite_frac=0.22,
                sigma_init=0.07,
                sigma_floor=0.003,
                ema_alpha=0.72,
                use_full_cov=True,
                mix_with_best_frac=0.08,
                jitter_anchor=0.0015,
                key=jax.random.PRNGKey(seed + 200 + fin_idx),
                init_x=x_init,
                loss_threshold=0.10,
                log_topk=2,
                eval_chunk_size=400,
                batched_hard_fn=batched_hard_refine,
            )
            hard_fin = float(hard_fin)
            if hard_fin < best_hard:
                best_hard = hard_fin
                best_phys = np.asarray(x_fin, dtype=np.float32)

        if best_phys is None:
            best_idx = int(np.argmin(np.asarray([r["refine_hard"] for r in shortlist], dtype=np.float32)))
            best_phys = shortlist[best_idx]["x_phys"]
            best_hard = float(shortlist[best_idx]["refine_hard"])

        return best_phys, float(best_hard), proposal_hard, next_pts_inb, float(warm_init_hard), used_coarse_fallback

    def run_portfolio_large_targeted_cem(prev_slots: np.ndarray,
                                         prev_slot_mask: np.ndarray,
                                         thrower_block: int,
                                         target_block0_full: np.ndarray,
                                         target_block0_mask: np.ndarray,
                                         target_block1_full: np.ndarray,
                                         target_block1_mask: np.ndarray,
                                         warm_start_x: Optional[np.ndarray] = None,
                                         seed: int = 0):
        warm_accept_threshold = 0.10
        warm_global_threshold = 0.50
        (
            next_pts_inb,
            prev_j,
            prev_slot_mask_j,
            thrower_block_j,
            target_block0_j,
            target_block0_mask_j,
            target_block1_j,
            target_block1_mask_j,
            eval_hard_np,
        ) = _prepare_objective(
            prev_slots,
            prev_slot_mask,
            thrower_block,
            target_block0_full,
            target_block0_mask,
            target_block1_full,
            target_block1_mask,
        )

        warm_init = None
        warm_init_hard = np.nan
        used_coarse_fallback = False
        if warm_start_x is not None:
            warm_arr = np.asarray(warm_start_x, dtype=np.float32).reshape(-1)
            if warm_arr.shape[0] == 4 and np.all(np.isfinite(warm_arr)):
                warm_init = np.clip(warm_arr, lo, hi).astype(np.float32)
                warm_init_hard = float(eval_hard_np(batched_hard_refine, warm_init)[0])
                if warm_init_hard > warm_global_threshold:
                    used_coarse_fallback = True
                if warm_init_hard <= warm_accept_threshold:
                    return warm_init, float(warm_init_hard), np.nan, next_pts_inb, float(warm_init_hard), used_coarse_fallback

        generic_bank = _make_seed_bank_large(seed=seed, init_x=warm_init)
        targeted_bank = _make_transition_seed_bank_targeted(
            seed=seed + 17,
            init_x=warm_init,
            prev_slots=prev_slots,
            prev_slot_mask=prev_slot_mask,
            next_pts_inb=next_pts_inb,
        )
        if targeted_bank.size:
            seed_bank_x01 = np.concatenate([generic_bank, targeted_bank], axis=0)
            seed_bank_x01 = np.unique(np.round(seed_bank_x01, 6), axis=0).astype(np.float32)
        else:
            seed_bank_x01 = generic_bank
        seed_bank_phys = _xphys_from_x01_np(seed_bank_x01)
        seed_bank_losses = eval_hard_np(batched_hard_coarse, seed_bank_phys)
        restart_seed_idx = _pick_distinct_indices(
            seed_bank_x01,
            seed_bank_losses,
            max_keep=6,
            min_dist=0.14,
        )

        scout_results = []
        for restart_idx, seed_idx in enumerate(restart_seed_idx):
            seed_x01 = seed_bank_x01[seed_idx]
            seed_phys = _xphys_from_x01_np(seed_x01)
            x_scout, hard_scout = solve_inverse_by_block(
                p_coarse,
                prev_j,
                prev_slot_mask_j,
                thrower_block_j,
                target_block0_j,
                target_block0_mask_j,
                target_block1_j,
                target_block1_mask_j,
                bounds=bounds,
                pop_size=352,
                generations=10,
                elite_frac=0.22,
                sigma_init=0.15,
                sigma_floor=0.01,
                ema_alpha=0.68,
                use_full_cov=False,
                mix_with_best_frac=0.12,
                jitter_anchor=0.007,
                key=jax.random.PRNGKey(seed + 120 + restart_idx),
                init_x=seed_phys,
                loss_threshold=0.25,
                log_topk=2,
                eval_chunk_size=352,
                batched_hard_fn=batched_hard_coarse,
            )
            scout_results.append(
                {
                    "x_phys": np.asarray(x_scout, dtype=np.float32),
                    "x01": _x01_from_phys_np(np.asarray(x_scout, dtype=np.float32)),
                    "coarse_hard": float(hard_scout),
                }
            )

        shortlist = list(scout_results)
        if warm_init is not None:
            shortlist.append(
                {
                    "x_phys": warm_init.astype(np.float32),
                    "x01": _x01_from_phys_np(warm_init),
                    "coarse_hard": float(eval_hard_np(batched_hard_coarse, warm_init)[0]),
                }
            )

        shortlist_phys = np.asarray([r["x_phys"] for r in shortlist], dtype=np.float32)
        shortlist_refine_losses = eval_hard_np(batched_hard_refine, shortlist_phys)
        for idx, refine_hard in enumerate(shortlist_refine_losses.tolist()):
            shortlist[idx]["refine_hard"] = float(refine_hard)

        shortlist_x01 = np.asarray([r["x01"] for r in shortlist], dtype=np.float32)
        finisher_idx = _pick_distinct_indices(
            shortlist_x01,
            np.asarray([r["refine_hard"] for r in shortlist], dtype=np.float32),
            max_keep=3,
            min_dist=0.08,
        )

        best_phys = None
        best_hard = np.inf
        proposal_hard = float(np.min([r["coarse_hard"] for r in shortlist])) if shortlist else np.nan
        for fin_idx, idx in enumerate(finisher_idx):
            x_init = shortlist[idx]["x_phys"]
            x_fin, hard_fin = solve_inverse_by_block(
                p_refine,
                prev_j,
                prev_slot_mask_j,
                thrower_block_j,
                target_block0_j,
                target_block0_mask_j,
                target_block1_j,
                target_block1_mask_j,
                bounds=bounds,
                pop_size=416,
                generations=28,
                elite_frac=0.20,
                sigma_init=0.06,
                sigma_floor=0.003,
                ema_alpha=0.72,
                use_full_cov=True,
                mix_with_best_frac=0.05,
                jitter_anchor=0.0012,
                key=jax.random.PRNGKey(seed + 240 + fin_idx),
                init_x=x_init,
                loss_threshold=0.10,
                log_topk=2,
                eval_chunk_size=416,
                batched_hard_fn=batched_hard_refine,
            )
            hard_fin = float(hard_fin)
            if hard_fin < best_hard:
                best_hard = hard_fin
                best_phys = np.asarray(x_fin, dtype=np.float32)

        if best_phys is None:
            best_idx = int(np.argmin(np.asarray([r["refine_hard"] for r in shortlist], dtype=np.float32)))
            best_phys = shortlist[best_idx]["x_phys"]
            best_hard = float(shortlist[best_idx]["refine_hard"])

        return best_phys, float(best_hard), proposal_hard, next_pts_inb, float(warm_init_hard), used_coarse_fallback

    def run_portfolio_large_coord_cem(prev_slots: np.ndarray,
                                      prev_slot_mask: np.ndarray,
                                      thrower_block: int,
                                      target_block0_full: np.ndarray,
                                      target_block0_mask: np.ndarray,
                                      target_block1_full: np.ndarray,
                                      target_block1_mask: np.ndarray,
                                      warm_start_x: Optional[np.ndarray] = None,
                                      seed: int = 0):
        result = run_portfolio_large_cem(
            prev_slots,
            prev_slot_mask,
            thrower_block,
            target_block0_full,
            target_block0_mask,
            target_block1_full,
            target_block1_mask,
            warm_start_x=warm_start_x,
            seed=seed,
        )
        x_best, hard_refine, hard_coarse, next_inb, warm_init_hard, used_coarse_fallback = result
        (
            _next_pts_inb,
            _prev_j,
            _prev_slot_mask_j,
            _thrower_block_j,
            _target_block0_j,
            _target_block0_mask_j,
            _target_block1_j,
            _target_block1_mask_j,
            eval_hard_np,
        ) = _prepare_objective(
            prev_slots,
            prev_slot_mask,
            thrower_block,
            target_block0_full,
            target_block0_mask,
            target_block1_full,
            target_block1_mask,
        )
        x_polish, hard_polish = _coordinate_pattern_polish(np.asarray(x_best, dtype=np.float32), eval_hard_np=eval_hard_np)
        if float(hard_polish) < float(hard_refine):
            return x_polish, float(hard_polish), hard_coarse, next_inb, warm_init_hard, used_coarse_fallback
        return result

    def run_portfolio_large_targeted_coord_cem(prev_slots: np.ndarray,
                                               prev_slot_mask: np.ndarray,
                                               thrower_block: int,
                                               target_block0_full: np.ndarray,
                                               target_block0_mask: np.ndarray,
                                               target_block1_full: np.ndarray,
                                               target_block1_mask: np.ndarray,
                                               warm_start_x: Optional[np.ndarray] = None,
                                               seed: int = 0):
        result = run_portfolio_large_targeted_cem(
            prev_slots,
            prev_slot_mask,
            thrower_block,
            target_block0_full,
            target_block0_mask,
            target_block1_full,
            target_block1_mask,
            warm_start_x=warm_start_x,
            seed=seed,
        )
        x_best, hard_refine, hard_coarse, next_inb, warm_init_hard, used_coarse_fallback = result
        (
            _next_pts_inb,
            _prev_j,
            _prev_slot_mask_j,
            _thrower_block_j,
            _target_block0_j,
            _target_block0_mask_j,
            _target_block1_j,
            _target_block1_mask_j,
            eval_hard_np,
        ) = _prepare_objective(
            prev_slots,
            prev_slot_mask,
            thrower_block,
            target_block0_full,
            target_block0_mask,
            target_block1_full,
            target_block1_mask,
        )
        x_polish, hard_polish = _coordinate_pattern_polish(np.asarray(x_best, dtype=np.float32), eval_hard_np=eval_hard_np)
        if float(hard_polish) < float(hard_refine):
            return x_polish, float(hard_polish), hard_coarse, next_inb, warm_init_hard, used_coarse_fallback
        return result

    def run_portfolio_large_coord_rescue_cem(prev_slots: np.ndarray,
                                             prev_slot_mask: np.ndarray,
                                             thrower_block: int,
                                             target_block0_full: np.ndarray,
                                             target_block0_mask: np.ndarray,
                                             target_block1_full: np.ndarray,
                                             target_block1_mask: np.ndarray,
                                             warm_start_x: Optional[np.ndarray] = None,
                                             seed: int = 0,
                                             rescue_threshold: float = 0.25):
        coord_result = run_portfolio_large_coord_cem(
            prev_slots,
            prev_slot_mask,
            thrower_block,
            target_block0_full,
            target_block0_mask,
            target_block1_full,
            target_block1_mask,
            warm_start_x=warm_start_x,
            seed=seed,
        )
        x_best, hard_refine, hard_coarse, next_inb, warm_init_hard, used_coarse_fallback = coord_result
        if float(hard_refine) <= float(rescue_threshold):
            return coord_result

        multi_result = run_portfolio_large_multistart_cem(
            prev_slots,
            prev_slot_mask,
            thrower_block,
            target_block0_full,
            target_block0_mask,
            target_block1_full,
            target_block1_mask,
            warm_start_x=warm_start_x,
            seed=seed + 20000,
        )
        x_multi, hard_multi, hard_multi_coarse, next_inb_multi, warm_init_hard_multi, used_coarse_fallback_multi = multi_result
        if float(hard_multi) < float(hard_refine):
            return x_multi, float(hard_multi), hard_multi_coarse, next_inb_multi, warm_init_hard_multi, used_coarse_fallback_multi
        return coord_result

    def run_portfolio_large_refinebank_coord_nm_cem(prev_slots: np.ndarray,
                                                    prev_slot_mask: np.ndarray,
                                                    thrower_block: int,
                                                    target_block0_full: np.ndarray,
                                                    target_block0_mask: np.ndarray,
                                                    target_block1_full: np.ndarray,
                                                    target_block1_mask: np.ndarray,
                                                    warm_start_x: Optional[np.ndarray] = None,
                                                    seed: int = 0):
        warm_accept_threshold = 0.10
        (
            next_pts_inb,
            prev_j,
            prev_slot_mask_j,
            thrower_block_j,
            target_block0_j,
            target_block0_mask_j,
            target_block1_j,
            target_block1_mask_j,
            eval_hard_np,
        ) = _prepare_objective(
            prev_slots,
            prev_slot_mask,
            thrower_block,
            target_block0_full,
            target_block0_mask,
            target_block1_full,
            target_block1_mask,
        )

        warm_init = None
        warm_init_hard = np.nan
        used_coarse_fallback = False
        if warm_start_x is not None:
            warm_arr = np.asarray(warm_start_x, dtype=np.float32).reshape(-1)
            if warm_arr.shape[0] == 4 and np.all(np.isfinite(warm_arr)):
                warm_init = np.clip(warm_arr, lo, hi).astype(np.float32)
                warm_init_hard = float(eval_hard_np(batched_hard_refine, warm_init)[0])
                if warm_init_hard <= warm_accept_threshold:
                    return warm_init, float(warm_init_hard), np.nan, next_pts_inb, float(warm_init_hard), used_coarse_fallback

        seed_bank_x01 = _make_seed_bank_halton_refine(
            seed=seed,
            init_x=warm_init,
            prev_slots=prev_slots,
            prev_slot_mask=prev_slot_mask,
            next_pts_inb=next_pts_inb,
        )
        seed_bank_phys = _xphys_from_x01_np(seed_bank_x01)
        seed_bank_losses = eval_hard_np(batched_hard_refine, seed_bank_phys)
        finisher_seed_idx = _pick_distinct_indices(
            seed_bank_x01,
            seed_bank_losses,
            max_keep=4,
            min_dist=0.10,
        )

        best_idx = int(np.argmin(seed_bank_losses))
        best_phys = seed_bank_phys[best_idx].astype(np.float32)
        best_hard = float(seed_bank_losses[best_idx])
        proposal_hard = float(best_hard)

        for fin_idx, seed_idx in enumerate(finisher_seed_idx):
            x_init = seed_bank_phys[seed_idx]
            x_fin, hard_fin = solve_inverse_by_block(
                p_refine,
                prev_j,
                prev_slot_mask_j,
                thrower_block_j,
                target_block0_j,
                target_block0_mask_j,
                target_block1_j,
                target_block1_mask_j,
                bounds=bounds,
                pop_size=448,
                generations=26,
                elite_frac=0.20,
                sigma_init=0.05,
                sigma_floor=0.002,
                ema_alpha=0.74,
                use_full_cov=True,
                mix_with_best_frac=0.03,
                jitter_anchor=0.0010,
                key=jax.random.PRNGKey(seed + 400 + fin_idx),
                init_x=x_init,
                loss_threshold=0.10,
                log_topk=2,
                eval_chunk_size=400,
                batched_hard_fn=batched_hard_refine,
            )
            hard_fin = float(hard_fin)
            if hard_fin < best_hard:
                best_hard = hard_fin
                best_phys = np.asarray(x_fin, dtype=np.float32)

        x_coord, hard_coord = _coordinate_pattern_polish(best_phys, eval_hard_np=eval_hard_np)
        if float(hard_coord) < best_hard:
            best_hard = float(hard_coord)
            best_phys = x_coord.astype(np.float32)

        x_nm, hard_nm = _nelder_mead_polish(best_phys, eval_hard_np=eval_hard_np, batched_hard_fn=batched_hard_refine)
        if float(hard_nm) < best_hard:
            best_hard = float(hard_nm)
            best_phys = x_nm.astype(np.float32)

        return best_phys, float(best_hard), proposal_hard, next_pts_inb, float(warm_init_hard), used_coarse_fallback

    def run_portfolio_large_coord_nm_rescue_cem(prev_slots: np.ndarray,
                                                prev_slot_mask: np.ndarray,
                                                thrower_block: int,
                                                target_block0_full: np.ndarray,
                                                target_block0_mask: np.ndarray,
                                                target_block1_full: np.ndarray,
                                                target_block1_mask: np.ndarray,
                                                warm_start_x: Optional[np.ndarray] = None,
                                                seed: int = 0,
                                                rescue_threshold: float = 0.25):
        result = run_portfolio_large_coord_rescue_cem(
            prev_slots,
            prev_slot_mask,
            thrower_block,
            target_block0_full,
            target_block0_mask,
            target_block1_full,
            target_block1_mask,
            warm_start_x=warm_start_x,
            seed=seed,
            rescue_threshold=rescue_threshold,
        )
        x_best, hard_refine, hard_coarse, next_inb, warm_init_hard, used_coarse_fallback = result
        (
            _next_pts_inb,
            _prev_j,
            _prev_slot_mask_j,
            _thrower_block_j,
            _target_block0_j,
            _target_block0_mask_j,
            _target_block1_j,
            _target_block1_mask_j,
            eval_hard_np,
        ) = _prepare_objective(
            prev_slots,
            prev_slot_mask,
            thrower_block,
            target_block0_full,
            target_block0_mask,
            target_block1_full,
            target_block1_mask,
        )
        x_nm, hard_nm = _nelder_mead_polish(np.asarray(x_best, dtype=np.float32), eval_hard_np=eval_hard_np, batched_hard_fn=batched_hard_refine)
        if float(hard_nm) < float(hard_refine):
            return x_nm, float(hard_nm), hard_coarse, next_inb, warm_init_hard, used_coarse_fallback
        return result

    def run_portfolio_large_coord_optrescue_cem(prev_slots: np.ndarray,
                                                prev_slot_mask: np.ndarray,
                                                thrower_block: int,
                                                target_block0_full: np.ndarray,
                                                target_block0_mask: np.ndarray,
                                                target_block1_full: np.ndarray,
                                                target_block1_mask: np.ndarray,
                                                warm_start_x: Optional[np.ndarray] = None,
                                                seed: int = 0,
                                                rescue_threshold: float = 0.25):
        result = run_portfolio_large_coord_nm_rescue_cem(
            prev_slots,
            prev_slot_mask,
            thrower_block,
            target_block0_full,
            target_block0_mask,
            target_block1_full,
            target_block1_mask,
            warm_start_x=warm_start_x,
            seed=seed,
            rescue_threshold=rescue_threshold,
        )
        x_best, hard_refine, hard_coarse, next_inb, warm_init_hard, used_coarse_fallback = result
        if float(hard_refine) <= float(rescue_threshold):
            return result

        (
            _next_pts_inb,
            prev_j,
            prev_slot_mask_j,
            thrower_block_j,
            target_block0_j,
            target_block0_mask_j,
            target_block1_j,
            target_block1_mask_j,
            eval_hard_np,
        ) = _prepare_objective(
            prev_slots,
            prev_slot_mask,
            thrower_block,
            target_block0_full,
            target_block0_mask,
            target_block1_full,
            target_block1_mask,
        )

        alt_refine = _get_batched_hard("refine", "optimal")
        seed_bank_x01 = _make_seed_bank_halton_refine(
            seed=seed + 91,
            init_x=np.asarray(x_best, dtype=np.float32),
            prev_slots=prev_slots,
            prev_slot_mask=prev_slot_mask,
            next_pts_inb=next_inb,
        )
        seed_bank_phys = _xphys_from_x01_np(seed_bank_x01)
        alt_seed_losses = eval_hard_np(alt_refine, seed_bank_phys)
        alt_seed_idx = _pick_distinct_indices(
            seed_bank_x01,
            alt_seed_losses,
            max_keep=3,
            min_dist=0.08,
        )

        best_phys = np.asarray(x_best, dtype=np.float32)
        best_hard = float(hard_refine)

        for fin_idx, seed_idx in enumerate(alt_seed_idx):
            x_init = seed_bank_phys[seed_idx]
            x_alt, hard_alt = solve_inverse_by_block(
                p_refine,
                prev_j,
                prev_slot_mask_j,
                thrower_block_j,
                target_block0_j,
                target_block0_mask_j,
                target_block1_j,
                target_block1_mask_j,
                bounds=bounds,
                pop_size=416,
                generations=22,
                elite_frac=0.20,
                sigma_init=0.05,
                sigma_floor=0.002,
                ema_alpha=0.74,
                use_full_cov=True,
                mix_with_best_frac=0.03,
                jitter_anchor=0.0010,
                key=jax.random.PRNGKey(seed + 500 + fin_idx),
                init_x=x_init,
                loss_threshold=0.10,
                log_topk=2,
                eval_chunk_size=400,
                batched_hard_fn=alt_refine,
            )
            x_alt = np.asarray(x_alt, dtype=np.float32)
            hard_alt_baseline = float(eval_hard_np(batched_hard_refine_baseline, x_alt)[0])
            if hard_alt_baseline < best_hard:
                best_hard = hard_alt_baseline
                best_phys = x_alt

        x_coord, hard_coord = _coordinate_pattern_polish(best_phys, eval_hard_np=eval_hard_np)
        if float(hard_coord) < best_hard:
            best_hard = float(hard_coord)
            best_phys = x_coord.astype(np.float32)
        x_nm, hard_nm = _nelder_mead_polish(best_phys, eval_hard_np=eval_hard_np, batched_hard_fn=batched_hard_refine_baseline)
        if float(hard_nm) < best_hard:
            best_hard = float(hard_nm)
            best_phys = x_nm.astype(np.float32)

        if float(best_hard) < float(hard_refine):
            return best_phys, float(best_hard), hard_coarse, next_inb, warm_init_hard, used_coarse_fallback
        return result

    def run_portfolio_large_targeted_optrescue_cem(prev_slots: np.ndarray,
                                                   prev_slot_mask: np.ndarray,
                                                   thrower_block: int,
                                                   target_block0_full: np.ndarray,
                                                   target_block0_mask: np.ndarray,
                                                   target_block1_full: np.ndarray,
                                                   target_block1_mask: np.ndarray,
                                                   warm_start_x: Optional[np.ndarray] = None,
                                                   seed: int = 0,
                                                   rescue_threshold: float = 0.25):
        result = run_portfolio_large_targeted_coord_cem(
            prev_slots,
            prev_slot_mask,
            thrower_block,
            target_block0_full,
            target_block0_mask,
            target_block1_full,
            target_block1_mask,
            warm_start_x=warm_start_x,
            seed=seed,
        )
        x_best, hard_refine, hard_coarse, next_inb, warm_init_hard, used_coarse_fallback = result
        if float(hard_refine) <= float(rescue_threshold):
            return result

        (
            _next_pts_inb,
            prev_j,
            prev_slot_mask_j,
            thrower_block_j,
            target_block0_j,
            target_block0_mask_j,
            target_block1_j,
            target_block1_mask_j,
            eval_hard_np,
        ) = _prepare_objective(
            prev_slots,
            prev_slot_mask,
            thrower_block,
            target_block0_full,
            target_block0_mask,
            target_block1_full,
            target_block1_mask,
        )

        alt_refine = _get_batched_hard("refine", "optimal")
        seed_bank_x01 = _make_seed_bank_halton_refine(
            seed=seed + 191,
            init_x=np.asarray(x_best, dtype=np.float32),
            prev_slots=prev_slots,
            prev_slot_mask=prev_slot_mask,
            next_pts_inb=next_inb,
        )
        seed_bank_phys = _xphys_from_x01_np(seed_bank_x01)
        alt_seed_losses = eval_hard_np(alt_refine, seed_bank_phys)
        alt_seed_idx = _pick_distinct_indices(
            seed_bank_x01,
            alt_seed_losses,
            max_keep=4,
            min_dist=0.08,
        )

        best_phys = np.asarray(x_best, dtype=np.float32)
        best_hard = float(hard_refine)

        for fin_idx, seed_idx in enumerate(alt_seed_idx):
            x_init = seed_bank_phys[seed_idx]
            x_alt, hard_alt = solve_inverse_by_block(
                p_refine,
                prev_j,
                prev_slot_mask_j,
                thrower_block_j,
                target_block0_j,
                target_block0_mask_j,
                target_block1_j,
                target_block1_mask_j,
                bounds=bounds,
                pop_size=416,
                generations=22,
                elite_frac=0.20,
                sigma_init=0.05,
                sigma_floor=0.002,
                ema_alpha=0.74,
                use_full_cov=True,
                mix_with_best_frac=0.03,
                jitter_anchor=0.0010,
                key=jax.random.PRNGKey(seed + 600 + fin_idx),
                init_x=x_init,
                loss_threshold=0.10,
                log_topk=2,
                eval_chunk_size=400,
                batched_hard_fn=alt_refine,
            )
            x_alt = np.asarray(x_alt, dtype=np.float32)
            hard_alt_baseline = float(eval_hard_np(batched_hard_refine_baseline, x_alt)[0])
            if hard_alt_baseline < best_hard:
                best_hard = hard_alt_baseline
                best_phys = x_alt

        x_coord, hard_coord = _coordinate_pattern_polish(best_phys, eval_hard_np=eval_hard_np)
        if float(hard_coord) < best_hard:
            best_hard = float(hard_coord)
            best_phys = x_coord.astype(np.float32)

        x_nm, hard_nm = _nelder_mead_polish(best_phys, eval_hard_np=eval_hard_np, batched_hard_fn=batched_hard_refine_baseline)
        if float(hard_nm) < best_hard:
            best_hard = float(hard_nm)
            best_phys = x_nm.astype(np.float32)

        if float(best_hard) < float(hard_refine):
            return best_phys, float(best_hard), hard_coarse, next_inb, warm_init_hard, used_coarse_fallback
        return result

    def run_portfolio_large_coord_optrescue_propbank_cem(prev_slots: np.ndarray,
                                                         prev_slot_mask: np.ndarray,
                                                         thrower_block: int,
                                                         target_block0_full: np.ndarray,
                                                         target_block0_mask: np.ndarray,
                                                         target_block1_full: np.ndarray,
                                                         target_block1_mask: np.ndarray,
                                                         proposal_bank_phys: Optional[np.ndarray] = None,
                                                         warm_start_x: Optional[np.ndarray] = None,
                                                         seed: int = 0,
                                                         rescue_threshold: float = 0.25):
        prop_bank = np.empty((0, 4), dtype=np.float32)
        if proposal_bank_phys is not None:
            prop_bank = np.asarray(proposal_bank_phys, dtype=np.float32).reshape(-1, 4)
            finite_mask = np.all(np.isfinite(prop_bank), axis=1)
            prop_bank = prop_bank[finite_mask]
            if prop_bank.size:
                prop_bank = np.clip(prop_bank, lo, hi).astype(np.float32)
                prop_bank = np.unique(np.round(prop_bank, 6), axis=0).astype(np.float32)

        warm_local = warm_start_x
        if warm_local is None and prop_bank.size:
            warm_local = prop_bank[0]

        result = run_portfolio_large_coord_optrescue_cem(
            prev_slots,
            prev_slot_mask,
            thrower_block,
            target_block0_full,
            target_block0_mask,
            target_block1_full,
            target_block1_mask,
            warm_start_x=warm_local,
            seed=seed,
            rescue_threshold=rescue_threshold,
        )
        if not prop_bank.size:
            return result

        x_best, hard_refine, hard_coarse, next_inb, warm_init_hard, used_coarse_fallback = result

        (
            _next_pts_inb,
            prev_j,
            prev_slot_mask_j,
            thrower_block_j,
            target_block0_j,
            target_block0_mask_j,
            target_block1_j,
            target_block1_mask_j,
            eval_hard_np,
        ) = _prepare_objective(
            prev_slots,
            prev_slot_mask,
            thrower_block,
            target_block0_full,
            target_block0_mask,
            target_block1_full,
            target_block1_mask,
        )

        candidate_bank = prop_bank
        if np.all(np.isfinite(x_best)):
            candidate_bank = np.concatenate([candidate_bank, np.asarray(x_best, dtype=np.float32)[None, :]], axis=0)
        candidate_bank = np.unique(np.round(np.clip(candidate_bank, lo, hi), 6), axis=0).astype(np.float32)
        cand_losses = eval_hard_np(batched_hard_refine, candidate_bank)
        prop_pick_idx = _pick_distinct_indices(
            _x01_from_phys_np(candidate_bank),
            cand_losses,
            max_keep=min(4, int(candidate_bank.shape[0])),
            min_dist=0.05,
        )

        best_phys = np.asarray(x_best, dtype=np.float32)
        best_hard = float(hard_refine)

        for fin_idx, idx in enumerate(prop_pick_idx):
            x_init = candidate_bank[idx].astype(np.float32)
            seed_hard = float(cand_losses[idx])
            if seed_hard < best_hard:
                best_hard = seed_hard
                best_phys = x_init
            x_fin, hard_fin = solve_inverse_by_block(
                p_refine,
                prev_j,
                prev_slot_mask_j,
                thrower_block_j,
                target_block0_j,
                target_block0_mask_j,
                target_block1_j,
                target_block1_mask_j,
                bounds=bounds,
                pop_size=288,
                generations=16,
                elite_frac=0.20,
                sigma_init=0.035,
                sigma_floor=0.002,
                ema_alpha=0.74,
                use_full_cov=True,
                mix_with_best_frac=0.03,
                jitter_anchor=0.0010,
                key=jax.random.PRNGKey(seed + 900 + fin_idx),
                init_x=x_init,
                loss_threshold=0.10,
                log_topk=2,
                eval_chunk_size=288,
                batched_hard_fn=batched_hard_refine,
            )
            hard_fin = float(hard_fin)
            if hard_fin < best_hard:
                best_hard = hard_fin
                best_phys = np.asarray(x_fin, dtype=np.float32)

        x_coord, hard_coord = _coordinate_pattern_polish(best_phys, eval_hard_np=eval_hard_np)
        if float(hard_coord) < best_hard:
            best_hard = float(hard_coord)
            best_phys = x_coord.astype(np.float32)

        x_nm, hard_nm = _nelder_mead_polish(best_phys, eval_hard_np=eval_hard_np, batched_hard_fn=batched_hard_refine_baseline)
        if float(hard_nm) < best_hard:
            best_hard = float(hard_nm)
            best_phys = x_nm.astype(np.float32)

        if float(best_hard) < float(hard_refine):
            return best_phys, float(best_hard), hard_coarse, next_inb, warm_init_hard, used_coarse_fallback
        return result

    def run_portfolio_large_transition_adaptive_cem(prev_slots: np.ndarray,
                                                    prev_slot_mask: np.ndarray,
                                                    thrower_block: int,
                                                    target_block0_full: np.ndarray,
                                                    target_block0_mask: np.ndarray,
                                                    target_block1_full: np.ndarray,
                                                    target_block1_mask: np.ndarray,
                                                    warm_start_x: Optional[np.ndarray] = None,
                                                    seed: int = 0):
        next_pts_inb = np.concatenate(
            [
                target_block0_full[target_block0_mask & _in_bounds_mask_np(target_block0_full)],
                target_block1_full[target_block1_mask & _in_bounds_mask_np(target_block1_full)],
            ],
            axis=0,
        )
        prev_pts = prev_slots[prev_slot_mask]
        prev_inb_n = int(np.sum(_in_bounds_mask_np(prev_pts)))
        next_inb_n = int(next_pts_inb.shape[0])
        takeout_like = next_inb_n <= prev_inb_n

        if takeout_like:
            return run_portfolio_large_refinebank_coord_nm_cem(
                prev_slots,
                prev_slot_mask,
                thrower_block,
                target_block0_full,
                target_block0_mask,
                target_block1_full,
                target_block1_mask,
                warm_start_x=warm_start_x,
                seed=seed,
            )
        return run_portfolio_large_coord_nm_rescue_cem(
            prev_slots,
            prev_slot_mask,
            thrower_block,
            target_block0_full,
            target_block0_mask,
            target_block1_full,
            target_block1_mask,
            warm_start_x=warm_start_x,
            seed=seed,
            rescue_threshold=0.25,
        )

    def run_portfolio_rescue_cem(prev_slots: np.ndarray,
                                 prev_slot_mask: np.ndarray,
                                 thrower_block: int,
                                 target_block0_full: np.ndarray,
                                 target_block0_mask: np.ndarray,
                                 target_block1_full: np.ndarray,
                                 target_block1_mask: np.ndarray,
                                 warm_start_x: Optional[np.ndarray] = None,
                                 seed: int = 0):
        warm_accept_threshold = 0.10
        (
            next_pts_inb,
            prev_j,
            prev_slot_mask_j,
            thrower_block_j,
            target_block0_j,
            target_block0_mask_j,
            target_block1_j,
            target_block1_mask_j,
            eval_hard_np,
        ) = _prepare_objective(
            prev_slots,
            prev_slot_mask,
            thrower_block,
            target_block0_full,
            target_block0_mask,
            target_block1_full,
            target_block1_mask,
        )

        warm_init = None
        warm_init_hard = np.nan
        used_coarse_fallback = False
        if warm_start_x is not None:
            warm_arr = np.asarray(warm_start_x, dtype=np.float32).reshape(-1)
            if warm_arr.shape[0] == 4 and np.all(np.isfinite(warm_arr)):
                warm_init = np.clip(warm_arr, lo, hi).astype(np.float32)
                warm_init_hard = float(eval_hard_np(batched_hard_refine, warm_init)[0])
                used_coarse_fallback = True
                if warm_init_hard <= warm_accept_threshold:
                    return warm_init, float(warm_init_hard), np.nan, next_pts_inb, float(warm_init_hard), used_coarse_fallback

        warm_score = float(warm_init_hard) if np.isfinite(warm_init_hard) else 1.0
        if warm_score <= 0.25:
            scout_keep, scout_pop, scout_gen = 4, 320, 10
            fin_keep, fin_pop, fin_gen = 2, 512, 28
            polish_pop, polish_gen = 192, 10
        elif warm_score <= 1.0:
            scout_keep, scout_pop, scout_gen = 5, 384, 12
            fin_keep, fin_pop, fin_gen = 3, 512, 32
            polish_pop, polish_gen = 224, 12
        else:
            scout_keep, scout_pop, scout_gen = 6, 384, 14
            fin_keep, fin_pop, fin_gen = 3, 640, 36
            polish_pop, polish_gen = 256, 16

        seed_bank_x01 = _make_seed_bank_rescue(seed=seed, init_x=warm_init)
        seed_bank_phys = _xphys_from_x01_np(seed_bank_x01)
        seed_bank_losses = eval_hard_np(batched_hard_coarse, seed_bank_phys)
        restart_seed_idx = _pick_distinct_indices(
            seed_bank_x01,
            seed_bank_losses,
            max_keep=scout_keep,
            min_dist=0.18,
        )

        scout_results = []
        for restart_idx, seed_idx in enumerate(restart_seed_idx):
            seed_x01 = seed_bank_x01[seed_idx]
            seed_phys = _xphys_from_x01_np(seed_x01)
            x_scout, hard_scout = solve_inverse_by_block(
                p_coarse,
                prev_j,
                prev_slot_mask_j,
                thrower_block_j,
                target_block0_j,
                target_block0_mask_j,
                target_block1_j,
                target_block1_mask_j,
                bounds=bounds,
                pop_size=scout_pop,
                generations=scout_gen,
                elite_frac=0.22,
                sigma_init=0.18,
                sigma_floor=0.01,
                ema_alpha=0.68,
                use_full_cov=False,
                mix_with_best_frac=0.14,
                jitter_anchor=0.008,
                key=jax.random.PRNGKey(seed + 300 + restart_idx),
                init_x=seed_phys,
                loss_threshold=0.25,
                log_topk=2,
                eval_chunk_size=400,
                batched_hard_fn=batched_hard_coarse,
            )
            scout_results.append(
                {
                    "x_phys": np.asarray(x_scout, dtype=np.float32),
                    "x01": _x01_from_phys_np(np.asarray(x_scout, dtype=np.float32)),
                    "coarse_hard": float(hard_scout),
                }
            )

        shortlist = list(scout_results)
        if warm_init is not None:
            shortlist.append(
                {
                    "x_phys": warm_init.astype(np.float32),
                    "x01": _x01_from_phys_np(warm_init),
                    "coarse_hard": float(eval_hard_np(batched_hard_coarse, warm_init)[0]),
                }
            )

        shortlist_phys = np.asarray([r["x_phys"] for r in shortlist], dtype=np.float32)
        shortlist_refine_losses = eval_hard_np(batched_hard_refine, shortlist_phys)
        for idx, refine_hard in enumerate(shortlist_refine_losses.tolist()):
            shortlist[idx]["refine_hard"] = float(refine_hard)

        shortlist_x01 = np.asarray([r["x01"] for r in shortlist], dtype=np.float32)
        finisher_idx = _pick_distinct_indices(
            shortlist_x01,
            np.asarray([r["refine_hard"] for r in shortlist], dtype=np.float32),
            max_keep=fin_keep,
            min_dist=0.08,
        )

        best_phys = None
        best_hard = np.inf
        proposal_hard = float(np.min([r["coarse_hard"] for r in shortlist])) if shortlist else np.nan
        for fin_idx, idx in enumerate(finisher_idx):
            x_init = shortlist[idx]["x_phys"]
            x_fin, hard_fin = solve_inverse_by_block(
                p_refine,
                prev_j,
                prev_slot_mask_j,
                thrower_block_j,
                target_block0_j,
                target_block0_mask_j,
                target_block1_j,
                target_block1_mask_j,
                bounds=bounds,
                pop_size=fin_pop,
                generations=fin_gen,
                elite_frac=0.20,
                sigma_init=0.07,
                sigma_floor=0.003,
                ema_alpha=0.72,
                use_full_cov=True,
                mix_with_best_frac=0.05,
                jitter_anchor=0.0010,
                key=jax.random.PRNGKey(seed + 400 + fin_idx),
                init_x=x_init,
                loss_threshold=0.10,
                log_topk=2,
                eval_chunk_size=400,
                batched_hard_fn=batched_hard_refine,
            )
            hard_fin = float(hard_fin)
            if hard_fin < best_hard:
                best_hard = hard_fin
                best_phys = np.asarray(x_fin, dtype=np.float32)

        if best_phys is None:
            best_idx = int(np.argmin(np.asarray([r["refine_hard"] for r in shortlist], dtype=np.float32)))
            best_phys = shortlist[best_idx]["x_phys"]
            best_hard = float(shortlist[best_idx]["refine_hard"])

        x_polish, hard_polish = solve_inverse_by_block(
            p_refine,
            prev_j,
            prev_slot_mask_j,
            thrower_block_j,
            target_block0_j,
            target_block0_mask_j,
            target_block1_j,
            target_block1_mask_j,
            bounds=bounds,
            pop_size=polish_pop,
            generations=polish_gen,
            elite_frac=0.20,
            sigma_init=0.025,
            sigma_floor=0.002,
            ema_alpha=0.75,
            use_full_cov=False,
            mix_with_best_frac=0.02,
            jitter_anchor=0.0008,
            key=jax.random.PRNGKey(seed + 500),
            init_x=best_phys,
            loss_threshold=0.10,
            log_topk=2,
            eval_chunk_size=256,
            batched_hard_fn=batched_hard_refine,
        )
        hard_polish = float(hard_polish)
        if hard_polish < best_hard:
            best_hard = hard_polish
            best_phys = np.asarray(x_polish, dtype=np.float32)

        return best_phys, float(best_hard), proposal_hard, next_pts_inb, float(warm_init_hard), used_coarse_fallback

    def run_portfolio_refit_hybrid_cem(prev_slots: np.ndarray,
                                       prev_slot_mask: np.ndarray,
                                       thrower_block: int,
                                       target_block0_full: np.ndarray,
                                       target_block0_mask: np.ndarray,
                                       target_block1_full: np.ndarray,
                                       target_block1_mask: np.ndarray,
                                       warm_start_x: Optional[np.ndarray] = None,
                                       seed: int = 0):
        warm_accept_threshold = 0.10
        (
            next_pts_inb,
            prev_j,
            prev_slot_mask_j,
            thrower_block_j,
            target_block0_j,
            target_block0_mask_j,
            target_block1_j,
            target_block1_mask_j,
            eval_hard_np,
        ) = _prepare_objective(
            prev_slots,
            prev_slot_mask,
            thrower_block,
            target_block0_full,
            target_block0_mask,
            target_block1_full,
            target_block1_mask,
        )

        warm_init = None
        warm_init_hard = np.nan
        if warm_start_x is not None:
            warm_arr = np.asarray(warm_start_x, dtype=np.float32).reshape(-1)
            if warm_arr.shape[0] == 4 and np.all(np.isfinite(warm_arr)):
                warm_init = np.clip(warm_arr, lo, hi).astype(np.float32)
                warm_init_hard = float(eval_hard_np(batched_hard_refine, warm_init)[0])
                if warm_init_hard <= warm_accept_threshold:
                    return warm_init, float(warm_init_hard), np.nan, next_pts_inb, float(warm_init_hard), False

        if warm_init is None:
            return run_portfolio_large_cem(
                prev_slots,
                prev_slot_mask,
                thrower_block,
                target_block0_full,
                target_block0_mask,
                target_block1_full,
                target_block1_mask,
                warm_start_x=None,
                seed=seed,
            )

        if float(warm_init_hard) > 1.0:
            return run_portfolio_rescue_cem(
                prev_slots,
                prev_slot_mask,
                thrower_block,
                target_block0_full,
                target_block0_mask,
                target_block1_full,
                target_block1_mask,
                warm_start_x=warm_init,
                seed=seed,
            )

        if float(warm_init_hard) <= 0.25:
            fin_keep, fin_pop, fin_gen = 4, 224, 12
            fin_sigma_init, fin_sigma_floor = 0.035, 0.0015
            fin_mix, fin_jitter, fin_eval_chunk = 0.03, 0.0006, 224
            polish_pop, polish_gen = 224, 10
            polish_sigma_init = 0.010
            shortlist_dist = 0.025
        else:
            fin_keep, fin_pop, fin_gen = 5, 288, 16
            fin_sigma_init, fin_sigma_floor = 0.050, 0.0020
            fin_mix, fin_jitter, fin_eval_chunk = 0.03, 0.0008, 288
            polish_pop, polish_gen = 256, 12
            polish_sigma_init = 0.014
            shortlist_dist = 0.035

        seed_bank_x01 = _make_seed_bank_refine_local(seed=seed, init_x=warm_init)
        seed_bank_phys = _xphys_from_x01_np(seed_bank_x01)
        seed_bank_losses = eval_hard_np(batched_hard_refine, seed_bank_phys)
        shortlist_idx = _pick_distinct_indices(
            seed_bank_x01,
            seed_bank_losses,
            max_keep=fin_keep,
            min_dist=shortlist_dist,
        )

        best_idx = int(np.argmin(seed_bank_losses))
        best_phys = seed_bank_phys[best_idx].astype(np.float32)
        best_hard = float(seed_bank_losses[best_idx])
        proposal_hard = best_hard

        if best_hard <= warm_accept_threshold:
            return best_phys, best_hard, proposal_hard, next_pts_inb, float(warm_init_hard), False

        for fin_i, idx in enumerate(shortlist_idx):
            x_init = seed_bank_phys[idx].astype(np.float32)
            x_fin, hard_fin = solve_inverse_by_block(
                p_refine,
                prev_j,
                prev_slot_mask_j,
                thrower_block_j,
                target_block0_j,
                target_block0_mask_j,
                target_block1_j,
                target_block1_mask_j,
                bounds=bounds,
                pop_size=fin_pop,
                generations=fin_gen,
                elite_frac=0.20,
                sigma_init=fin_sigma_init,
                sigma_floor=fin_sigma_floor,
                ema_alpha=0.74,
                use_full_cov=True,
                mix_with_best_frac=fin_mix,
                jitter_anchor=fin_jitter,
                key=jax.random.PRNGKey(seed + 600 + fin_i),
                init_x=x_init,
                loss_threshold=0.10,
                log_topk=2,
                eval_chunk_size=fin_eval_chunk,
                batched_hard_fn=batched_hard_refine,
            )
            hard_fin = float(hard_fin)
            if hard_fin < best_hard:
                best_hard = hard_fin
                best_phys = np.asarray(x_fin, dtype=np.float32)

        x_polish, hard_polish = solve_inverse_by_block(
            p_refine,
            prev_j,
            prev_slot_mask_j,
            thrower_block_j,
            target_block0_j,
            target_block0_mask_j,
            target_block1_j,
            target_block1_mask_j,
            bounds=bounds,
            pop_size=polish_pop,
            generations=polish_gen,
            elite_frac=0.18,
            sigma_init=polish_sigma_init,
            sigma_floor=0.0012,
            ema_alpha=0.76,
            use_full_cov=False,
            mix_with_best_frac=0.01,
            jitter_anchor=0.0004,
            key=jax.random.PRNGKey(seed + 700),
            init_x=best_phys,
            loss_threshold=0.10,
            log_topk=2,
            eval_chunk_size=min(polish_pop, 256),
            batched_hard_fn=batched_hard_refine,
        )
        hard_polish = float(hard_polish)
        if hard_polish < best_hard:
            best_hard = hard_polish
            best_phys = np.asarray(x_polish, dtype=np.float32)

        if (float(warm_init_hard) > 0.5) and (best_hard > 0.25):
            x_rescue, hard_rescue, _, _, _, used_coarse_fallback = run_portfolio_rescue_cem(
                prev_slots,
                prev_slot_mask,
                thrower_block,
                target_block0_full,
                target_block0_mask,
                target_block1_full,
                target_block1_mask,
                warm_start_x=warm_init,
                seed=seed + 1000,
            )
            if float(hard_rescue) < best_hard:
                return x_rescue, float(hard_rescue), proposal_hard, next_pts_inb, float(warm_init_hard), used_coarse_fallback

        return best_phys, float(best_hard), proposal_hard, next_pts_inb, float(warm_init_hard), False

    def run_portfolio_large_multistart_cem(prev_slots: np.ndarray,
                                           prev_slot_mask: np.ndarray,
                                           thrower_block: int,
                                           target_block0_full: np.ndarray,
                                           target_block0_mask: np.ndarray,
                                           target_block1_full: np.ndarray,
                                           target_block1_mask: np.ndarray,
                                           warm_start_x: Optional[np.ndarray] = None,
                                           seed: int = 0):
        best_result = None
        best_hard = np.inf
        for restart_idx in range(2):
            result = run_portfolio_large_cem(
                prev_slots,
                prev_slot_mask,
                thrower_block,
                target_block0_full,
                target_block0_mask,
                target_block1_full,
                target_block1_mask,
                warm_start_x=warm_start_x,
                seed=seed + 10000 * restart_idx,
            )
            x_best, hard_refine, hard_coarse, next_inb, warm_init_hard, used_coarse_fallback = result
            if float(hard_refine) < best_hard:
                best_hard = float(hard_refine)
                best_result = (
                    np.asarray(x_best, dtype=np.float32),
                    float(hard_refine),
                    float(hard_coarse) if np.isfinite(hard_coarse) else hard_coarse,
                    next_inb,
                    float(warm_init_hard),
                    bool(used_coarse_fallback),
                )
            if best_hard <= 0.10:
                break
        assert best_result is not None
        return best_result

    part_idx = 0
    buffer_rows: List[dict] = []

    for local_i, key in enumerate(keys):
        # Build prev/next states in meters
        prev_state_m, next_state_m, shot_row = _load_prev_next_states(df_end_index, key)

        prev_slots, prev_slot_mask = _state_to_fixed_slot_arrays(prev_state_m)
        prev_ids = sorted(prev_state_m.keys())
        next_ids = sorted(next_state_m.keys())

        # Optional sanitize prev to avoid starting overlaps (solver stability)
        if int(np.sum(prev_slot_mask)) > 1:
            prev_compact = _separate_overlaps(prev_slots[prev_slot_mask], min_gap=MIN_CLEAR)
            prev_slots = prev_slots.copy()
            prev_slots[prev_slot_mask] = prev_compact

        obs_throw_slot_id = float(shot_row.get("obs_throw_slot_id", np.nan))
        team_slot_block = float(shot_row.get("team_slot_block", np.nan))
        thrower_block = infer_thrower_block(prev_ids, next_ids, obs_throw_slot_id, team_slot_block)
        target_block0, target_block0_mask = _targets_for_block(next_state_m, 1, 6)
        target_block1, target_block1_mask = _targets_for_block(next_state_m, 7, 12)
        warm_cols = ["warm_est_speed", "warm_est_angle", "warm_est_spin", "warm_est_y0"]
        if all(col in shot_row.index for col in warm_cols):
            warm_start_x = np.asarray([shot_row.get(col, np.nan) for col in warm_cols], dtype=np.float32)
        else:
            warm_start_x = None
        proposal_bank_x = _proposal_bank_from_row(shot_row)

        # Run inverse search
        try:
            seed_val = base_seed + (key.comp ^ key.sess ^ key.game ^ key.end ^ key.shot) * 2 + gpu_local_index
            if solver_method == "portfolio":
                x_best, hard_refine, hard_coarse, next_inb, warm_init_hard, used_coarse_fallback = run_portfolio_cem(
                    prev_slots,
                    prev_slot_mask,
                    thrower_block,
                    target_block0,
                    target_block0_mask,
                    target_block1,
                    target_block1_mask,
                    warm_start_x=warm_start_x,
                    seed=seed_val,
                )
            elif solver_method == "portfolio_large":
                x_best, hard_refine, hard_coarse, next_inb, warm_init_hard, used_coarse_fallback = run_portfolio_large_cem(
                    prev_slots,
                    prev_slot_mask,
                    thrower_block,
                    target_block0,
                    target_block0_mask,
                    target_block1,
                    target_block1_mask,
                    warm_start_x=warm_start_x,
                    seed=seed_val,
                )
            elif solver_method == "portfolio_large_targeted":
                x_best, hard_refine, hard_coarse, next_inb, warm_init_hard, used_coarse_fallback = run_portfolio_large_targeted_cem(
                    prev_slots,
                    prev_slot_mask,
                    thrower_block,
                    target_block0,
                    target_block0_mask,
                    target_block1,
                    target_block1_mask,
                    warm_start_x=warm_start_x,
                    seed=seed_val,
                )
            elif solver_method == "portfolio_large_coord":
                x_best, hard_refine, hard_coarse, next_inb, warm_init_hard, used_coarse_fallback = run_portfolio_large_coord_cem(
                    prev_slots,
                    prev_slot_mask,
                    thrower_block,
                    target_block0,
                    target_block0_mask,
                    target_block1,
                    target_block1_mask,
                    warm_start_x=warm_start_x,
                    seed=seed_val,
                )
            elif solver_method == "portfolio_large_targeted_coord":
                x_best, hard_refine, hard_coarse, next_inb, warm_init_hard, used_coarse_fallback = run_portfolio_large_targeted_coord_cem(
                    prev_slots,
                    prev_slot_mask,
                    thrower_block,
                    target_block0,
                    target_block0_mask,
                    target_block1,
                    target_block1_mask,
                    warm_start_x=warm_start_x,
                    seed=seed_val,
                )
            elif solver_method == "portfolio_large_coord_rescue025":
                x_best, hard_refine, hard_coarse, next_inb, warm_init_hard, used_coarse_fallback = run_portfolio_large_coord_rescue_cem(
                    prev_slots,
                    prev_slot_mask,
                    thrower_block,
                    target_block0,
                    target_block0_mask,
                    target_block1,
                    target_block1_mask,
                    warm_start_x=warm_start_x,
                    seed=seed_val,
                    rescue_threshold=0.25,
                )
            elif solver_method == "portfolio_large_coord_rescue050":
                x_best, hard_refine, hard_coarse, next_inb, warm_init_hard, used_coarse_fallback = run_portfolio_large_coord_rescue_cem(
                    prev_slots,
                    prev_slot_mask,
                    thrower_block,
                    target_block0,
                    target_block0_mask,
                    target_block1,
                    target_block1_mask,
                    warm_start_x=warm_start_x,
                    seed=seed_val,
                    rescue_threshold=0.50,
                )
            elif solver_method == "portfolio_large_refinebank_coord_nm":
                x_best, hard_refine, hard_coarse, next_inb, warm_init_hard, used_coarse_fallback = run_portfolio_large_refinebank_coord_nm_cem(
                    prev_slots,
                    prev_slot_mask,
                    thrower_block,
                    target_block0,
                    target_block0_mask,
                    target_block1,
                    target_block1_mask,
                    warm_start_x=warm_start_x,
                    seed=seed_val,
                )
            elif solver_method == "portfolio_large_coord_nm_rescue025":
                x_best, hard_refine, hard_coarse, next_inb, warm_init_hard, used_coarse_fallback = run_portfolio_large_coord_nm_rescue_cem(
                    prev_slots,
                    prev_slot_mask,
                    thrower_block,
                    target_block0,
                    target_block0_mask,
                    target_block1,
                    target_block1_mask,
                    warm_start_x=warm_start_x,
                    seed=seed_val,
                    rescue_threshold=0.25,
                )
            elif solver_method == "portfolio_large_coord_optrescue025":
                x_best, hard_refine, hard_coarse, next_inb, warm_init_hard, used_coarse_fallback = run_portfolio_large_coord_optrescue_cem(
                    prev_slots,
                    prev_slot_mask,
                    thrower_block,
                    target_block0,
                    target_block0_mask,
                    target_block1,
                    target_block1_mask,
                    warm_start_x=warm_start_x,
                    seed=seed_val,
                    rescue_threshold=0.25,
                )
            elif solver_method == "portfolio_large_targeted_optrescue025":
                x_best, hard_refine, hard_coarse, next_inb, warm_init_hard, used_coarse_fallback = run_portfolio_large_targeted_optrescue_cem(
                    prev_slots,
                    prev_slot_mask,
                    thrower_block,
                    target_block0,
                    target_block0_mask,
                    target_block1,
                    target_block1_mask,
                    warm_start_x=warm_start_x,
                    seed=seed_val,
                    rescue_threshold=0.25,
                )
            elif solver_method == "portfolio_large_coord_optrescue025_propbank":
                x_best, hard_refine, hard_coarse, next_inb, warm_init_hard, used_coarse_fallback = run_portfolio_large_coord_optrescue_propbank_cem(
                    prev_slots,
                    prev_slot_mask,
                    thrower_block,
                    target_block0,
                    target_block0_mask,
                    target_block1,
                    target_block1_mask,
                    proposal_bank_phys=proposal_bank_x,
                    warm_start_x=warm_start_x,
                    seed=seed_val,
                    rescue_threshold=0.25,
                )
            elif solver_method == "portfolio_large_transition_adaptive":
                x_best, hard_refine, hard_coarse, next_inb, warm_init_hard, used_coarse_fallback = run_portfolio_large_transition_adaptive_cem(
                    prev_slots,
                    prev_slot_mask,
                    thrower_block,
                    target_block0,
                    target_block0_mask,
                    target_block1,
                    target_block1_mask,
                    warm_start_x=warm_start_x,
                    seed=seed_val,
                )
            elif solver_method == "portfolio_rescue":
                x_best, hard_refine, hard_coarse, next_inb, warm_init_hard, used_coarse_fallback = run_portfolio_rescue_cem(
                    prev_slots,
                    prev_slot_mask,
                    thrower_block,
                    target_block0,
                    target_block0_mask,
                    target_block1,
                    target_block1_mask,
                    warm_start_x=warm_start_x,
                    seed=seed_val,
                )
            elif solver_method == "portfolio_refit_hybrid":
                x_best, hard_refine, hard_coarse, next_inb, warm_init_hard, used_coarse_fallback = run_portfolio_refit_hybrid_cem(
                    prev_slots,
                    prev_slot_mask,
                    thrower_block,
                    target_block0,
                    target_block0_mask,
                    target_block1,
                    target_block1_mask,
                    warm_start_x=warm_start_x,
                    seed=seed_val,
                )
            elif solver_method == "portfolio_large_multistart":
                x_best, hard_refine, hard_coarse, next_inb, warm_init_hard, used_coarse_fallback = run_portfolio_large_multistart_cem(
                    prev_slots,
                    prev_slot_mask,
                    thrower_block,
                    target_block0,
                    target_block0_mask,
                    target_block1,
                    target_block1_mask,
                    warm_start_x=warm_start_x,
                    seed=seed_val,
                )
            else:
                x_best, hard_refine, hard_coarse, next_inb, warm_init_hard, used_coarse_fallback = run_two_stage_cem(
                    prev_slots,
                    prev_slot_mask,
                    thrower_block,
                    target_block0,
                    target_block0_mask,
                    target_block1,
                    target_block1_mask,
                    warm_start_x=warm_start_x,
                    seed=seed_val,
                )
            solver_ok = True
            err_msg = ""
        except Exception as e:
            # On failure, emit NaNs but keep row
            import numpy as _np
            x_best = np.array([_np.nan, _np.nan, _np.nan, _np.nan], dtype=np.float32)
            hard_refine = np.nan
            hard_coarse = np.nan
            next_inb = np.empty((0, 2), dtype=np.float32)
            warm_init_hard = np.nan
            used_coarse_fallback = False
            solver_ok = False
            err_msg = repr(e)

        (
            _next_pts_inb_eval,
            _prev_j_eval,
            _prev_slot_mask_j_eval,
            _thrower_block_j_eval,
            _target_block0_j_eval,
            _target_block0_mask_j_eval,
            _target_block1_j_eval,
            _target_block1_mask_j_eval,
            eval_hard_np_post,
        ) = _prepare_objective(
            prev_slots,
            prev_slot_mask,
            thrower_block,
            target_block0,
            target_block0_mask,
            target_block1,
            target_block1_mask,
        )

        objective_hard_refine = float(hard_refine) if np.isfinite(hard_refine) else np.nan
        objective_hard_coarse = float(hard_coarse) if np.isfinite(hard_coarse) else np.nan
        if solver_ok and np.all(np.isfinite(x_best)):
            baseline_hard_refine = float(eval_hard_np_post(batched_hard_refine_baseline, x_best)[0])
            baseline_hard_coarse = float(eval_hard_np_post(batched_hard_coarse_baseline, x_best)[0])
        else:
            baseline_hard_refine = np.nan
            baseline_hard_coarse = np.nan

        prev_N = int(np.sum(prev_slot_mask))
        next_total_N = int(len(next_state_m))
        next_inb_N = int(next_inb.shape[0])

        row = {
            "CompetitionID": key.comp,
            "SessionID": key.sess,
            "GameID": key.game,
            "EndID": key.end,
            "ShotID": key.shot,
            "prev_N": prev_N,
            "next_total_N": next_total_N,
            "next_in_bounds_N": next_inb_N,
            "est_speed": float(x_best[0]),
            "est_angle": float(x_best[1]),
            "est_spin": float(x_best[2]),
            "est_y0": float(x_best[3]),
            "hard_loss_coarse": float(baseline_hard_coarse),
            "hard_loss_refine": float(baseline_hard_refine),
            "objective_hard_loss_coarse": float(objective_hard_coarse),
            "objective_hard_loss_refine": float(objective_hard_refine),
            "warm_start_init_hard": float(warm_init_hard),
            "used_coarse_fallback": bool(used_coarse_fallback),
            "solver_method": solver_method,
            "loss_variant": loss_variant,
            "solver_ok": solver_ok,
            "solver_error": err_msg,
        }

        # Add per-stone converted coordinates (meters) for PREV and NEXT rows
        for i in range(1, CSV_STONE_COUNT + 1):
            # prev
            px, py = (np.nan, np.nan)
            if i in prev_state_m:
                px, py = prev_state_m[i]
            row[f"prev_stone_{i}_x_m"] = float(px) if not np.isnan(px) else np.nan
            row[f"prev_stone_{i}_y_m"] = float(py) if not np.isnan(py) else np.nan

            # next
            nx, ny = (np.nan, np.nan)
            if i in next_state_m:
                nx, ny = next_state_m[i]
            row[f"next_stone_{i}_x_m"] = float(nx) if not np.isnan(nx) else np.nan
            row[f"next_stone_{i}_y_m"] = float(ny) if not np.isnan(ny) else np.nan

            # in-bounds for NEXT (consistent with inverse target selection)
            if not (np.isnan(nx) or np.isnan(ny)):
                inb = (MIN_X < nx < MAX_X) and (MIN_Y < ny < MAX_Y)
            else:
                inb = False
            row[f"next_stone_{i}_inbounds"] = int(inb)

        buffer_rows.append(row)

        # Periodic flush to a part CSV
        if (local_i + 1) % flush_every == 0:
            part_path = os.path.join(
                out_dir,
                f"chunk{chunk_idx:04d}.gpu{gpu_local_index}.part{part_idx:03d}.csv",
            )
            pd.DataFrame(buffer_rows).to_csv(part_path, index=False)
            buffer_rows.clear()
            part_idx += 1
            if verbose:
                print(f"[GPU {gpu_local_index}] wrote {part_path}", flush=True)

    # Final flush for remaining rows
    if buffer_rows:
        part_path = os.path.join(
            out_dir,
            f"chunk{chunk_idx:04d}.gpu{gpu_local_index}.part{part_idx:03d}.csv",
        )
        pd.DataFrame(buffer_rows).to_csv(part_path, index=False)
        if verbose:
            print(f"[GPU {gpu_local_index}] wrote {part_path}", flush=True)


# -------------------- Orchestration --------------------
def _gather_keys(df: pd.DataFrame, limit: Optional[int]) -> List[ShotKey]:
    keys = list(_iter_shots_with_prev(df))
    if limit is not None:
        keys = keys[:limit]
    return keys


def _split_into_chunks(keys: List[ShotKey], chunk_size: int) -> List[List[ShotKey]]:
    return [keys[i:i + chunk_size] for i in range(0, len(keys), chunk_size)]


def _round_robin_slices(keys: List[ShotKey], n: int) -> List[List[ShotKey]]:
    """Split keys into n disjoint slices (round-robin) to balance load."""
    buckets = [[] for _ in range(n)]
    for idx, k in enumerate(keys):
        buckets[idx % n].append(k)
    return buckets


def _load_key_filter_csv(path: Optional[str]) -> Optional[set[tuple[int, int, int, int, int]]]:
    if not path:
        return None
    df = pd.read_csv(path)
    req = SHOT_KEY_COLS
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"keys-csv missing required columns: {missing}")
    return {
        tuple(int(row[c]) for c in req)
        for _, row in df[req].iterrows()
    }


def main(
    csv_path: str,
    out_prefix: str,
    base_seed: Optional[int],
    limit: Optional[int],
    chunk_size: int,
    flush_every: int,
    warm_start_glob: str,
    keys_csv: Optional[str],
    solver_method: str,
    loss_variant: str,
    sim_cfg: SimConfig,
    verbose: bool,
):
    # Read once in parent
    df = pd.read_csv(csv_path)
    df = _attach_throw_slot_features(df)
    warm_df = _load_warm_start_df(warm_start_glob)
    if warm_df is not None:
        df = pd.merge(df, warm_df, on=SHOT_KEY_COLS, how="left")
        if "warm_solver_ok" in df.columns:
            bad_mask = ~(df["warm_solver_ok"] == True)  # noqa: E712
            for col in ["warm_est_speed", "warm_est_angle", "warm_est_spin", "warm_est_y0"]:
                if col in df.columns:
                    df.loc[bad_mask, col] = np.nan
        n_warm = int(df["warm_est_speed"].notna().sum()) if "warm_est_speed" in df.columns else 0
        print(f"[info] warm-start rows available: {n_warm}")

    keys_all = _gather_keys(df, limit)
    key_filter = _load_key_filter_csv(keys_csv)
    if key_filter is not None:
        keys_all = [
            k for k in keys_all
            if (k.comp, k.sess, k.game, k.end, k.shot) in key_filter
        ]
    print(f"[info] eligible shots with previous row: {len(keys_all)}")
    if not keys_all:
        print("[info] nothing to do")
        return
    print(f"[info] solver method: {solver_method}")
    print(f"[info] loss variant: {loss_variant}")

    chunks = _split_into_chunks(keys_all, chunk_size)
    num_chunks = len(chunks)

    # Discover visible GPUs
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if visible is not None:
        # Respect any user-specified mask
        gpu_ids = [g for g in visible.split(",") if g.strip() != ""]
        num_gpus = len(gpu_ids)
    else:
        # Probe with JAX if available; otherwise fallback to 1
        try:
            import jax  # lightweight import in parent to count devices
            num_gpus = len([d for d in jax.devices() if d.platform == "gpu"])
        except Exception:
            num_gpus = 1

    if num_gpus < 1:
        print("[warn] No GPUs detected by JAX — running single worker on CPU.")
        num_gpus = 1

    print(f"[info] using {num_gpus} worker(s) (1 per GPU)")

    # Work directory for parts
    out_dir = os.path.abspath(f"{out_prefix}.parts")
    os.makedirs(out_dir, exist_ok=True)

    # Process each CHUNK of 5000
    for chunk_idx, keys_chunk in enumerate(chunks):
        print(f"\n[chunk {chunk_idx+1}/{num_chunks}] shots: {len(keys_chunk)}")

        # Round-robin across GPUs
        per_gpu_keys = _round_robin_slices(keys_chunk, num_gpus)

        # Spawn processes (spawn method ensures imports happen after pinning)
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        procs = []

        # For worker convenience (filter to only rows used in this chunk’s comps)
        # Using full df is fine; filtering isn't necessary because _load_prev_next_states
        # already queries by indices. We pass df to each worker independently (copy).
        df_for_workers = df

        for gpu_local_index in range(num_gpus):
            if not per_gpu_keys[gpu_local_index]:
                continue
            gpu_visible_id = str(gpu_ids[gpu_local_index]) if visible is not None else str(gpu_local_index)
            p = ctx.Process(
                target=_worker_run,
                args=(
                    gpu_local_index,
                    gpu_visible_id,
                    per_gpu_keys[gpu_local_index],
                    df_for_workers,
                    out_dir,
                    chunk_idx,
                    0 if base_seed is None else int(base_seed),
                    flush_every,
                    solver_method,
                    loss_variant,
                    sim_cfg,
                    verbose,
                ),
                daemon=False,
            )
            p.start()
            procs.append(p)

        # Wait for workers
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Worker process exited with code {p.exitcode}")

        # Merge part CSVs for this chunk
        part_glob = os.path.join(out_dir, f"chunk{chunk_idx:04d}.gpu*.part*.csv")
        part_files = sorted(glob.glob(part_glob))
        if not part_files:
            print(f"[warn] no parts found for chunk {chunk_idx}")
            continue

        dfs = []
        total_rows = 0
        for pf in part_files:
            dfi = pd.read_csv(pf)
            total_rows += len(dfi)
            dfs.append(dfi)

        merged = pd.concat(dfs, ignore_index=True)
        out_chunk_path = f"{out_prefix}.chunk{chunk_idx:04d}.csv"
        merged.to_csv(out_chunk_path, index=False)
        print(f"[done] merged {len(part_files)} parts => {out_chunk_path} ({len(merged)} rows)")

        # Optional: clean up part files for this chunk to save space
        for pf in part_files:
            try:
                os.remove(pf)
            except Exception:
                pass

    print("\n[all done]")
    print(f"Per-chunk CSVs written with prefix: {out_prefix}.chunkXXXX.csv")
    print(f"Temp parts directory retained at: {out_dir} (safe to delete)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="/mnt/data/curling2/brax/2026/Stones.csv", help="Path to Stones.csv")
    ap.add_argument("--out-prefix", type=str, default="stones_with_estimates", help="Output CSV prefix (chunks will be appended)")
    ap.add_argument("--seed", type=int, default=None, help="Base RNG seed")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N eligible shots")
    ap.add_argument("--chunk-size", type=int, default=5000, help="Shots per chunk")
    ap.add_argument("--flush-every", type=int, default=500, help="Rows per per-GPU partial CSV write")
    ap.add_argument("--warm-start-glob", type=str, default="../inverseDataset_ownership_fixed/stones_with_estimates.chunk*.csv", help="Existing inverse CSV glob used to warm-start ownership-aware refits.")
    ap.add_argument("--keys-csv", type=str, default=None, help="Optional CSV of explicit shot keys to process.")
    ap.add_argument("--solver-method", type=str, default="two_stage_diag", choices=["two_stage_diag", "portfolio", "portfolio_large", "portfolio_large_targeted", "portfolio_large_coord", "portfolio_large_targeted_coord", "portfolio_large_coord_rescue025", "portfolio_large_coord_rescue050", "portfolio_large_refinebank_coord_nm", "portfolio_large_coord_nm_rescue025", "portfolio_large_coord_optrescue025", "portfolio_large_coord_optrescue025_propbank", "portfolio_large_targeted_optrescue025", "portfolio_large_transition_adaptive", "portfolio_rescue", "portfolio_refit_hybrid", "portfolio_large_multistart"], help="Inverse search schedule to use.")
    ap.add_argument("--loss-variant", type=str, default="current", choices=["current", "optimal", "greedy_transition", "optimal_transition", "slot_identity", "slot_transition", "slot_identity_hybrid", "slot_transition_hybrid", "slot_transition_huber", "slot_transition_huber_hybrid"], help="Inverse loss used during search. Output hard_loss_* columns are always evaluated with the current baseline loss.")
    ap.add_argument("--sim-refine-dt", type=float, default=0.02)
    ap.add_argument("--sim-refine-substeps", type=int, default=2)
    ap.add_argument("--sim-coarse-dt", type=float, default=0.03)
    ap.add_argument("--sim-coarse-substeps", type=int, default=1)
    ap.add_argument("--sim-coarse-max-steps", type=int, default=900)
    ap.add_argument("--sim-refine-k-penalty", type=float, default=2.5e4)
    ap.add_argument("--sim-coarse-k-penalty", type=float, default=2.0e4)
    ap.add_argument("--sim-c-damp", type=float, default=CONTACT_MILD_SIM_KWARGS["c_damp"])
    ap.add_argument("--sim-c-damp-sep-frac", type=float, default=CONTACT_MILD_SIM_KWARGS["c_damp_sep_frac"])
    ap.add_argument("--sim-c-tangent", type=float, default=CONTACT_MILD_SIM_KWARGS["c_tangent"])
    ap.add_argument("--sim-mu-tangent", type=float, default=CONTACT_MILD_SIM_KWARGS["mu_tangent"])
    ap.add_argument("--sim-spin-contact", type=float, default=CONTACT_MILD_SIM_KWARGS["spin_contact"])
    ap.add_argument("--sim-k-curl", type=float, default=CONTACT_MILD_SIM_KWARGS["k_curl"])
    ap.add_argument("--sim-a-linear", type=float, default=CONTACT_MILD_SIM_KWARGS["a_linear"])
    ap.add_argument("--sim-gamma-spin", type=float, default=CONTACT_MILD_SIM_KWARGS["gamma_spin"])
    ap.add_argument("--verbose", action="store_true", help="Extra logging from workers")
    args = ap.parse_args()

    sim_cfg = SimConfig(
        refine_dt=float(args.sim_refine_dt),
        refine_substeps=int(args.sim_refine_substeps),
        coarse_dt=float(args.sim_coarse_dt),
        coarse_substeps=int(args.sim_coarse_substeps),
        coarse_max_steps=int(args.sim_coarse_max_steps),
        refine_k_penalty=float(args.sim_refine_k_penalty),
        coarse_k_penalty=float(args.sim_coarse_k_penalty),
        c_damp=float(args.sim_c_damp),
        c_damp_sep_frac=float(args.sim_c_damp_sep_frac),
        c_tangent=float(args.sim_c_tangent),
        mu_tangent=float(args.sim_mu_tangent),
        spin_contact=float(args.sim_spin_contact),
        k_curl=float(args.sim_k_curl),
        a_linear=float(args.sim_a_linear),
        gamma_spin=float(args.sim_gamma_spin),
    )

    main(
        csv_path=args.csv,
        out_prefix=args.out_prefix,
        base_seed=args.seed,
        limit=args.limit,
        chunk_size=int(args.chunk_size),
        flush_every=int(args.flush_every),
        warm_start_glob=args.warm_start_glob,
        keys_csv=args.keys_csv,
        solver_method=str(args.solver_method),
        loss_variant=str(args.loss_variant),
        sim_cfg=sim_cfg,
        verbose=bool(args.verbose),
    )
