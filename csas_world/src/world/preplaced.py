"""Pre-placed mixed-doubles start-of-end states (standard / pp_left / pp_right).

The human policy data covers ``throws_remaining`` 1..9 only: the annotators skipped the FIRST
thrown stone (ShotID 7, throws_remaining 10), treating the pre-placed configuration as trivial.
Those are exactly the missing h10 roots -- the start of the end, with two pre-placed stones (a
guard + an in-house stone) already on the ice, before anyone throws. This module materialises those
canonical states as csas_world ``Root`` / ``H2HRoot`` objects so the deepest curriculum horizon
(h10) has real roots instead of degenerate nearest-horizon fallbacks.

Canonical positions are physical metres ``[along=distance-from-tee, lateral=side-offset]`` and are
therefore sheet-geometry-agnostic; we map them to the normalized 24-vector via csas's full-sheet
``compact_m_to_raw`` (which writes dead slots as ``POS_MAX`` -> norm 1.0, matching the human-state
convention; ``_live_mask_from_state`` reads both 0 and POS_MAX as dead, but POS_MAX is the canonical
encoding the world model trained on). Ported from csas_fixed_moreMCTS / ``csas.preplaced_value_data``.
"""
from __future__ import annotations

from typing import List

import numpy as np

from csas.common import NUM_STONES, POS_MAX, compact_m_to_raw

# Canonical [along, lateral] metres per pre-placement mode (mixed doubles): a guard in front of the
# tee and one stone in the house, mirrored left/right for pp_left / pp_right.
_CANON = {
    "standard": {"guard": (-3.4016, 0.0), "inhouse": (0.4572, 0.0)},
    "pp_right": {"guard": (-3.4016, 1.0333), "inhouse": (-0.1524, 1.2192)},
    "pp_left": {"guard": (-3.4016, -1.0333), "inhouse": (-0.1524, -1.2192)},
}
MODES = ("standard", "pp_left", "pp_right")
PREPLACED_SHOTS_IN_END = 10   # standard mixed-doubles end = 10 thrown stones
PREPLACED_HORIZON = 10        # start-of-end first thrown stone: throws_remaining = clip(10-0) = 10


def board_norm(mode: str, guard_slot: int) -> np.ndarray:
    """Normalized 24-vector for a (mode, guard_slot): 2 live pre-placed stones, rest dead (POS_MAX).

    guard_slot in {1, 7}; slots 1-6 are block-0 stones, slots 7-12 block-1. The guard belongs to
    the slot's block; the in-house stone is the other block's slot (7 if guard_slot==1 else 1)."""
    m = np.full((NUM_STONES, 2), np.nan, dtype=np.float32)   # NaN -> dead -> POS_MAX in compact_m_to_raw
    inhouse_slot = 7 if int(guard_slot) == 1 else 1
    canon = _CANON[str(mode)]
    m[int(guard_slot) - 1] = np.asarray(canon["guard"], dtype=np.float32)
    m[inhouse_slot - 1] = np.asarray(canon["inhouse"], dtype=np.float32)
    raw = compact_m_to_raw(m)                                 # (12,2) raw; dead slots = POS_MAX
    return (raw.reshape(-1) / POS_MAX).astype(np.float32)


def _preplaced_rows(split: str, seed: int, max_roots: int, num_shards: int, shard_id: int,
                    balance: bool = False):
    """Subset of the pre-placed first-shot data (mode, guard_slot, thrower_block).

    Holdout split matches the value/human convention: CompetitionID==0 is val/test, others train.

    ``balance``: the h10 states are CANONICAL -- ``board_norm`` yields only ~6 distinct states
    (3 modes x 2 guard_slots), so the data's 77/11/12 mode skew is NOT a data constraint and should
    NOT carry into collection. With balance=True, sample EQUALLY across the (mode, guard_slot) groups
    so pp_left/pp_right get the same search budget as standard (each shard independently balanced via
    seed+shard_id; duplicates across shards are fine -- they are distinct noisy collection runs)."""
    import pandas as pd
    from csas.preplaced_value_data import load_preplaced_training_frame

    df = load_preplaced_training_frame()
    comp = df["CompetitionID"].astype(int).to_numpy()
    df = df[(comp == 0) if split == "val" else (comp != 0)].copy()
    rng = np.random.default_rng(int(seed) + 7919 * int(shard_id))
    if balance:
        df["_g"] = df["mode"].astype(str) + ":" + df["guard_slot"].astype(int).astype(str)
        groups = sorted(df["_g"].unique())
        per = max(1, -(-int(max_roots) // len(groups)))   # ceil(max_roots / n_groups)
        picks = []
        for g in groups:
            sub = df[df["_g"] == g]
            picks.append(sub.iloc[rng.integers(0, len(sub), size=per)])   # equal per group (w/ replacement)
        out = pd.concat(picks, ignore_index=True)
        return out.iloc[rng.permutation(len(out))][:max_roots]
    idx = rng.permutation(len(df))
    if num_shards > 1:
        idx = idx[shard_id::num_shards]
    return df.iloc[idx[:max_roots]]


def build_preplaced_roots(horizon: int, max_roots: int, split: str = "train", seed: int = 0,
                          num_shards: int = 1, shard_id: int = 0, balance: bool = True) -> List:
    """Pre-placed start-of-end ``Root`` objects (the throws_remaining==10 collection stage).

    balance=True (default): equal collection budget per (mode, guard_slot) -> pp_left/right are NOT
    under-collected relative to standard (the states are canonical, so this is free; see _preplaced_rows)."""
    from .search.collect import Root

    rows = _preplaced_rows(split, seed, max_roots, num_shards, shard_id, balance=balance)
    roots: List = []
    for _, row in rows.iterrows():
        block = int(round(float(row["thrower_block"])))
        roots.append(Root(
            x=board_norm(str(row["mode"]), int(row["guard_slot"])),
            c=np.asarray([0.0, 0.0, float(block)], dtype=np.float32),
            shots_in_end=PREPLACED_SHOTS_IN_END,
            perspective_block=block,
            horizon=int(horizon),
        ))
    return roots


def build_preplaced_h2h_roots(horizon: int, n_roots: int, split: str = "val", seed: int = 0) -> List:
    """Pre-placed start-of-end ``H2HRoot`` objects for head-to-head evaluation at h10."""
    from .eval.head_to_head import H2HRoot

    rows = _preplaced_rows(split, seed, n_roots, num_shards=1, shard_id=0)
    roots: List = []
    for _, row in rows.iterrows():
        block = int(round(float(row["thrower_block"])))
        roots.append(H2HRoot(
            board_norm(str(row["mode"]), int(row["guard_slot"])),
            np.asarray([0.0, 0.0, float(block)], dtype=np.float32),
            PREPLACED_SHOTS_IN_END,
            int(horizon),
        ))
    return roots
