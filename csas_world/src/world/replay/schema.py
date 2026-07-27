"""Fixed-shape replay record schema shared by all buffers.

Every buffer (human / sim-transition / MCTS) emits records with the SAME keys
and shapes so they can be concatenated into one mixed dataset and collated into
a single fixed-shape batch.  Absent targets are zero-filled and flagged by their
``*_mask`` field, so the loss masks them out cleanly.

K = unroll steps, M = number of distillation candidate actions (soft_topk).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

SOURCE_HUMAN = 0
SOURCE_SIM = 1
SOURCE_MCTS = 2
SOURCE_VALUE = 3
SOURCE_NAMES = {0: "human", 1: "sim", 2: "mcts", 3: "value"}

# scalar fields are stored shape () ; the rest as below (per-record, no batch dim)
FLOAT_FIELDS = (
    "x0", "c0", "a_raw", "next_states", "next_conds", "next_live",
    "value_target", "value_mask", "reward_target", "reward_mask",
    "outcome_margin", "outcome_mask", "consistency_mask",
    "bc_action_raw", "bc_mask", "dist_actions_raw", "dist_weights", "dist_mask",
)
INT_FIELDS = ("horizon", "source")


def field_shapes(K: int, M: int) -> Dict[str, tuple]:
    return {
        "x0": (24,), "c0": (3,),
        "a_raw": (K, 4),
        "next_states": (K, 24), "next_conds": (K, 3), "next_live": (K, 12),
        "value_target": (K + 1,), "value_mask": (K + 1,),
        "reward_target": (K,), "reward_mask": (K,),
        "outcome_margin": (), "outcome_mask": (),
        "consistency_mask": (K,),
        "bc_action_raw": (4,), "bc_mask": (),
        "dist_actions_raw": (M, 4), "dist_weights": (M,), "dist_mask": (),
        "horizon": (), "source": (),
    }


def empty_record(K: int, M: int) -> Dict[str, np.ndarray]:
    rec: Dict[str, np.ndarray] = {}
    for name, shape in field_shapes(K, M).items():
        dt = np.int64 if name in INT_FIELDS else np.float32
        rec[name] = np.zeros(shape, dtype=dt)
    return rec


def collate(records: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    keys = records[0].keys()
    for k in keys:
        arr = np.stack([r[k] for r in records], axis=0)
        out[k] = torch.from_numpy(arr)
    return out


def to_device(batch: Dict[str, torch.Tensor], device, non_blocking: bool = True) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}


__all__ = [
    "SOURCE_HUMAN", "SOURCE_SIM", "SOURCE_MCTS",
    "FLOAT_FIELDS", "INT_FIELDS", "field_shapes", "empty_record", "collate", "to_device",
]
