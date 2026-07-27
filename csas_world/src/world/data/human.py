"""Human behaviour-cloning buffer + offline value buffer.

These reuse the *exact* csas data pipelines so the policy/value heads train on the
same data as the canonical baselines (apples-to-apples comparison):

  * ``HumanPolicyDataset``  -> (prev_state, cond) -> human throw   [policy BC NLL]
  * ``ValueStateDataset``   -> (state, cond)      -> end-score diff [value reg]
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..replay.schema import SOURCE_HUMAN, SOURCE_VALUE, empty_record


# --------------------------------------------------------------------------- #
# Human policy (behaviour cloning)
# --------------------------------------------------------------------------- #
def load_human_policy_tensors(csas_v3_root: str, holdout: int = 0, split: str = "train",
                              max_loss: float = 0.08) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                               np.ndarray, np.ndarray]:
    """Returns x[N,24], c[N,3], a_raw[N,4], shots_in_end[N], shot_index[N]."""
    from csas.policy_dataset import build_policy_tensors

    root = Path(csas_v3_root)
    default_inverse = root / "data" / "processed" / "inverse_realistic.csv"
    spin20_inverse = root / "data" / "processed" / "inverse_realistic_spin20.csv"
    inverse_glob = os.environ.get(
        "CSAS_WORLD_HUMAN_INVERSE_GLOB",
        str(spin20_inverse if spin20_inverse.exists() else default_inverse),
    )
    x, c, a, _Y, meta = build_policy_tensors(
        fixed_root=root / "data" / "raw",
        inverse_glob=inverse_glob,
        holdout=holdout, split=split, max_loss=max_loss,
    )
    x = x.numpy().astype(np.float32)
    c = c.numpy().astype(np.float32)
    a = a.numpy().astype(np.float32)
    sie = meta["ShotsInEnd"].to_numpy(dtype=np.float32)
    si = meta["ShotIndex"].to_numpy(dtype=np.float32)
    return x, c, a, sie, si


class HumanPolicyDataset(Dataset):
    def __init__(self, csas_v3_root: str, K: int, M: int, holdout: int = 0,
                 split: str = "train", max_loss: float = 0.08):
        self.K, self.M = K, M
        self.x, self.c, self.a, sie, si = load_human_policy_tensors(
            csas_v3_root, holdout, split, max_loss)
        self.horizon = np.clip((sie - si), 1, 10).astype(np.int64)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        r = empty_record(self.K, self.M)
        r["x0"] = self.x[idx]
        r["c0"] = self.c[idx]
        r["bc_action_raw"] = self.a[idx]
        r["bc_mask"] = np.float32(1.0)
        r["horizon"] = np.int64(self.horizon[idx])
        r["source"] = np.int64(SOURCE_HUMAN)
        return r


# --------------------------------------------------------------------------- #
# Offline value targets (end-score differential)
# --------------------------------------------------------------------------- #
class ValueStateDataset(Dataset):
    def __init__(self, stones_csv: str, ends_csv: str, K: int, M: int,
                 holdout: int = 0, split: str = "train", split_seed: int = 123,
                 val_end_frac: float = 0.10, max_rows: Optional[int] = None):
        from csas.dataset import ValueDataset
        from csas.splits import make_holdout_split, materialize

        ds = ValueDataset(stones_csv, ends_csv, augment_positions=False, augment_flip=False)
        Xp, Xc, Y = materialize(ds)
        if split == "all":
            # single-competition / synthetic sources cannot be holdout-split
            idx = np.arange(len(ds.df))
        else:
            tr, va, te, _ = make_holdout_split(ds.df, holdout, val_end_frac, split_seed)
            idx = {"train": tr, "val": va, "test": te}[split]
        if max_rows is not None and len(idx) > max_rows:
            rng = np.random.default_rng(0)
            idx = rng.choice(idx, size=max_rows, replace=False)
        self.x = Xp[idx].numpy().astype(np.float32)
        self.c = Xc[idx].numpy().astype(np.float32)
        self.v = Y[idx].numpy().astype(np.float32).reshape(-1)
        self.K, self.M = K, M

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        r = empty_record(self.K, self.M)
        r["x0"] = self.x[idx]
        r["c0"] = self.c[idx]
        r["value_target"][0] = self.v[idx]
        r["value_mask"][0] = 1.0
        r["outcome_margin"] = np.float32(self.v[idx])
        r["outcome_mask"] = np.float32(1.0)
        r["source"] = np.int64(SOURCE_VALUE)
        return r


__all__ = ["HumanPolicyDataset", "ValueStateDataset", "load_human_policy_tensors"]
