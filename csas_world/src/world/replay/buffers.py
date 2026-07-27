"""Replay buffers and the DDP-friendly mixed sampler.

Three source families:
  * human / value : map-style datasets built from csas data (see world.data.human)
  * sim / mcts    : ``.npz`` shards written by the collectors (world.search.collect)

``MixedReplay`` is a map-style dataset of fixed virtual length that, per access,
samples a source by weight and a random record from it.  Being map-style it
composes with ``DistributedSampler`` for clean multi-GPU sharding; an explicit
``set_epoch`` reseeds so every epoch sees fresh draws from the large pools.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import schema


class RecordArrayDataset(Dataset):
    """Wraps a dict of stacked arrays {field: [N, *shape]} as record dicts."""

    def __init__(self, arrays: Dict[str, np.ndarray]):
        self.arrays = arrays
        self.n = len(next(iter(arrays.values())))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return {k: v[idx] for k, v in self.arrays.items()}


def load_shards(shard_dir: str, K: int, M: int) -> Optional[RecordArrayDataset]:
    """Concatenate all ``*.npz`` shards in a directory into one dataset."""
    d = Path(shard_dir)
    files = sorted(d.rglob("*.npz"))   # recurse into per-horizon subdirs
    if not files:
        return None
    fields = list(schema.field_shapes(K, M).keys())
    bufs: Dict[str, List[np.ndarray]] = {f: [] for f in fields}
    for fp in files:
        with np.load(fp) as z:
            for f in fields:
                if f in z:
                    bufs[f].append(z[f])
    arrays = {}
    n_ref = None
    for f in fields:
        if bufs[f]:
            arrays[f] = np.concatenate(bufs[f], axis=0)
            n_ref = len(arrays[f]) if n_ref is None else n_ref
    if n_ref is None:
        return None
    # fill any field absent from shards with zeros (mask fields stay 0 -> ignored)
    shapes = schema.field_shapes(K, M)
    for f in fields:
        if f not in arrays:
            dt = np.int64 if f in schema.INT_FIELDS else np.float32
            arrays[f] = np.zeros((n_ref, *shapes[f]), dtype=dt)
    return RecordArrayDataset(arrays)


def save_shard(path: str, records: Sequence[Dict[str, np.ndarray]]) -> int:
    if not records:
        return 0
    stacked = {k: np.stack([r[k] for r in records], axis=0) for k in records[0].keys()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **stacked)
    return len(records)


class MixedReplay(Dataset):
    """Weighted mixture over named sub-datasets, fixed virtual length."""

    def __init__(self, sources: Dict[str, Tuple[Dataset, float]], virtual_len: int,
                 seed: int = 0):
        # drop empty / zero-weight sources
        self.names: List[str] = []
        self.datasets: List[Dataset] = []
        weights: List[float] = []
        for name, (ds, w) in sources.items():
            if ds is None or len(ds) == 0 or w <= 0:
                continue
            self.names.append(name)
            self.datasets.append(ds)
            weights.append(float(w))
        if not self.datasets:
            raise ValueError("MixedReplay: no non-empty sources")
        w = np.array(weights, dtype=np.float64)
        self.weights = w / w.sum()
        self.virtual_len = int(virtual_len)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def composition(self) -> Dict[str, float]:
        return {n: float(w) for n, w in zip(self.names, self.weights)}

    def __len__(self) -> int:
        return self.virtual_len

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng((self.seed * 1_000_003 + self.epoch * 9_176 + idx) & 0xFFFFFFFF)
        s = int(rng.choice(len(self.datasets), p=self.weights))
        ds = self.datasets[s]
        row = int(rng.integers(len(ds)))
        return ds[row]


def make_loader(dataset: Dataset, batch_size: int, distributed: bool, rank: int = 0,
                world_size: int = 1, num_workers: int = 4, shuffle: bool = True,
                seed: int = 0) -> Tuple[DataLoader, Optional[object]]:
    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=shuffle, seed=seed, drop_last=True)
    loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        shuffle=(shuffle and sampler is None), drop_last=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0),
        collate_fn=schema.collate,
    )
    return loader, sampler


__all__ = ["RecordArrayDataset", "MixedReplay", "load_shards", "save_shard", "make_loader"]
