"""DDP helpers.

The instance image crashes on NCCL, so the default backend is ``gloo`` (this is
what the canonical full-covariance prior's 4-GPU run used).  Each rank still
binds its own CUDA device; gloo only carries the gradient all-reduce.
"""
from __future__ import annotations

import os
from typing import List

import torch
import torch.distributed as dist


def ddp_setup(rank: int, world_size: int, gpus: List[int], backend: str = "gloo",
              master_port: str = "29521") -> torch.device:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", master_port)
    if world_size > 1:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    dev = torch.device(f"cuda:{gpus[rank]}" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    return dev


def ddp_cleanup(world_size: int) -> None:
    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def all_reduce_mean(value: float, device: torch.device, world_size: int) -> float:
    if world_size <= 1 or not dist.is_initialized():
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / world_size)


__all__ = ["ddp_setup", "ddp_cleanup", "is_main", "all_reduce_mean"]
