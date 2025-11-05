# gpu_psro/io_utils.py
from __future__ import annotations
import os, json
from typing import Dict, Any
import numpy as np
import torch

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _unwrap_module(model: torch.nn.Module) -> torch.nn.Module:
    # Handle DataParallel / DistributedDataParallel
    return model.module if hasattr(model, "module") else model

def save_policy_state_dict(model: torch.nn.Module, path: str):
    ensure_dir(os.path.dirname(path))
    base = _unwrap_module(model)
    torch.save(base.state_dict(), path)

def load_policy_state_dict(model: torch.nn.Module, path: str, map_location: str = "cpu"):
    sd = torch.load(path, map_location=map_location)
    model.load_state_dict(sd)
    return model

def export_torchscript(model: torch.nn.Module, example_input: torch.Tensor, path: str):
    ensure_dir(os.path.dirname(path))
    base = _unwrap_module(model).eval()
    ts = torch.jit.trace(base, example_input)
    ts.save(path)

def save_json(obj: Dict[str, Any], path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def save_numpy(arr: np.ndarray, path: str):
    ensure_dir(os.path.dirname(path))
    np.save(path, arr)

def append_event_log(path: str, entry: Dict[str, Any]):
    ensure_dir(os.path.dirname(path))
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            data.append(entry)
        else:
            data = [data, entry]
    else:
        data = [entry]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
