"""Bootstrap the csas_world package.

This MUST be imported before any ``csas.gnn_models`` / ``csas.policy_graph_model``
import, because those modules read the ``GNN_*`` graph-feature environment
variables at import time to size their edge/node feature tensors.  The canonical
GraphTF trunk + full-covariance prior were trained with the configuration set
below; using anything else silently breaks weight loading.

It also makes the canonical ``csas`` package (which lives in ``csas_v3/src`` and
is *not* pip-installed) importable, and selects the CPU backend for JAX because
the simulator cannot JIT on GPU in the current instance image (cuDNN/plugin
mismatch -- see memory ``curling-jax-gpu-broken``).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# --- canonical graph-feature env (trunk + prior were trained with this) -------
GNN_FEATURE_ENV = {
    "GNN_EDGE_SCALAR_MODE": "button_visible_plus_curl_arc_reach_with_outgoing",
    "GNN_NODE_FEATURE_MODE": "none",
    "GNN_RELEASE_NODE_MODE": "three_plus_takeout_boundary",
    "GNN_EDGE_PRUNE_MODE": "none",
}
for _k, _v in GNN_FEATURE_ENV.items():
    os.environ.setdefault(_k, _v)

# --- make the canonical csas package importable -------------------------------
CSAS_V3_ROOT = Path(os.environ.get("CSAS_V3_ROOT", "/mnt/data/curling2/csas_v3"))
_csas_src = CSAS_V3_ROOT / "src"
if _csas_src.is_dir() and str(_csas_src) not in sys.path:
    sys.path.insert(0, str(_csas_src))

# --- the JAX simulator must run on CPU here (see memory note) ------------------
# Only force CPU if the caller has not already chosen a platform.  Training never
# touches JAX; only collection / eval do, and those want CPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# Keep JAX from grabbing host RAM aggressively when it does init.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def assert_csas_importable() -> None:
    import importlib

    importlib.import_module("csas")


__all__ = ["GNN_FEATURE_ENV", "CSAS_V3_ROOT", "assert_csas_importable"]
