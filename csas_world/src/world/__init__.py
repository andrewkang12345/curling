"""csas_world -- EfficientZero-style multi-head graph-transformer for curling.

Importing ``world`` wires up the canonical ``csas`` dependency and the graph
feature environment (see :mod:`world._bootstrap`).
"""
from __future__ import annotations

from . import _bootstrap  # noqa: F401  (side effects: sys.path + env vars)
from ._bootstrap import CSAS_V3_ROOT, GNN_FEATURE_ENV

__version__ = "0.1.0"
__all__ = ["CSAS_V3_ROOT", "GNN_FEATURE_ENV", "__version__"]
