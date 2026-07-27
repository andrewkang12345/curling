from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict


class JsonlLogger:
    """Minimal append-only JSONL metrics logger (main rank only)."""

    def __init__(self, path: str, enabled: bool = True):
        self.enabled = enabled
        self.path = Path(path)
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict) -> None:
        if not self.enabled:
            return
        record = {"t": round(time.time(), 3), **record}
        with open(self.path, "a") as fh:
            fh.write(json.dumps(record) + "\n")


def fmt_metrics(metrics: Dict[str, float]) -> str:
    return " ".join(f"{k}={v:.4f}" for k, v in metrics.items())


__all__ = ["JsonlLogger", "fmt_metrics"]
