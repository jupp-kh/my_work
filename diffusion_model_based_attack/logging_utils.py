from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, row: dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


class TensorBoardLogger:
    def __init__(self, log_dir: Path, enabled: bool = True) -> None:
        self.log_dir = log_dir
        self.writer = None
        if not enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            print("TensorBoard is not installed; continuing without TensorBoard logs.")
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    @property
    def enabled(self) -> bool:
        return self.writer is not None

    def add_scalar(self, name: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(name, value, step)

    def add_text(self, name: str, value: str, step: int = 0) -> None:
        if self.writer is not None:
            self.writer.add_text(name, value, step)

    def add_image(self, name: str, image: torch.Tensor, step: int) -> None:
        if self.writer is not None:
            self.writer.add_image(name, image.detach().cpu().clamp(0.0, 1.0), step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

