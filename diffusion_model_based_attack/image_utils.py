from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import make_grid, save_image


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def load_image_tensor(path: Path, device: torch.device | str = "cpu", size: int | None = None) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if size is not None:
        image = image.resize((size, size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def resize_tensor(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if image.shape[-2:] == size:
        return image
    return F.interpolate(image, size=size, mode="bilinear", align_corners=False)


def save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(tensor.detach().cpu().clamp(0.0, 1.0), str(path))


def save_tensor_grid(tensor: torch.Tensor, path: Path, nrow: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    batch_size = tensor.shape[0]
    grid = make_grid(
        tensor.detach().cpu().clamp(0.0, 1.0),
        nrow=nrow or min(batch_size, 4),
        padding=2,
    )
    save_image(grid, str(path))


def image_grid(tensor: torch.Tensor, nrow: int | None = None) -> torch.Tensor:
    batch_size = tensor.shape[0]
    return make_grid(
        tensor.detach().cpu().clamp(0.0, 1.0),
        nrow=nrow or min(batch_size, 4),
        padding=2,
    )


def safe_stem(value: str) -> str:
    stem = Path(value).stem if value else "target"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)[:80] or "target"


def timestamped_run_name(target_label: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{safe_stem(target_label)}"

