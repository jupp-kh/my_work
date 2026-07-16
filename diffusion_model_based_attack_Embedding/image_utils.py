from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def load_image_tensor(path: Path, device: torch.device, size: tuple[int, int] | None = None) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if size is not None:
        image = image.resize((size[1], size[0]), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu().clamp(0.0, 1.0)
    if image.ndim == 4:
        image = image[0]
    array = image.permute(1, 2, 0).numpy()
    return np.rint(array * 255.0).clip(0, 255).astype(np.uint8)


def save_image_tensor(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_uint8_image(tensor)).save(path)


def make_grid(images: torch.Tensor, nrow: int | None = None, padding: int = 2) -> torch.Tensor:
    images = images.detach().cpu().clamp(0.0, 1.0)
    if images.ndim == 3:
        return images
    batch, channels, height, width = images.shape
    nrow = nrow or min(batch, 4)
    ncol = int(math.ceil(batch / nrow))
    grid = torch.ones(
        channels,
        ncol * height + padding * max(0, ncol - 1),
        nrow * width + padding * max(0, nrow - 1),
        dtype=images.dtype,
    )
    for index, image in enumerate(images):
        row = index // nrow
        col = index % nrow
        y0 = row * (height + padding)
        x0 = col * (width + padding)
        grid[:, y0 : y0 + height, x0 : x0 + width] = image
    return grid


def save_image_grid(images: torch.Tensor, path: Path) -> None:
    save_image_tensor(make_grid(images), path)


def exact_rgb_l1_distance(generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    generated_255 = torch.round(generated.detach().clamp(0.0, 1.0) * 255.0)
    target_255 = torch.round(target.detach().clamp(0.0, 1.0) * 255.0)
    if target_255.shape[0] == 1 and generated_255.shape[0] > 1:
        target_255 = target_255.expand_as(generated_255)
    return torch.sum(torch.abs(generated_255 - target_255), dim=(1, 2, 3))


def mean_abs_pixel_difference_from_l1(rgb_l1: float, height: int, width: int) -> float:
    return float(rgb_l1) / float(3 * height * width)


def image_range_loss(images: torch.Tensor) -> torch.Tensor:
    return F.relu(-images).mean() + F.relu(images - 1.0).mean()
