from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .config import CERTPHASH_ATTACK_ROOT, DEFAULT_CERTPHASH_MODEL, IMAGENET_MEAN, IMAGENET_STD
except ImportError:  # pragma: no cover - script execution fallback.
    from config import CERTPHASH_ATTACK_ROOT, DEFAULT_CERTPHASH_MODEL, IMAGENET_MEAN, IMAGENET_STD


@dataclass(frozen=True)
class HashDetails:
    quantized: list[int]
    bit_string: str
    base64: str


@dataclass(frozen=True)
class TargetHash:
    quantized: torch.Tensor
    details: HashDetails
    source: str


def import_certphash_symbols() -> tuple[Any, Any]:
    if str(CERTPHASH_ATTACK_ROOT) not in sys.path:
        sys.path.insert(0, str(CERTPHASH_ATTACK_ROOT))
    try:
        from models.resnet_v5 import resnet_v5
        from utils.hashing import compute_hash_coco
    except Exception as exc:  # pragma: no cover - exercised in the real environment.
        raise ImportError(
            f"Could not import CertPhash implementation from {CERTPHASH_ATTACK_ROOT}. "
            "Expected models.resnet_v5 and utils.hashing.compute_hash_coco."
        ) from exc
    return resnet_v5, compute_hash_coco


def clean_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            checkpoint = checkpoint["state_dict"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            checkpoint = checkpoint["model"]
    if not isinstance(checkpoint, dict):
        raise TypeError("CertPhash checkpoint must be a state dict or contain a state dict.")
    return {str(key).removeprefix("module."): value for key, value in checkpoint.items()}


class CertPhashWrapper:
    """Repository CertPhash model with continuous features and exact byte-hash evaluation."""

    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CERTPHASH_MODEL,
        device: torch.device | str = "cuda:0",
        model_input_size: int = 64,
        hash_length: int = 144,
    ) -> None:
        resnet_v5, compute_hash_coco = import_certphash_symbols()
        self.compute_hash_coco = compute_hash_coco
        self.device = torch.device(device)
        self.model_input_size = model_input_size
        self.hash_length = hash_length
        self.model = resnet_v5(input_dim=model_input_size)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(clean_state_dict(checkpoint))
        self.model.to(device=self.device, dtype=torch.float32)
        self.model.eval()
        self.model.requires_grad_(False)
        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=self.device, dtype=torch.float32).view(1, 3, 1, 1)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(device=self.device, dtype=torch.float32)
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("CertPhash expects images shaped [batch, 3, height, width].")
        if images.shape[-2:] != (self.model_input_size, self.model_input_size):
            images = F.interpolate(
                images,
                size=(self.model_input_size, self.model_input_size),
                mode="bilinear",
                align_corners=False,
            )
        return (images - self.mean) / self.std

    def continuous_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(self.preprocess(images))

    @staticmethod
    def quantize_features(features: torch.Tensor) -> torch.Tensor:
        return torch.relu(torch.round(features))

    def exact_hash_tensor(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.quantize_features(self.continuous_features(images.detach())).detach()

    def target_from_image(self, image: torch.Tensor, source: str) -> TargetHash:
        quantized = self.exact_hash_tensor(image).view(-1)
        if quantized.numel() != self.hash_length:
            raise ValueError(f"Expected a {self.hash_length}-byte CertPhash output, got {quantized.numel()}.")
        return TargetHash(quantized=quantized, details=self.hash_details(quantized), source=source)

    def hash_details(self, quantized: torch.Tensor) -> HashDetails:
        hash_bytes, bit_string, encoded = self.compute_hash_coco(quantized.detach().view(-1))
        values = np.asarray(hash_bytes).reshape(-1).astype(np.uint8)
        return HashDetails(
            quantized=[int(value) for value in values.tolist()],
            bit_string=str(bit_string),
            base64=str(encoded),
        )

    def surrogate_loss(
        self,
        features: torch.Tensor,
        target_quantized: torch.Tensor,
        loss_type: Literal["smooth_l1", "l1", "mse"] = "smooth_l1",
        hash_scale: float = 255.0,
        smooth_l1_beta: float = 0.02,
    ) -> torch.Tensor:
        target = target_quantized.to(device=features.device, dtype=features.dtype).view(1, -1).expand_as(features)
        current = features / hash_scale
        target = target / hash_scale
        if loss_type == "smooth_l1":
            return F.smooth_l1_loss(current, target, beta=smooth_l1_beta)
        if loss_type == "l1":
            return F.l1_loss(current, target)
        if loss_type == "mse":
            return F.mse_loss(current, target)
        raise ValueError(f"Unsupported hash surrogate loss: {loss_type}")

    @staticmethod
    def exact_hash_l1(quantized: torch.Tensor, target_quantized: torch.Tensor) -> torch.Tensor:
        target = target_quantized.to(device=quantized.device, dtype=quantized.dtype).view(1, -1)
        return torch.sum(torch.abs(quantized.view(quantized.shape[0], -1) - target), dim=1)
