from __future__ import annotations

import base64
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
CERTPHASH_ATTACK_ROOT = REPO_ROOT / "CertPhash" / "attack"
DEFAULT_CERTPHASH_MODEL = (
    REPO_ROOT
    / "CertPhash"
    / "train_verify"
    / "saved_models"
    / "coco_photodna_ep8"
    / "ckpt_best.pth"
)

if str(CERTPHASH_ATTACK_ROOT) not in sys.path:
    sys.path.insert(0, str(CERTPHASH_ATTACK_ROOT))

from models.resnet_v5 import resnet_v5  # noqa: E402
from utils.hashing import compute_hash_coco  # noqa: E402


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def _clean_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            checkpoint = checkpoint["state_dict"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            checkpoint = checkpoint["model"]

    if not isinstance(checkpoint, dict):
        raise TypeError("CertPhash checkpoint must be a state dict or contain a state dict.")

    return {str(key).removeprefix("module."): value for key, value in checkpoint.items()}


def _parse_hash_values(hash_value: str, expected_length: int) -> torch.Tensor:
    cleaned = hash_value.strip()
    try:
        decoded = base64.b64decode(cleaned, validate=True)
        values = np.frombuffer(decoded, dtype=np.uint8)
        if values.size == expected_length:
            return torch.tensor(values.astype(np.float32))
    except Exception:
        pass

    parts = [part for part in re.split(r"[\s,;\[\]()]+", cleaned) if part]
    try:
        values = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError("Target hash must be base64 or a list of integer byte values.") from exc

    if len(values) != expected_length:
        raise ValueError(f"Expected {expected_length} hash bytes, got {len(values)}.")
    if any(value < 0 or value > 255 for value in values):
        raise ValueError("Hash byte values must be in [0, 255].")
    return torch.tensor(values, dtype=torch.float32)


class CertPhashWrapper:
    """Thin wrapper around the repo's differentiable CertPhash model and discrete hash."""

    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CERTPHASH_MODEL,
        device: torch.device | str = "cuda:0",
        model_input_size: int = 64,
        hash_length: int = 144,
    ) -> None:
        self.device = torch.device(device)
        self.model_input_size = model_input_size
        self.hash_length = hash_length
        self.model = resnet_v5(input_dim=model_input_size)

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(_clean_state_dict(checkpoint))
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device, dtype=torch.float32)
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected images with shape [batch, 3, height, width].")
        if images.shape[-2:] != (self.model_input_size, self.model_input_size):
            images = F.interpolate(
                images,
                size=(self.model_input_size, self.model_input_size),
                mode="bilinear",
                align_corners=False,
            )
        return (images - self.mean) / self.std

    def logits(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(self.preprocess(images))

    @staticmethod
    def quantize_logits(logits: torch.Tensor) -> torch.Tensor:
        return torch.relu(torch.round(logits))

    def quantized_hash(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.quantize_logits(self.logits(images))

    def target_from_image(self, image: torch.Tensor, source: str) -> TargetHash:
        quantized = self.quantized_hash(image).view(-1)
        if quantized.numel() != self.hash_length:
            raise ValueError(f"Expected one target image producing {self.hash_length} hash bytes.")
        return TargetHash(
            quantized=quantized.detach(),
            details=self.hash_details(quantized),
            source=source,
        )

    def target_from_hash(self, hash_value: str, source: str = "target_hash") -> TargetHash:
        quantized = _parse_hash_values(hash_value, self.hash_length).to(self.device)
        return TargetHash(
            quantized=quantized.detach(),
            details=self.hash_details(quantized),
            source=source,
        )

    def surrogate_loss(
        self,
        logits: torch.Tensor,
        target_quantized: torch.Tensor,
        loss_type: str = "smooth_l1",
        hash_scale: float = 255.0,
        smooth_l1_beta: float = 0.02,
    ) -> torch.Tensor:
        target = target_quantized.to(logits.device, dtype=logits.dtype).view(1, -1).expand_as(logits)
        current = logits / hash_scale
        target = target / hash_scale

        if loss_type == "l1":
            return F.l1_loss(current, target)
        if loss_type == "mse":
            return F.mse_loss(current, target)
        if loss_type == "smooth_l1":
            return F.smooth_l1_loss(current, target, beta=smooth_l1_beta)
        raise ValueError(f"Unsupported hash loss: {loss_type}")

    def l1_distance_per_sample(
        self,
        quantized: torch.Tensor,
        target_quantized: torch.Tensor,
    ) -> torch.Tensor:
        target = target_quantized.to(quantized.device, dtype=quantized.dtype).view(1, -1)
        return torch.sum(torch.abs(quantized.view(quantized.shape[0], -1) - target), dim=1)

    def logit_l1_distance_per_sample(
        self,
        logits: torch.Tensor,
        target_quantized: torch.Tensor,
    ) -> torch.Tensor:
        target = target_quantized.to(logits.device, dtype=logits.dtype).view(1, -1)
        return torch.sum(torch.abs(logits.view(logits.shape[0], -1) - target), dim=1)

    def hash_details(self, quantized: torch.Tensor) -> HashDetails:
        hash_bytes, bit_string, encoded = compute_hash_coco(quantized.detach().view(-1))
        values = np.asarray(hash_bytes).reshape(-1).astype(np.uint8)
        return HashDetails(
            quantized=[int(value) for value in values.tolist()],
            bit_string=bit_string,
            base64=encoded,
        )
