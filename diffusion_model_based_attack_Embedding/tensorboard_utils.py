from __future__ import annotations

import torch

try:
    from .diffusion_wrapper import SDXLTurboEmbeddingExperiment
    from .image_utils import make_grid
except ImportError:  # pragma: no cover - script execution fallback.
    from diffusion_wrapper import SDXLTurboEmbeddingExperiment
    from image_utils import make_grid


def log_tensorboard_images(
    writer: object,
    step: int,
    *,
    target_image: torch.Tensor,
    current_images: torch.Tensor,
    best_image: torch.Tensor | None,
) -> None:
    writer.add_image("images/target", target_image.detach().cpu().squeeze(0).clamp(0.0, 1.0), step)
    writer.add_image("images/current_generated", make_grid(current_images), step)
    if best_image is not None:
        writer.add_image("images/best_generated", best_image.detach().cpu().squeeze(0).clamp(0.0, 1.0), step)
        diff = torch.abs(best_image.detach().cpu().clamp(0.0, 1.0) - target_image.detach().cpu().clamp(0.0, 1.0))
        writer.add_image("images/best_absolute_pixel_difference", diff.squeeze(0), step)


def log_embedding_histograms(writer: object, step: int, experiment: SDXLTurboEmbeddingExperiment) -> None:
    for manager in experiment.managers:
        if manager.parameter is None:
            continue
        writer.add_histogram(f"embeddings/{manager.name}_values", manager.parameter.detach().cpu(), step)
        if manager.parameter.grad is not None:
            writer.add_histogram(f"embeddings/{manager.name}_gradients", manager.parameter.grad.detach().cpu(), step)
