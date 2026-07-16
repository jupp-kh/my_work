from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

try:
    from .certphash_wrapper import CertPhashWrapper, HashDetails, TargetHash
    from .config import METRIC_FIELDS
    from .image_utils import exact_rgb_l1_distance, mean_abs_pixel_difference_from_l1
except ImportError:  # pragma: no cover - script execution fallback.
    from certphash_wrapper import CertPhashWrapper, HashDetails, TargetHash
    from config import METRIC_FIELDS
    from image_utils import exact_rgb_l1_distance, mean_abs_pixel_difference_from_l1


@dataclass
class BestState:
    l1_distance: float = math.inf
    hash_distance: float = math.inf
    exact_rgb_l1_distance: float = math.inf
    mean_absolute_pixel_difference: float = math.inf
    loss: float = math.inf
    step: int = 0
    image: torch.Tensor | None = None
    quantized_hash: torch.Tensor | None = None
    hash_details: HashDetails | None = None
    checkpoint_path: str | None = None


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_stem(value: str) -> str:
    stem = Path(value).stem if value else "target"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)[:80] or "target"


def unique_run_dir(runs_dir: Path, run_name: str, target_image: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = runs_dir / f"{safe_stem(run_name)}_{safe_stem(target_image.name)}_{timestamp}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = Path(f"{base}_{suffix:02d}")
        suffix += 1
    return candidate


def newest_checkpoint_in_run(run_dir: Path) -> Path:
    checkpoint_dir = run_dir / "checkpoints"
    candidates = sorted(checkpoint_dir.glob("*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No checkpoint .pt files found under {checkpoint_dir}.")


def resolve_resume_checkpoint(args: argparse.Namespace) -> Path | None:
    if args.resume_from is None:
        return None
    if args.resume_from != "latest":
        return Path(args.resume_from).expanduser().resolve()
    if args.run_dir is not None:
        return newest_checkpoint_in_run(args.run_dir.expanduser().resolve())

    target_stem = safe_stem(args.target_image.name)
    run_name = safe_stem(args.run_name)
    pattern = f"{run_name}_{target_stem}_*"
    run_candidates = sorted(args.runs_dir.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    for run_candidate in run_candidates:
        if run_candidate.is_dir():
            try:
                return newest_checkpoint_in_run(run_candidate)
            except FileNotFoundError:
                continue
    raise FileNotFoundError(
        "Could not resolve --resume-from latest. Pass --run-dir <RUN_DIRECTORY> or use a run name/target image "
        "that matches an existing run under --runs-dir."
    )


def ensure_run_layout(run_dir: Path) -> dict[str, Path]:
    paths = {
        "run": run_dir,
        "checkpoints": run_dir / "checkpoints",
        "images": run_dir / "images",
        "tensorboard": run_dir / "tensorboard",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(jsonable(data), handle, indent=2, sort_keys=True)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(row), sort_keys=True) + "\n")


def append_csv(path: Path, row: dict[str, Any]) -> None:
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({field: jsonable(row.get(field, "")) for field in METRIC_FIELDS})


def torch_load(path: Path, map_location: torch.device | str = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def random_state_payload() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_random_state(payload: dict[str, Any]) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.random.set_rng_state(payload["torch_cpu"])
    if torch.cuda.is_available() and payload.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(payload["torch_cuda"])


def optimizer_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def effective_eval_interval(
    base_interval: int,
    *,
    best_l1_distance: float,
    last_eval_l1_distance: float,
    close_delta: float,
    best_l1_threshold: float,
    factor: float,
    min_interval: int,
    enabled: bool = True,
) -> int:
    base = max(1, int(base_interval))
    minimum = max(1, int(min_interval))
    if not enabled or factor <= 1.0:
        return base

    close_to_best = (
        math.isfinite(best_l1_distance)
        and math.isfinite(last_eval_l1_distance)
        and last_eval_l1_distance <= best_l1_distance + max(0.0, float(close_delta))
    )
    low_best_l1 = math.isfinite(best_l1_distance) and best_l1_distance <= float(best_l1_threshold)
    if not close_to_best and not low_best_l1:
        return base

    shortened = int(math.ceil(base / float(factor)))
    return min(base, max(minimum, shortened))


def decay_optimizer_lr(
    optimizer: torch.optim.Optimizer,
    factor: float,
    min_learning_rate: float,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> float:
    last_lr = min_learning_rate
    for group in optimizer.param_groups:
        group["lr"] = max(float(group["lr"]) * factor, min_learning_rate)
        last_lr = float(group["lr"])
    if lr_scheduler is not None and hasattr(lr_scheduler, "base_lrs"):
        lr_scheduler.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    return last_lr


def all_parameters_finite(parameters: Iterable[nn.Parameter]) -> bool:
    for parameter in parameters:
        if not torch.isfinite(parameter.detach()).all():
            return False
    return True


def all_gradients_finite(parameters: Iterable[nn.Parameter]) -> bool:
    for parameter in parameters:
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            return False
    return True


def gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    grads = [parameter.grad.detach().reshape(-1).float() for parameter in parameters if parameter.grad is not None]
    if not grads:
        return 0.0
    return float(torch.linalg.vector_norm(torch.cat(grads)).cpu())


def save_embedding_file(path: Path, experiment: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(experiment.embedding_state(), path)


def next_checkpoint_path(checkpoint_dir: Path, step: int) -> Path:
    base = checkpoint_dir / f"step_{step:06d}.pt"
    if not base.exists():
        return base
    suffix = 1
    while True:
        candidate = checkpoint_dir / f"step_{step:06d}_resume{suffix:02d}.pt"
        if not candidate.exists():
            return candidate
        suffix += 1


def best_checkpoint_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "best.pt"


def save_checkpoint(
    path: Path,
    *,
    step: int,
    experiment: Any,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    best: BestState,
    fixed_latents: torch.Tensor,
    config: dict[str, Any],
    resume_history: list[dict[str, Any]],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "global_step": step,
            "placeholder_embeddings": experiment.embedding_state(),
            "optimizer_state": optimizer.state_dict(),
            "lr_scheduler_state": lr_scheduler.state_dict(),
            "best_l1_distance": best.l1_distance,
            "best_hash_distance": best.hash_distance,
            "best_exact_rgb_l1_distance": best.exact_rgb_l1_distance,
            "best_mean_absolute_pixel_difference": best.mean_absolute_pixel_difference,
            "best_loss": best.loss,
            "best_step": best.step,
            "best_checkpoint_path": best.checkpoint_path,
            "best_image": best.image.detach().cpu() if best.image is not None else None,
            "best_quantized_hash": best.quantized_hash.detach().cpu() if best.quantized_hash is not None else None,
            "best_hash_details": asdict(best.hash_details) if best.hash_details is not None else None,
            "fixed_latents": fixed_latents.detach().cpu(),
            "config": config,
            "resume_history": resume_history,
            "random_state": random_state_payload(),
        },
        path,
    )
    return str(path)


def load_checkpoint_into_run(
    checkpoint_path: Path,
    experiment: Any,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor, BestState, int, list[dict[str, Any]]]:
    payload = torch_load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"{checkpoint_path} does not contain a checkpoint dictionary.")
    experiment.load_embedding_state(payload["placeholder_embeddings"])
    optimizer.load_state_dict(payload["optimizer_state"])
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)
    lr_scheduler.load_state_dict(payload["lr_scheduler_state"])
    restore_random_state(payload["random_state"])
    fixed_latents = payload["fixed_latents"].to(device=device)
    best = BestState(
        l1_distance=float(payload.get("best_l1_distance", math.inf)),
        hash_distance=float(payload.get("best_hash_distance", math.inf)),
        exact_rgb_l1_distance=float(payload.get("best_exact_rgb_l1_distance", math.inf)),
        mean_absolute_pixel_difference=float(payload.get("best_mean_absolute_pixel_difference", math.inf)),
        loss=float(payload.get("best_loss", math.inf)),
        step=int(payload.get("best_step", 0)),
        image=payload.get("best_image"),
        quantized_hash=payload.get("best_quantized_hash"),
        checkpoint_path=payload.get("best_checkpoint_path"),
    )
    if payload.get("best_hash_details") is not None:
        best.hash_details = HashDetails(**payload["best_hash_details"])
    resume_history = list(payload.get("resume_history", []))
    start_step = int(payload.get("global_step", 0))
    return payload, fixed_latents, best, start_step, resume_history


def evaluate_exact(
    *,
    images: torch.Tensor,
    target_image: torch.Tensor,
    target_hash: TargetHash,
    certphash: CertPhashWrapper,
    height: int,
    width: int,
) -> dict[str, Any]:
    clamped = images.detach().clamp(0.0, 1.0)
    quantized = certphash.exact_hash_tensor(clamped)
    hash_l1_values = certphash.exact_hash_l1(quantized, target_hash.quantized)
    rgb_l1_values = exact_rgb_l1_distance(clamped, target_image)
    best_index = int(torch.argmin(hash_l1_values).cpu())
    hash_l1 = float(hash_l1_values[best_index].cpu())
    rgb_l1 = float(rgb_l1_values[best_index].cpu())
    details = certphash.hash_details(quantized[best_index])
    return {
        "best_index": best_index,
        "exact_l1_distance": hash_l1,
        "exact_hash_distance": hash_l1,
        "exact_rgb_l1_distance": rgb_l1,
        "mean_absolute_pixel_difference": mean_abs_pixel_difference_from_l1(rgb_l1, height, width),
        "quantized_hash": quantized[best_index].detach().cpu(),
        "hash_details": details,
        "bit_match_percentage": bit_match_percentage(details.bit_string, target_hash.details.bit_string),
        "image": clamped[best_index : best_index + 1].detach().cpu(),
    }


def bit_match_percentage(left: str, right: str) -> float | None:
    if not left or not right or len(left) != len(right):
        return None
    matches = sum(1 for a, b in zip(left, right) if a == b)
    return 100.0 * matches / len(left)


def is_better_result(candidate: dict[str, Any], candidate_loss: float, best: BestState) -> bool:
    l1 = float(candidate["exact_l1_distance"])
    hash_distance = float(candidate["exact_hash_distance"])
    if l1 < best.l1_distance:
        return True
    if math.isclose(l1, best.l1_distance) and hash_distance < best.hash_distance:
        return True
    if math.isclose(l1, best.l1_distance) and math.isclose(hash_distance, best.hash_distance):
        return candidate_loss < best.loss
    return False


def software_report(device: torch.device) -> dict[str, Any]:
    gpu_info: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpu_info.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory": props.total_memory,
                    "major": props.major,
                    "minor": props.minor,
                }
            )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "selected_device": str(device),
        "gpus": gpu_info,
    }


def args_to_config(args: argparse.Namespace, run_dir: Path, dtype: torch.dtype, device: torch.device) -> dict[str, Any]:
    data = vars(args).copy()
    data["run_dir"] = str(run_dir)
    data["dtype"] = str(dtype)
    data["device"] = str(device)
    data["created_at"] = datetime.now().isoformat()
    return jsonable(data)
