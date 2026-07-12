from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from certphash_wrapper import CertPhashWrapper, DEFAULT_CERTPHASH_MODEL  # type: ignore
    from diffusion_wrapper import DiffusionLoadConfig, FrozenStableDiffusion, resolve_torch_dtype  # type: ignore
    from image_utils import image_grid, load_image_tensor, save_tensor_grid, save_tensor_image, timestamped_run_name  # type: ignore
    from logging_utils import JsonlLogger, TensorBoardLogger  # type: ignore
    from prompt_search import PromptSearchConfig, generate_or_load_prompt_candidates, score_prompt_candidates  # type: ignore
else:
    from .certphash_wrapper import CertPhashWrapper, DEFAULT_CERTPHASH_MODEL
    from .diffusion_wrapper import DiffusionLoadConfig, FrozenStableDiffusion, resolve_torch_dtype
    from .image_utils import image_grid, load_image_tensor, save_tensor_grid, save_tensor_image, timestamped_run_name
    from .logging_utils import JsonlLogger, TensorBoardLogger
    from .prompt_search import PromptSearchConfig, generate_or_load_prompt_candidates, score_prompt_candidates


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "my_work" / "diffusion_model_based_attack" /"results" 
DEFAULT_TENSORBOARD_DIR = REPO_ROOT / "my_work" / "diffusion_model_based_attack" /"results" / "tensorboard" 
DEFAULT_DIFFUSION_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize only a diffusion model's initial latent/noise toward a target CertPhash hash."
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-image", type=Path, help="Image whose CertPhash should be targeted.")
    target.add_argument("--target-hash", type=str, help="Target hash as base64 or 144 integer byte values.")

    parser.add_argument(
        "--prompt",
        type=str,
        default="a realistic overhead food photograph of a lunch box filled with colorful vegetables",
        help="Text prompt for generation.",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="abstract, glitch, distorted, blurry, low quality, noisy, duplicate, tiled",
        help="Negative prompt.",
    )
    parser.add_argument("--diffusion-model", type=str, default=DEFAULT_DIFFUSION_MODEL)
    parser.add_argument("--diffusion-variant", type=str, default=None, help="Optional model variant, such as fp16.")
    parser.add_argument("--scheduler", choices=["ddim", "default"], default="ddim")
    parser.add_argument("--certphash-model", type=Path, default=DEFAULT_CERTPHASH_MODEL)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")

    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--diffusion-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--model-input-size", type=int, default=64)

    parser.add_argument("--optimization-steps", type=int, default=200)
    parser.add_argument("--optimizer", choices=["Adam", "AdamW", "SGD"], default="Adam")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--latent-l2-weight", type=float, default=1e-4)
    parser.add_argument("--optimize-prompt-embeds", action="store_true")
    parser.add_argument("--prompt-learning-rate", type=float, default=1e-5)
    parser.add_argument("--prompt-l2-weight", type=float, default=1e-3)
    parser.add_argument("--prompt-grad-clip", type=float, default=0.03)
    parser.add_argument(
        "--rollback-patience",
        type=int,
        default=0,
        help="Restore the best latent after this many checked steps without improvement. Use 0 to disable.",
    )
    parser.add_argument(
        "--rollback-min-regression",
        type=float,
        default=500.0,
        help="Only rollback when current true L1 is this much worse than best_l1.",
    )
    parser.add_argument(
        "--rollback-lr-factor",
        type=float,
        default=0.5,
        help="Multiply learning rate by this factor after rollback.",
    )
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--rollback-noise-std",
        type=float,
        default=0.0,
        help="After rollback, add Gaussian noise with this std to the restored best latent. Use 0 to disable.",
    )
    parser.add_argument(
        "--image-anchor-weight",
        type=float,
        default=0.02,
        help="MSE penalty that keeps optimized images close to the unoptimized prompt-generated baseline.",
    )
    parser.add_argument(
        "--total-variation-weight",
        type=float,
        default=1e-4,
        help="Smoothness penalty for reducing noisy image artifacts.",
    )

    parser.add_argument("--threshold", type=float, default=1800.0, help="Stop when true rounded L1 hash distance is <= this.")
    parser.add_argument("--hash-loss", choices=["l1", "mse", "smooth_l1"], default="smooth_l1")
    parser.add_argument("--hash-scale", type=float, default=255.0)
    parser.add_argument("--smooth-l1-beta", type=float, default=0.02)

    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tensorboard-dir", type=Path, default=DEFAULT_TENSORBOARD_DIR)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--init-latents",
        type=Path,
        default=None,
        help="Initialize optimization from a saved latent tensor, such as best_latents.pt.",
    )
    parser.add_argument(
        "--init-latents-noise-std",
        type=float,
        default=0.0,
        help="Add Gaussian noise with this std after loading or creating the initial latent. Use 0 to disable.",
    )
    parser.add_argument(
        "--init-prompt-embeds",
        type=Path,
        default=None,
        help="Initialize prompt embeddings from a saved tensor, such as best_prompt_embeds.pt.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Initialize latents and prompt embeddings from a checkpoint_step_*.pt file.",
    )
    parser.add_argument(
        "--resume-optimizer",
        action="store_true",
        help="When --resume-checkpoint is used, also restore the saved optimizer state.",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--image-save-interval", type=int, default=10)
    parser.add_argument("--check-interval", type=int, default=1)
    parser.add_argument("--disable-tensorboard", action="store_true")
    parser.add_argument("--disable-progress-bar", action="store_true")
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Save the initial prompt-generated baseline and exit before hash optimization.",
    )
    parser.add_argument(
        "--empty-cache-interval",
        type=int,
        default=1,
        help="Call torch.cuda.empty_cache every N optimization steps. Use 0 to disable.",
    )

    parser.add_argument("--no-attention-slicing", action="store_true")
    parser.add_argument("--no-vae-slicing", action="store_true")
    parser.add_argument("--vae-tiling", action="store_true")
    parser.add_argument("--no-safetensors", action="store_true")
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    parser.add_argument("--disable-unet-checkpoint", action="store_true")
    parser.add_argument("--disable-vae-checkpoint", action="store_true")

    parser.add_argument("--prompt-search", action="store_true")
    parser.add_argument("--prompt-search-num", type=int, default=8)
    parser.add_argument("--prompt-search-cache-dir", type=Path, default=None)
    parser.add_argument("--prompt-search-force", action="store_true")
    parser.add_argument("--prompt-search-force-rescore", action="store_true")
    parser.add_argument(
        "--prompt-search-mode",
        choices=["best", "cycle", "random"],
        default="best",
        help="Use one best prompt, cycle through a top-k prompt pool, or sample that pool randomly.",
    )
    parser.add_argument(
        "--prompt-search-top-k",
        type=int,
        default=4,
        help="Number of top scored prompts to keep for cycle/random prompt-search modes.",
    )
    parser.add_argument(
        "--prompt-search-generator-model",
        type=str,
        default="templates",
        help="Use 'templates' for cached template prompts, or a small transformers text-generation model id.",
    )
    parser.add_argument(
        "--prompt-search-space",
        choices=["broad", "prompt"],
        default="broad",
        help="Use broad unrelated scene prompts for attack search, or prompt-near variants of the initial prompt.",
    )
    parser.add_argument("--prompt-search-hash-weight", type=float, default=1.0)
    parser.add_argument("--prompt-search-image-weight", type=float, default=0.0)
    return parser.parse_args()


def choose_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(
    name: str,
    latents: torch.Tensor,
    learning_rate: float,
    prompt_embeds: torch.Tensor | None = None,
    prompt_learning_rate: float = 1e-4,
) -> torch.optim.Optimizer:
    parameters: list[dict[str, Any]] = [{"params": [latents], "lr": learning_rate, "name": "latents"}]
    if prompt_embeds is not None:
        parameters.append({"params": [prompt_embeds], "lr": prompt_learning_rate, "name": "prompt_embeds"})
    if name == "Adam":
        return torch.optim.Adam(parameters)
    if name == "AdamW":
        return torch.optim.AdamW(parameters)
    if name == "SGD":
        return torch.optim.SGD(parameters)
    raise ValueError(f"Unsupported optimizer: {name}")


def reset_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    optimizer.state.clear()


def decay_optimizer_lr(
    optimizer: torch.optim.Optimizer,
    factor: float,
    min_learning_rate: float,
) -> float:
    last_lr = min_learning_rate
    for group in optimizer.param_groups:
        group["lr"] = max(float(group["lr"]) * factor, min_learning_rate)
        last_lr = float(group["lr"])
    return last_lr


def optimizer_lrs(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {
        str(group.get("name", f"group_{index}")): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def tensor_is_finite(tensor: torch.Tensor | None) -> bool:
    if tensor is None:
        return True
    return bool(torch.isfinite(tensor).all().detach().cpu().item())


def restore_best_state(
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    best_latents: torch.Tensor,
    best_prompt_embeds: torch.Tensor,
    optimize_prompt_embeds: bool,
) -> None:
    with torch.no_grad():
        latents.copy_(best_latents)
        if optimize_prompt_embeds:
            prompt_embeds.copy_(best_prompt_embeds)


def add_latent_noise_(latents: torch.Tensor, std: float) -> float:
    if std <= 0:
        return 0.0
    with torch.no_grad():
        noise = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype) * std
        latents.add_(noise)
        return float(torch.linalg.vector_norm(noise.reshape(-1)).detach().cpu())


def select_prompt_index(step: int, pool_size: int, mode: str, rng: random.Random) -> int:
    if pool_size <= 1:
        return 0
    if mode == "cycle":
        return (step - 1) % pool_size
    if mode == "random":
        return rng.randrange(pool_size)
    return 0


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def args_to_jsonable(args: argparse.Namespace) -> dict[str, Any]:
    data = vars(args).copy()
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def torch_load_local(path: Path, map_location: torch.device | str = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def tensor_from_payload(payload: Any, key: str, source: Path) -> torch.Tensor:
    if isinstance(payload, torch.Tensor):
        return payload
    if isinstance(payload, dict) and key in payload and isinstance(payload[key], torch.Tensor):
        return payload[key]
    raise ValueError(f"{source} does not contain tensor key '{key}'.")


def prepare_loaded_tensor(
    tensor: torch.Tensor,
    reference: torch.Tensor,
    name: str,
    source: Path,
    device: torch.device,
) -> torch.Tensor:
    loaded = tensor.detach()
    if tuple(loaded.shape) != tuple(reference.shape):
        raise ValueError(
            f"{name} from {source} has shape {tuple(loaded.shape)}, "
            f"but this run expects {tuple(reference.shape)}. "
            "Use the same height, width, batch size, prompt pool, and guidance settings as the source run."
        )
    return loaded.to(device=device, dtype=reference.dtype).detach()


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def save_checkpoint(
    path: Path,
    step: int,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    best: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "latents": latents.detach().cpu(),
            "prompt_embeds": prompt_embeds.detach().cpu(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best": best,
            "args": args_to_jsonable(args),
        },
        path,
    )


def total_variation(images: torch.Tensor) -> torch.Tensor:
    horizontal = torch.abs(images[:, :, :, 1:] - images[:, :, :, :-1]).mean()
    vertical = torch.abs(images[:, :, 1:, :] - images[:, :, :-1, :]).mean()
    return horizontal + vertical


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.optimization_steps < 1:
        raise ValueError("--optimization-steps must be >= 1.")
    if args.init_latents_noise_std < 0:
        raise ValueError("--init-latents-noise-std must be >= 0.")
    if args.rollback_noise_std < 0:
        raise ValueError("--rollback-noise-std must be >= 0.")
    if args.resume_checkpoint is not None and args.init_latents is not None:
        raise ValueError("Use either --resume-checkpoint or --init-latents, not both.")
    if args.resume_optimizer and args.resume_checkpoint is None:
        raise ValueError("--resume-optimizer requires --resume-checkpoint.")

    set_reproducibility(args.seed)
    device = choose_device(args.device)
    dtype = resolve_torch_dtype(args.dtype, device)
    if device.type == "cpu":
        print("Warning: CUDA is unavailable or not selected. Diffusion optimization on CPU will be very slow.")
        if dtype in {torch.float16, torch.bfloat16}:
            print("Warning: half precision is not supported well on CPU; using float32 instead.")
            dtype = torch.float32

    target_label = str(args.target_image) if args.target_image else "target_hash"
    run_name = args.run_name or timestamped_run_name(target_label)
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = args.tensorboard_dir / run_name

    write_json(run_dir / "config.json", args_to_jsonable(args))
    metrics_logger = JsonlLogger(run_dir / "metrics.jsonl")
    tb = TensorBoardLogger(tensorboard_dir, enabled=not args.disable_tensorboard and not args.preview_only)

    print(f"Run directory: {run_dir}")
    print(f"TensorBoard directory: {tensorboard_dir}")
    print(f"Device: {device}")
    print(f"Diffusion dtype: {dtype}")
    print(f"Target L1 threshold: {args.threshold}")

    resume_payload = None
    resume_optimizer_state = None
    if args.resume_checkpoint is not None:
        resume_payload = torch_load_local(args.resume_checkpoint, map_location="cpu")
        if not isinstance(resume_payload, dict):
            raise ValueError("--resume-checkpoint must point to a checkpoint dict.")
        if "latents" not in resume_payload:
            raise ValueError(f"{args.resume_checkpoint} does not contain checkpoint latents.")
        resume_optimizer_state = resume_payload.get("optimizer_state_dict")
        print(f"Loaded resume checkpoint: {args.resume_checkpoint}")
        if "step" in resume_payload:
            print(f"Checkpoint step: {resume_payload['step']}")

    certphash = CertPhashWrapper(
        checkpoint_path=args.certphash_model,
        device=device,
        model_input_size=args.model_input_size,
    )
    print(f"CertPhash trainable parameters: {certphash.trainable_parameter_count}")

    target_image = None
    if args.target_image:
        target_image = load_image_tensor(args.target_image, device=device)
        target_hash = certphash.target_from_image(target_image, source=str(args.target_image))
        save_tensor_image(target_image, run_dir / "target.png")
        if tb.enabled:
            tb.add_image("images/target", target_image.detach().cpu().squeeze(0), 0)
    else:
        target_hash = certphash.target_from_hash(args.target_hash, source="target_hash")

    tb.add_text("target/source", target_hash.source, 0)
    tb.add_text("target/hash_base64", target_hash.details.base64, 0)

    diffusion = FrozenStableDiffusion.load(
        DiffusionLoadConfig(
            model_id=args.diffusion_model,
            device=device,
            dtype=dtype,
            scheduler=args.scheduler,
            use_safetensors=not args.no_safetensors,
            variant=args.diffusion_variant,
            attention_slicing=not args.no_attention_slicing,
            vae_slicing=not args.no_vae_slicing,
            vae_tiling=args.vae_tiling,
            gradient_checkpointing=not args.disable_gradient_checkpointing,
            checkpoint_unet=not args.disable_unet_checkpoint,
            checkpoint_vae=not args.disable_vae_checkpoint,
        )
    )
    print(f"Diffusion trainable parameters: {diffusion.trainable_parameter_count}")

    prompt_search_result = None
    prompt_pool_prompts: list[str] | None = None
    if args.prompt_search:
        if args.prompt_search_num < 1:
            raise ValueError("--prompt-search-num must be >= 1 when --prompt-search is enabled.")
        if args.prompt_search_top_k < 1:
            raise ValueError("--prompt-search-top-k must be >= 1.")
        prompt_cache_dir = args.prompt_search_cache_dir or (args.output_dir / "prompt_cache")
        prompts, prompt_cache_path = generate_or_load_prompt_candidates(
            PromptSearchConfig(
                base_prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                target_label=target_label,
                num_prompts=args.prompt_search_num,
                seed=args.seed,
                generator_model=args.prompt_search_generator_model,
                cache_dir=prompt_cache_dir,
                search_space=args.prompt_search_space,
                force_regenerate=args.prompt_search_force,
            )
        )
        prompt_search_result = score_prompt_candidates(
            prompts=prompts,
            prompt_cache_path=prompt_cache_path,
            diffusion=diffusion,
            certphash=certphash,
            target_hash=target_hash,
            target_image=target_image,
            negative_prompt=args.negative_prompt,
            batch_size=args.batch_size,
            height=args.height,
            width=args.width,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
            diffusion_steps=args.diffusion_steps,
            diffusion_model=args.diffusion_model,
            hash_weight=args.prompt_search_hash_weight,
            image_weight=args.prompt_search_image_weight,
            run_dir=run_dir,
            force_rescore=args.prompt_search_force_rescore,
        )
        ranked_scores = sorted(prompt_search_result.scores, key=lambda item: item.score)
        if args.prompt_search_mode == "best":
            args.prompt = prompt_search_result.selected_prompt
            prompt_pool_prompts = None
        else:
            prompt_pool_prompts = [
                item.prompt for item in ranked_scores[: min(args.prompt_search_top_k, len(ranked_scores))]
            ]
            args.prompt = prompt_pool_prompts[0]
        write_json(
            run_dir / "selected_prompt.json",
            {
                "selected_prompt": args.prompt,
                "prompt_search_mode": args.prompt_search_mode,
                "prompt_search_space": args.prompt_search_space,
                "prompt_pool_prompts": prompt_pool_prompts,
                "selected_score": asdict(prompt_search_result.selected_score),
                "prompt_cache": str(prompt_search_result.prompt_cache),
                "score_cache": str(prompt_search_result.score_cache),
            },
        )
        print(f"Selected prompt: {args.prompt}")
        if prompt_pool_prompts is not None:
            print(f"Prompt pool ({len(prompt_pool_prompts)}):")
            for index, prompt in enumerate(prompt_pool_prompts, start=1):
                print(f"  {index:02d}. {prompt}")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if prompt_pool_prompts is None:
        prompt_embeds = diffusion.encode_prompt(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            batch_size=args.batch_size,
            guidance_scale=args.guidance_scale,
        )
    else:
        prompt_embed_list = [
            diffusion.encode_prompt(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                batch_size=args.batch_size,
                guidance_scale=args.guidance_scale,
            )
            for prompt in prompt_pool_prompts
        ]
        prompt_embeds = torch.stack(prompt_embed_list, dim=0)
    if args.optimize_prompt_embeds:
        prompt_embeds = prompt_embeds.float()

    resume_info: dict[str, Any] = {
        "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint is not None else None,
        "init_latents": str(args.init_latents) if args.init_latents is not None else None,
        "init_prompt_embeds": str(args.init_prompt_embeds) if args.init_prompt_embeds is not None else None,
        "init_latents_noise_std": args.init_latents_noise_std,
        "rollback_noise_std": args.rollback_noise_std,
        "resume_optimizer": bool(args.resume_optimizer),
    }
    if resume_payload is not None and "prompt_embeds" in resume_payload:
        prompt_embeds = prepare_loaded_tensor(
            tensor_from_payload(resume_payload, "prompt_embeds", args.resume_checkpoint),
            prompt_embeds,
            "prompt_embeds",
            args.resume_checkpoint,
            device,
        )
        print(f"Initialized prompt embeddings from checkpoint: {args.resume_checkpoint}")
    if args.init_prompt_embeds is not None:
        prompt_embed_payload = torch_load_local(args.init_prompt_embeds, map_location="cpu")
        prompt_embeds = prepare_loaded_tensor(
            tensor_from_payload(prompt_embed_payload, "prompt_embeds", args.init_prompt_embeds),
            prompt_embeds,
            "prompt_embeds",
            args.init_prompt_embeds,
            device,
        )
        print(f"Initialized prompt embeddings from: {args.init_prompt_embeds}")
    if args.optimize_prompt_embeds:
        prompt_embeds = prompt_embeds.float()
    initial_prompt_embeds = prompt_embeds.clone().detach()
    latents = diffusion.initial_latents(
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        seed=args.seed,
    ).detach()
    if resume_payload is not None:
        latents = prepare_loaded_tensor(
            tensor_from_payload(resume_payload, "latents", args.resume_checkpoint),
            latents,
            "latents",
            args.resume_checkpoint,
            device,
        )
        print(f"Initialized latents from checkpoint: {args.resume_checkpoint}")
    elif args.init_latents is not None:
        latent_payload = torch_load_local(args.init_latents, map_location="cpu")
        latents = prepare_loaded_tensor(
            tensor_from_payload(latent_payload, "latents", args.init_latents),
            latents,
            "latents",
            args.init_latents,
            device,
        )
        print(f"Initialized latents from: {args.init_latents}")
    init_noise_norm = add_latent_noise_(latents, args.init_latents_noise_std)
    resume_info["init_latents_noise_norm"] = init_noise_norm
    if init_noise_norm > 0:
        print(
            f"Added initial latent noise: std={args.init_latents_noise_std:g}, "
            f"noise_norm={init_noise_norm:.6f}"
        )
    initial_latents = latents.clone().detach()

    with torch.no_grad():
        baseline_prompt_embeds = prompt_embeds[0] if prompt_pool_prompts is not None else prompt_embeds
        baseline_images = diffusion.generate(
            initial_latents=initial_latents,
            prompt_embeds=baseline_prompt_embeds,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.diffusion_steps,
        ).detach()
    save_tensor_grid(baseline_images, run_dir / "initial_baseline.png")
    tb.add_image("images/initial_baseline", image_grid(baseline_images), 0)
    if args.preview_only:
        write_json(
            run_dir / "summary.json",
            {
                "preview_only": True,
                "run_dir": str(run_dir),
                "baseline_image": str(run_dir / "initial_baseline.png"),
                "prompt": args.prompt,
                "negative_prompt": args.negative_prompt,
                "height": args.height,
                "width": args.width,
                "diffusion_steps": args.diffusion_steps,
                "guidance_scale": args.guidance_scale,
                "resume": resume_info,
            },
        )
        print(f"Saved prompt preview to {run_dir / 'initial_baseline.png'}")
        tb.close()
        return
    if device.type == "cuda":
        torch.cuda.empty_cache()

    latents.requires_grad_(True)
    if args.optimize_prompt_embeds:
        prompt_embeds = prompt_embeds.detach().clone().float().requires_grad_(True)
    else:
        prompt_embeds = prompt_embeds.detach()
    optimizer = build_optimizer(
        args.optimizer,
        latents,
        args.learning_rate,
        prompt_embeds=prompt_embeds if args.optimize_prompt_embeds else None,
        prompt_learning_rate=args.prompt_learning_rate,
    )
    if args.resume_optimizer:
        if resume_optimizer_state is None:
            raise ValueError(f"{args.resume_checkpoint} does not contain an optimizer_state_dict.")
        try:
            optimizer.load_state_dict(resume_optimizer_state)
        except ValueError as exc:
            raise ValueError(
                "Could not restore optimizer state. Make sure --optimize-prompt-embeds and "
                "optimizer settings match the checkpoint, or resume without --resume-optimizer."
            ) from exc
        move_optimizer_state_to_device(optimizer, device)
        print(f"Restored optimizer state from checkpoint: {args.resume_checkpoint}")

    best_l1 = math.inf
    best_step = 0
    best_index = 0
    best_image = None
    best_quantized = None
    best_latents = latents.detach().clone()
    best_prompt_embeds = prompt_embeds.detach().clone()
    checks_since_best = 0
    success = False
    prompt_rng = random.Random(args.seed + 1337)
    previous_images = None
    previous_logits = None
    previous_prompt_index = None

    def recover_nonfinite_step(reason: str, step: int) -> float:
        optimizer.zero_grad(set_to_none=True)
        restore_best_state(
            latents,
            prompt_embeds,
            best_latents,
            best_prompt_embeds,
            args.optimize_prompt_embeds,
        )
        new_lr = decay_optimizer_lr(optimizer, args.rollback_lr_factor, args.min_learning_rate)
        reset_optimizer_state(optimizer)
        tb.add_scalar("optimization/nonfinite_recovery", 1.0, step)
        tb.add_scalar("optimization/learning_rate_after_nonfinite", new_lr, step)
        print(
            f"nonfinite recovery at step={step:05d}: {reason}; restored best state "
            f"from step {best_step}, best_l1={best_l1:.2f}, new_lr={new_lr:.6g}"
        )
        return new_lr

    try:
        if args.disable_progress_bar:
            iterator = range(1, args.optimization_steps + 1)
        else:
            try:
                from tqdm import tqdm

                iterator = tqdm(range(1, args.optimization_steps + 1))
            except ImportError:
                iterator = range(1, args.optimization_steps + 1)

        for step in iterator:
            active_prompt_index = (
                select_prompt_index(step, prompt_embeds.shape[0], args.prompt_search_mode, prompt_rng)
                if prompt_pool_prompts is not None
                else 0
            )
            active_prompt_embeds = prompt_embeds[active_prompt_index] if prompt_pool_prompts is not None else prompt_embeds
            images = diffusion.generate(
                initial_latents=latents,
                prompt_embeds=active_prompt_embeds,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.diffusion_steps,
            )
            logits = certphash.logits(images)
            with torch.no_grad():
                logit_l1_distances = certphash.logit_l1_distance_per_sample(logits.detach(), target_hash.quantized)
                current_min_logit_l1_tensor = torch.min(logit_l1_distances)
                current_min_logit_l1 = float(current_min_logit_l1_tensor.cpu())
                comparable_previous = (
                    previous_images is not None
                    and previous_logits is not None
                    and previous_prompt_index == active_prompt_index
                )
                image_delta = (
                    float(torch.mean(torch.abs(images.detach() - previous_images)).cpu())
                    if comparable_previous
                    else math.nan
                )
                logit_delta = (
                    float(torch.mean(torch.abs(logits.detach() - previous_logits)).cpu())
                    if comparable_previous
                    else math.nan
                )
            hash_loss = certphash.surrogate_loss(
                logits=logits,
                target_quantized=target_hash.quantized,
                loss_type=args.hash_loss,
                hash_scale=args.hash_scale,
                smooth_l1_beta=args.smooth_l1_beta,
            )
            latent_l2 = torch.mean((latents - initial_latents).pow(2))
            prompt_l2 = (
                torch.mean((prompt_embeds - initial_prompt_embeds).pow(2))
                if args.optimize_prompt_embeds and args.prompt_l2_weight > 0
                else torch.zeros((), device=device)
            )
            image_anchor_loss = F.mse_loss(images, baseline_images) if args.image_anchor_weight > 0 else torch.zeros((), device=device)
            tv_loss = total_variation(images) if args.total_variation_weight > 0 else torch.zeros((), device=device)
            loss = (
                hash_loss
                + args.latent_l2_weight * latent_l2
                + args.prompt_l2_weight * prompt_l2
                + args.image_anchor_weight * image_anchor_loss
                + args.total_variation_weight * tv_loss
            )

            if not tensor_is_finite(loss):
                checks_since_best = 0
                new_lr = recover_nonfinite_step("loss became NaN or inf", step)
                metrics_logger.write(
                    {
                        "step": step,
                        "skipped_step": True,
                        "skip_reason": "nonfinite_loss",
                        "best_l1": best_l1,
                        "best_step": best_step,
                        "active_prompt_index": active_prompt_index,
                        "active_prompt": prompt_pool_prompts[active_prompt_index] if prompt_pool_prompts is not None else args.prompt,
                        "learning_rate": float(new_lr),
                        "success": success,
                    }
                )
                del images, logits, hash_loss, latent_l2, prompt_l2, image_anchor_loss, tv_loss, loss
                if args.empty_cache_interval > 0 and device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            latents_grad_finite = tensor_is_finite(latents.grad)
            prompt_grad_finite = tensor_is_finite(prompt_embeds.grad if args.optimize_prompt_embeds else None)
            if not latents_grad_finite or not prompt_grad_finite:
                checks_since_best = 0
                reason = "gradient became NaN or inf"
                if not latents_grad_finite:
                    reason = "latent gradient became NaN or inf"
                if not prompt_grad_finite:
                    reason = "prompt gradient became NaN or inf"
                new_lr = recover_nonfinite_step(reason, step)
                metrics_logger.write(
                    {
                        "step": step,
                        "skipped_step": True,
                        "skip_reason": "nonfinite_gradient",
                        "best_l1": best_l1,
                        "best_step": best_step,
                        "active_prompt_index": active_prompt_index,
                        "active_prompt": prompt_pool_prompts[active_prompt_index] if prompt_pool_prompts is not None else args.prompt,
                        "learning_rate": float(new_lr),
                        "success": success,
                    }
                )
                del images, logits, hash_loss, latent_l2, prompt_l2, image_anchor_loss, tv_loss, loss
                if args.empty_cache_interval > 0 and device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            grad_norm = float(torch.linalg.vector_norm(latents.grad.detach()).cpu()) if latents.grad is not None else 0.0
            prompt_grad_norm = (
                float(torch.linalg.vector_norm(prompt_embeds.grad.detach()).cpu())
                if args.optimize_prompt_embeds and prompt_embeds.grad is not None
                else 0.0
            )
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([latents], max_norm=args.grad_clip)
            if args.optimize_prompt_embeds and args.prompt_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([prompt_embeds], max_norm=args.prompt_grad_clip)
            clipped_grad_norm = float(torch.linalg.vector_norm(latents.grad.detach()).cpu()) if latents.grad is not None else 0.0
            clipped_prompt_grad_norm = (
                float(torch.linalg.vector_norm(prompt_embeds.grad.detach()).cpu())
                if args.optimize_prompt_embeds and prompt_embeds.grad is not None
                else 0.0
            )
            # The L1 check below uses images/logits from the pre-step state.
            # Keep matching tensors so best_l1, best.png, and best_latents.pt agree.
            checked_latents = latents.detach().clone()
            checked_prompt_embeds = prompt_embeds.detach().clone()
            optimizer.step()
            latent_update_norm = float(torch.linalg.vector_norm((latents.detach() - checked_latents).reshape(-1)).cpu())
            prompt_update_norm = (
                float(torch.linalg.vector_norm((prompt_embeds.detach() - checked_prompt_embeds).reshape(-1)).cpu())
                if args.optimize_prompt_embeds
                else 0.0
            )

            params_finite = tensor_is_finite(latents) and tensor_is_finite(prompt_embeds if args.optimize_prompt_embeds else None)
            if not params_finite:
                checks_since_best = 0
                new_lr = recover_nonfinite_step("optimizer produced NaN or inf parameters", step)
                metrics_logger.write(
                    {
                        "step": step,
                        "skipped_step": True,
                        "skip_reason": "nonfinite_parameters",
                        "best_l1": best_l1,
                        "best_step": best_step,
                        "grad_norm": grad_norm,
                        "prompt_grad_norm": prompt_grad_norm,
                        "active_prompt_index": active_prompt_index,
                        "active_prompt": prompt_pool_prompts[active_prompt_index] if prompt_pool_prompts is not None else args.prompt,
                        "learning_rate": float(new_lr),
                        "success": success,
                    }
                )
                del images, logits, hash_loss, latent_l2, prompt_l2, image_anchor_loss, tv_loss, loss
                if args.empty_cache_interval > 0 and device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            should_check = step == 1 or step % args.check_interval == 0 or step == args.optimization_steps
            current_min_l1 = math.nan
            if should_check:
                with torch.no_grad():
                    quantized = certphash.quantize_logits(logits.detach())
                    l1_distances = certphash.l1_distance_per_sample(quantized, target_hash.quantized)
                    current_min_l1_tensor, current_best_index_tensor = torch.min(l1_distances, dim=0)
                    current_min_l1 = float(current_min_l1_tensor.cpu())
                    current_best_index = int(current_best_index_tensor.cpu())

                    if current_min_l1 < best_l1:
                        best_l1 = current_min_l1
                        best_step = step
                        best_index = current_best_index
                        best_image = images[current_best_index : current_best_index + 1].detach().cpu()
                        best_quantized = quantized[current_best_index].detach().cpu()
                        best_latents = checked_latents
                        best_prompt_embeds = checked_prompt_embeds
                        checks_since_best = 0
                        save_tensor_image(best_image, run_dir / "best.png")
                        torch.save(checked_latents.cpu(), run_dir / "best_latents.pt")
                        if args.optimize_prompt_embeds:
                            torch.save(checked_prompt_embeds.cpu(), run_dir / "best_prompt_embeds.pt")
                    else:
                        checks_since_best += 1

                    if best_l1 <= args.threshold:
                        success = True

                print(
                    f"step={step:05d} loss={float(loss.detach().cpu()):.6f} "
                    f"hash_loss={float(hash_loss.detach().cpu()):.6f} "
                    f"l1={current_min_l1:.2f} best_l1={best_l1:.2f} "
                    f"logit_l1={current_min_logit_l1:.2f} grad_norm={grad_norm:.4f} "
                    f"clipped_grad_norm={clipped_grad_norm:.4f} update_norm={latent_update_norm:.6f} "
                    f"image_delta={image_delta:.6f} logit_delta={logit_delta:.6f} "
                    f"prompt_index={active_prompt_index}"
                )

                should_rollback = (
                    args.rollback_patience > 0
                    and checks_since_best >= args.rollback_patience
                    and current_min_l1 > best_l1 + args.rollback_min_regression
                )
                if should_rollback:
                    restore_best_state(
                        latents,
                        prompt_embeds,
                        best_latents,
                        best_prompt_embeds,
                        args.optimize_prompt_embeds,
                    )
                    rollback_noise_norm = add_latent_noise_(latents, args.rollback_noise_std)
                    new_lr = decay_optimizer_lr(optimizer, args.rollback_lr_factor, args.min_learning_rate)
                    reset_optimizer_state(optimizer)
                    checks_since_best = 0
                    tb.add_scalar("optimization/rollback", 1.0, step)
                    tb.add_scalar("optimization/learning_rate_after_rollback", new_lr, step)
                    tb.add_scalar("optimization/rollback_noise_norm", rollback_noise_norm, step)
                    print(
                        f"rollback at step={step:05d}: restored best latent from step {best_step}, "
                        f"best_l1={best_l1:.2f}, new_lr={new_lr:.6g}, "
                        f"rollback_noise_norm={rollback_noise_norm:.6f}"
                    )

            learning_rate = optimizer.param_groups[0]["lr"]
            lrs = optimizer_lrs(optimizer)
            tb.add_scalar("loss/total", float(loss.detach().cpu()), step)
            tb.add_scalar("loss/hash_surrogate", float(hash_loss.detach().cpu()), step)
            tb.add_scalar("loss/latent_l2", float(latent_l2.detach().cpu()), step)
            tb.add_scalar("loss/prompt_l2", float(prompt_l2.detach().cpu()), step)
            tb.add_scalar("loss/image_anchor", float(image_anchor_loss.detach().cpu()), step)
            tb.add_scalar("loss/total_variation", float(tv_loss.detach().cpu()), step)
            tb.add_scalar("optimization/learning_rate", float(learning_rate), step)
            tb.add_scalar("optimization/latent_learning_rate", lrs.get("latents", float(learning_rate)), step)
            if args.optimize_prompt_embeds:
                tb.add_scalar("optimization/prompt_learning_rate", lrs.get("prompt_embeds", args.prompt_learning_rate), step)
            tb.add_scalar("optimization/grad_norm", grad_norm, step)
            tb.add_scalar("optimization/clipped_grad_norm", clipped_grad_norm, step)
            tb.add_scalar("optimization/prompt_grad_norm", prompt_grad_norm, step)
            tb.add_scalar("optimization/clipped_prompt_grad_norm", clipped_prompt_grad_norm, step)
            tb.add_scalar("optimization/latent_update_norm", latent_update_norm, step)
            tb.add_scalar("optimization/prompt_update_norm", prompt_update_norm, step)
            tb.add_scalar("optimization/active_prompt_index", active_prompt_index, step)
            tb.add_scalar("diagnostics/image_delta", image_delta, step)
            tb.add_scalar("diagnostics/logit_delta", logit_delta, step)
            tb.add_scalar("hash/current_min_logit_l1", current_min_logit_l1, step)
            if should_check:
                tb.add_scalar("hash/current_min_l1", current_min_l1, step)
                tb.add_scalar("hash/best_l1", best_l1, step)
                tb.add_scalar("optimization/step", step, step)

            if step == 1 or step % args.image_save_interval == 0 or success or step == args.optimization_steps:
                grid = image_grid(images)
                tb.add_image("images/generated_samples", grid, step)
                save_tensor_grid(images, run_dir / "samples" / f"step_{step:06d}.png")
                if best_image is not None:
                    tb.add_image("images/best", best_image.squeeze(0), step)

            if step % args.checkpoint_interval == 0 or success or step == args.optimization_steps:
                save_checkpoint(
                    run_dir / "checkpoints" / f"checkpoint_step_{step:06d}.pt",
                    step,
                    latents,
                    prompt_embeds,
                    optimizer,
                    {
                        "best_l1": best_l1,
                        "best_step": best_step,
                        "best_index": best_index,
                        "success": success,
                    },
                    args,
                )

            metrics_logger.write(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "hash_surrogate_loss": float(hash_loss.detach().cpu()),
                    "latent_l2": float(latent_l2.detach().cpu()),
                    "prompt_l2": float(prompt_l2.detach().cpu()),
                    "image_anchor": float(image_anchor_loss.detach().cpu()),
                    "total_variation": float(tv_loss.detach().cpu()),
                    "current_min_l1": current_min_l1,
                    "best_l1": best_l1,
                    "best_step": best_step,
                    "best_index": best_index,
                    "grad_norm": grad_norm,
                    "clipped_grad_norm": clipped_grad_norm,
                    "prompt_grad_norm": prompt_grad_norm,
                    "clipped_prompt_grad_norm": clipped_prompt_grad_norm,
                    "latent_update_norm": latent_update_norm,
                    "prompt_update_norm": prompt_update_norm,
                    "current_min_logit_l1": current_min_logit_l1,
                    "image_delta": image_delta,
                    "logit_delta": logit_delta,
                    "active_prompt_index": active_prompt_index,
                    "active_prompt": prompt_pool_prompts[active_prompt_index] if prompt_pool_prompts is not None else args.prompt,
                    "learning_rate": float(learning_rate),
                    "learning_rates": lrs,
                    "success": success,
                }
            )

            previous_images = images.detach()
            previous_logits = logits.detach()
            previous_prompt_index = active_prompt_index

            del images, logits, hash_loss, latent_l2, prompt_l2, image_anchor_loss, tv_loss, loss
            if should_check:
                del quantized, l1_distances, current_min_l1_tensor, current_best_index_tensor

            if (
                args.empty_cache_interval > 0
                and device.type == "cuda"
                and step % args.empty_cache_interval == 0
            ):
                torch.cuda.empty_cache()

            if success:
                print(f"Reached target threshold: best_l1={best_l1:.2f} <= {args.threshold:.2f}")
                break

        best_hash_details = certphash.hash_details(best_quantized) if best_quantized is not None else None
        summary = {
            "success": success,
            "threshold": args.threshold,
            "best_l1": best_l1,
            "best_step": best_step,
            "best_index": best_index,
            "target_hash": asdict(target_hash.details),
            "best_hash": asdict(best_hash_details) if best_hash_details is not None else None,
            "run_dir": str(run_dir),
            "tensorboard_dir": str(tensorboard_dir),
            "diffusion_trainable_parameters": diffusion.trainable_parameter_count,
            "certphash_trainable_parameters": certphash.trainable_parameter_count,
            "optimize_prompt_embeds": args.optimize_prompt_embeds,
            "selected_prompt": args.prompt,
            "resume": resume_info,
            "prompt_search_mode": args.prompt_search_mode,
            "prompt_search_space": args.prompt_search_space,
            "prompt_pool_prompts": prompt_pool_prompts,
            "prompt_search": {
                "selected_score": asdict(prompt_search_result.selected_score),
                "prompt_cache": str(prompt_search_result.prompt_cache),
                "score_cache": str(prompt_search_result.score_cache),
            }
            if prompt_search_result is not None
            else None,
        }
        write_json(run_dir / "summary.json", summary)
        print(f"Saved summary to {run_dir / 'summary.json'}")
    finally:
        tb.close()


if __name__ == "__main__":
    main()
