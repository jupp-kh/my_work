from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from certphash_wrapper import CertPhashWrapper  # type: ignore
    from config import DEFAULT_CERTPHASH_MODEL, DEFAULT_MODEL_ID, DEFAULT_RUNS_DIR, SUPPORTED_HASH_ALGORITHMS  # type: ignore
    from diffusion_wrapper import SDXLTurboEmbeddingExperiment, autocast_context, choose_device, resolve_dtype  # type: ignore
    from image_utils import image_range_loss, load_image_tensor, save_image_grid, save_image_tensor  # type: ignore
    from run_utils import (  # type: ignore
        BestState,
        all_gradients_finite,
        all_parameters_finite,
        append_csv,
        append_jsonl,
        args_to_config,
        best_checkpoint_path,
        bit_match_percentage,
        decay_optimizer_lr,
        effective_eval_interval,
        ensure_run_layout,
        evaluate_exact,
        gradient_norm,
        is_better_result,
        load_checkpoint_into_run,
        next_checkpoint_path,
        optimizer_lr,
        resolve_resume_checkpoint,
        save_checkpoint,
        save_embedding_file,
        set_reproducibility,
        software_report,
        unique_run_dir,
        write_json,
    )
    from tensorboard_utils import log_embedding_histograms, log_tensorboard_images  # type: ignore
else:
    from .certphash_wrapper import CertPhashWrapper
    from .config import DEFAULT_CERTPHASH_MODEL, DEFAULT_MODEL_ID, DEFAULT_RUNS_DIR, SUPPORTED_HASH_ALGORITHMS
    from .diffusion_wrapper import SDXLTurboEmbeddingExperiment, autocast_context, choose_device, resolve_dtype
    from .image_utils import image_range_loss, load_image_tensor, save_image_grid, save_image_tensor
    from .run_utils import (
        BestState,
        all_gradients_finite,
        all_parameters_finite,
        append_csv,
        append_jsonl,
        args_to_config,
        best_checkpoint_path,
        bit_match_percentage,
        decay_optimizer_lr,
        effective_eval_interval,
        ensure_run_layout,
        evaluate_exact,
        gradient_norm,
        is_better_result,
        load_checkpoint_into_run,
        next_checkpoint_path,
        optimizer_lr,
        resolve_resume_checkpoint,
        save_checkpoint,
        save_embedding_file,
        set_reproducibility,
        software_report,
        unique_run_dir,
        write_json,
    )
    from .tensorboard_utils import log_embedding_histograms, log_tensorboard_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize SDXL-Turbo placeholder-token embeddings toward a target image and CertPhash hash."
    )
    parser.add_argument("--target-image", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="a photograph of <hash-concept>")
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--placeholder-token", type=str, default="<hash-concept>")
    parser.add_argument("--initializer-token", type=str, default="image")
    parser.add_argument("--num-vectors", type=int, default=1)
    parser.add_argument("--text-encoder-selection", choices=["first", "second", "both"], default="both")

    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--diffusion-variant", type=str, default=None)
    parser.add_argument("--certphash-model", type=Path, default=DEFAULT_CERTPHASH_MODEL)
    parser.add_argument("--hash-algorithm", choices=SUPPORTED_HASH_ALGORITHMS, default="certphash")
    parser.add_argument("--model-input-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--mixed-precision", choices=["auto", "no", "fp16", "bf16"], default="auto")
    parser.add_argument("--no-safetensors", action="store_true")

    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--inference-steps", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-latent-samples", type=int, default=1)

    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clipping", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--nonfinite-lr-factor", type=float, default=0.5)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--l1-threshold", type=float, default=1800.0)
    parser.add_argument("--early-stopping", dest="early_stopping", action="store_true", default=True)
    parser.add_argument("--no-early-stopping", dest="early_stopping", action="store_false")

    parser.add_argument("--hash-weight", type=float, default=1.0)
    parser.add_argument("--l1-weight", type=float, default=1.0)
    parser.add_argument("--embedding-reg-weight", type=float, default=1e-4)
    parser.add_argument("--image-range-weight", type=float, default=0.1)
    parser.add_argument("--hash-loss", choices=["smooth_l1", "l1", "mse"], default="smooth_l1")
    parser.add_argument("--hash-scale", type=float, default=255.0)
    parser.add_argument("--smooth-l1-beta", type=float, default=0.02)

    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--image-interval", type=int, default=10)
    parser.add_argument("--tensorboard-interval", type=int, default=1)
    parser.add_argument("--disable-tensorboard", action="store_true")
    parser.add_argument("--adaptive-eval", dest="adaptive_eval", action="store_true", default=True)
    parser.add_argument("--no-adaptive-eval", dest="adaptive_eval", action="store_false")
    parser.add_argument("--adaptive-eval-factor", type=float, default=2.0)
    parser.add_argument("--adaptive-eval-close-delta", type=float, default=100.0)
    parser.add_argument("--adaptive-eval-best-l1-threshold", type=float, default=100.0)
    parser.add_argument("--min-eval-interval", type=int, default=1)

    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--run-name", type=str, default="embedding_attack")
    parser.add_argument("--run-dir", type=Path, default=None, help="Existing run directory, mainly for --resume-from latest.")
    parser.add_argument("--resume-from", type=str, default=None, help="Checkpoint path, or 'latest'.")
    parser.add_argument("--new-run-on-resume", action="store_true", help="Create a new run even when --resume-from is used.")

    parser.add_argument("--attention-slicing", dest="attention_slicing", action="store_true", default=True)
    parser.add_argument("--no-attention-slicing", dest="attention_slicing", action="store_false")
    parser.add_argument("--vae-slicing", dest="vae_slicing", action="store_true", default=True)
    parser.add_argument("--no-vae-slicing", dest="vae_slicing", action="store_false")
    parser.add_argument("--vae-tiling", action="store_true")
    parser.add_argument("--vae-float32", dest="vae_float32", action="store_true", default=True)
    parser.add_argument("--no-vae-float32", dest="vae_float32", action="store_false")
    parser.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.height <= 0 or args.width <= 0:
        raise ValueError("--height and --width must be positive.")
    if args.inference_steps < 1:
        raise ValueError("--inference-steps must be >= 1.")
    if args.steps < 1:
        raise ValueError("--steps must be >= 1.")
    if args.num_vectors < 1:
        raise ValueError("--num-vectors must be >= 1.")
    if args.num_latent_samples < 1:
        raise ValueError("--num-latent-samples must be >= 1.")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be >= 1.")
    if not 0.0 < args.nonfinite_lr_factor <= 1.0:
        raise ValueError("--nonfinite-lr-factor must be in (0, 1].")
    if args.min_learning_rate < 0:
        raise ValueError("--min-learning-rate must be >= 0.")
    if args.eval_interval < 1 or args.checkpoint_interval < 1 or args.image_interval < 1:
        raise ValueError("Intervals must be >= 1.")
    if args.tensorboard_interval < 1:
        raise ValueError("--tensorboard-interval must be >= 1.")
    if args.min_eval_interval < 1:
        raise ValueError("--min-eval-interval must be >= 1.")
    if args.adaptive_eval_factor < 1.0:
        raise ValueError("--adaptive-eval-factor must be >= 1.")
    if args.adaptive_eval_close_delta < 0:
        raise ValueError("--adaptive-eval-close-delta must be >= 0.")
    if args.adaptive_eval_best_l1_threshold < 0:
        raise ValueError("--adaptive-eval-best-l1-threshold must be >= 0.")
    if not args.target_image.exists():
        raise FileNotFoundError(f"Target image not found: {args.target_image}")


def maybe_write_config(run_dir: Path, config: dict[str, Any], is_resume: bool) -> None:
    config_path = run_dir / "config.json"
    if is_resume and config_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        write_json(run_dir / f"resume_config_{timestamp}.json", config)
    else:
        write_json(config_path, config)


def open_tensorboard_writer(run_dir: Path, start_step: int, disabled: bool) -> Any | None:
    if disabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("TensorBoard is not installed; continuing without TensorBoard logs.")
        return None
    return SummaryWriter(log_dir=str(run_dir / "tensorboard"), purge_step=start_step + 1 if start_step else None)


def nonfinite_component_names(components: dict[str, torch.Tensor]) -> list[str]:
    names = []
    for name, tensor in components.items():
        if not torch.isfinite(tensor.detach()).all():
            names.append(name)
    return names


def nonfinite_generation_diagnostics(experiment: SDXLTurboEmbeddingExperiment) -> list[str]:
    return [name for name, is_finite in experiment.last_generation_diagnostics.items() if not is_finite]


def oom_message(args: argparse.Namespace) -> str:
    return (
        "CUDA out of memory during the differentiable SDXL forward. "
        "Try: --height 384 --width 384 --mixed-precision auto --vae-float32 "
        "--gradient-checkpointing --num-latent-samples 1. "
        "If it still OOMs, use --height 256 --width 256. "
        "Also start the process with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True."
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_reproducibility(args.seed)
    device = choose_device(args.device)
    dtype = resolve_dtype(args.mixed_precision, device)
    if device.type == "cpu":
        print("Warning: running SDXL-Turbo on CPU will be very slow; using float32.")

    resume_checkpoint = resolve_resume_checkpoint(args)
    is_resume = resume_checkpoint is not None and not args.new_run_on_resume
    run_dir = resume_checkpoint.parent.parent if is_resume else unique_run_dir(args.runs_dir, args.run_name, args.target_image)
    paths = ensure_run_layout(run_dir)
    config = args_to_config(args, run_dir, dtype, device)
    maybe_write_config(run_dir, config, is_resume=is_resume)

    print(f"Run directory: {run_dir}")
    print("TensorBoard command:")
    print(f'tensorboard --logdir "{paths["tensorboard"]}"')
    print(f"Device: {device}")
    print(f"Model dtype: {dtype}")
    print(f"L1 threshold: {args.l1_threshold}")

    target_image = load_image_tensor(args.target_image, device=device, size=(args.height, args.width))
    save_image_tensor(target_image, run_dir / "target.png")

    certphash = CertPhashWrapper(
        checkpoint_path=args.certphash_model,
        device=device,
        model_input_size=args.model_input_size,
    )
    target_hash = certphash.target_from_image(target_image, source=str(args.target_image))
    write_json(run_dir / "target_hash.json", asdict(target_hash.details) | {"source": target_hash.source})
    if certphash.trainable_parameter_count != 0:
        raise RuntimeError("CertPhash model unexpectedly has trainable parameters.")

    experiment = SDXLTurboEmbeddingExperiment(args, device=device, dtype=dtype)
    if experiment.frozen_trainable_parameter_count != 0:
        raise RuntimeError("A pretrained diffusion parameter outside placeholder embeddings is trainable.")

    optimizer = torch.optim.AdamW(
        experiment.trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    fixed_latents = experiment.initial_latents(args.num_latent_samples, seed=args.seed)
    best = BestState()
    start_step = 0
    resume_history: list[dict[str, Any]] = []

    if resume_checkpoint is not None:
        payload, fixed_latents, best, start_step, resume_history = load_checkpoint_into_run(
            resume_checkpoint,
            experiment,
            optimizer,
            lr_scheduler,
            device,
        )
        resume_history.append(
            {
                "checkpoint": str(resume_checkpoint),
                "resumed_at": datetime.now().isoformat(),
                "checkpoint_step": start_step,
            }
        )
        checkpoint_config = payload.get("config", {})
        if checkpoint_config:
            critical_keys = [
                "height",
                "width",
                "num_vectors",
                "placeholder_token",
                "text_encoder_selection",
                "num_latent_samples",
                "model_id",
            ]
            mismatches = [key for key in critical_keys if str(checkpoint_config.get(key)) != str(config.get(key))]
            if mismatches:
                raise ValueError(
                    "Resume configuration does not match checkpoint for keys: "
                    + ", ".join(mismatches)
                    + ". Use the original run settings or start a new run."
                )
        print(f"Resumed from {resume_checkpoint} at global step {start_step}.")

    writer = open_tensorboard_writer(run_dir, start_step, args.disable_tensorboard)
    if writer is not None:
        writer.add_text("run/tensorboard_command", f'tensorboard --logdir "{paths["tensorboard"]}"', start_step)
        writer.add_text("target/hash_base64", target_hash.details.base64, start_step)
        writer.add_scalar("threshold/l1", args.l1_threshold, start_step)
        writer.add_image("images/target", target_image.detach().cpu().squeeze(0), start_step)

    save_embedding_file(run_dir / "learned_embedding.pt", experiment)
    training_started = time.time()
    success = best.l1_distance <= args.l1_threshold
    final_images: torch.Tensor | None = None
    best_embedding_state = experiment.embedding_state()
    nonfinite_recovery_count = 0
    last_eval_l1_distance = best.l1_distance

    try:
        for step in range(start_step + 1, args.steps + 1):
            step_started = time.time()
            optimizer.zero_grad(set_to_none=True)
            micro_latents = fixed_latents.chunk(min(args.gradient_accumulation_steps, fixed_latents.shape[0]), dim=0)
            total_loss_value = 0.0
            hash_loss_value = 0.0
            normalized_l1_value = 0.0
            embedding_reg_value = 0.0
            range_loss_value = 0.0
            skipped_step = False
            skip_reason = ""
            nonfinite_components: list[str] = []
            current_images_for_log: torch.Tensor | None = None

            for chunk in micro_latents:
                chunk_weight = float(chunk.shape[0]) / float(fixed_latents.shape[0])
                try:
                    with autocast_context(device, dtype):
                        images = experiment.generate(chunk)
                except torch.OutOfMemoryError as exc:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    row = {
                        "step": step,
                        "best_l1_distance": best.l1_distance,
                        "best_hash_distance": best.hash_distance,
                        "learning_rate": optimizer_lr(optimizer),
                        "l1_threshold": args.l1_threshold,
                        "success": success,
                        "skipped_step": True,
                        "skip_reason": "cuda_out_of_memory",
                    }
                    append_jsonl(run_dir / "metrics.jsonl", row)
                    append_csv(run_dir / "metrics.csv", row)
                    raise RuntimeError(oom_message(args)) from exc
                current_images_for_log = images.detach().clamp(0.0, 1.0)
                hash_features = certphash.continuous_features(images)
                hash_surrogate_loss = certphash.surrogate_loss(
                    hash_features,
                    target_hash.quantized,
                    loss_type=args.hash_loss,
                    hash_scale=args.hash_scale,
                    smooth_l1_beta=args.smooth_l1_beta,
                )
                target_chunk = target_image.expand(images.shape[0], -1, -1, -1).to(images.device)
                normalized_l1_loss = F.l1_loss(images.float() * 255.0, target_chunk * 255.0)
                embedding_reg = experiment.embedding_regularization()
                range_loss = image_range_loss(images.float())
                total_loss = (
                    args.hash_weight * hash_surrogate_loss
                    + args.l1_weight * normalized_l1_loss
                    + args.embedding_reg_weight * embedding_reg
                    + args.image_range_weight * range_loss
                )
                weighted_loss = total_loss * chunk_weight

                if not torch.isfinite(weighted_loss.detach()):
                    skipped_step = True
                    skip_reason = "nonfinite_loss"
                    nonfinite_components = nonfinite_component_names(
                        {
                            "images": images,
                            "hash_features": hash_features,
                            "hash_surrogate_loss": hash_surrogate_loss,
                            "normalized_l1_loss": normalized_l1_loss,
                            "embedding_regularization": embedding_reg,
                            "image_range_loss": range_loss,
                            "total_loss": total_loss,
                        }
                    )
                    nonfinite_components.extend(nonfinite_generation_diagnostics(experiment))
                    break
                weighted_loss.backward()
                total_loss_value += float(total_loss.detach().cpu()) * chunk_weight
                hash_loss_value += float(hash_surrogate_loss.detach().cpu()) * chunk_weight
                normalized_l1_value += float(normalized_l1_loss.detach().cpu()) * chunk_weight
                embedding_reg_value += float(embedding_reg.detach().cpu()) * chunk_weight
                range_loss_value += float(range_loss.detach().cpu()) * chunk_weight

            if skipped_step or not all_gradients_finite(experiment.trainable_parameters):
                if not skipped_step:
                    skip_reason = "nonfinite_gradient"
                    nonfinite_components = nonfinite_component_names(
                        {
                            f"grad_{index}": parameter.grad
                            for index, parameter in enumerate(experiment.trainable_parameters)
                            if parameter.grad is not None
                        }
                    )
                optimizer.zero_grad(set_to_none=True)
                experiment.load_embedding_state(best_embedding_state)
                optimizer.state.clear()
                new_lr = decay_optimizer_lr(
                    optimizer,
                    args.nonfinite_lr_factor,
                    args.min_learning_rate,
                    lr_scheduler=lr_scheduler,
                )
                nonfinite_recovery_count += 1
                row = {
                    "step": step,
                    "best_l1_distance": best.l1_distance,
                    "best_hash_distance": best.hash_distance,
                    "learning_rate": new_lr,
                    "l1_threshold": args.l1_threshold,
                    "success": success,
                    "skipped_step": True,
                    "skip_reason": skip_reason,
                    "nonfinite_components": ",".join(nonfinite_components),
                }
                append_jsonl(run_dir / "metrics.jsonl", row)
                append_csv(run_dir / "metrics.csv", row)
                if writer is not None:
                    writer.add_scalar("optimization/nonfinite_recovery", nonfinite_recovery_count, step)
                    writer.add_scalar("optimization/learning_rate_after_nonfinite", new_lr, step)
                    if nonfinite_components:
                        writer.add_text("optimization/nonfinite_components", ",".join(nonfinite_components), step)
                component_text = f" components={','.join(nonfinite_components)}" if nonfinite_components else ""
                print(f"step={step:06d} recovered: {skip_reason}{component_text} lr={new_lr:.6g}")
                continue

            grad_norm_before_clip = gradient_norm(experiment.trainable_parameters)
            if args.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(experiment.trainable_parameters, args.gradient_clipping)
            optimizer.step()
            lr_scheduler.step()

            if not all_parameters_finite(experiment.trainable_parameters):
                optimizer.zero_grad(set_to_none=True)
                experiment.load_embedding_state(best_embedding_state)
                optimizer.state.clear()
                new_lr = decay_optimizer_lr(
                    optimizer,
                    args.nonfinite_lr_factor,
                    args.min_learning_rate,
                    lr_scheduler=lr_scheduler,
                )
                nonfinite_recovery_count += 1
                row = {
                    "step": step,
                    "best_l1_distance": best.l1_distance,
                    "best_hash_distance": best.hash_distance,
                    "learning_rate": new_lr,
                    "l1_threshold": args.l1_threshold,
                    "success": success,
                    "skipped_step": True,
                    "skip_reason": "nonfinite_parameters",
                }
                append_jsonl(run_dir / "metrics.jsonl", row)
                append_csv(run_dir / "metrics.csv", row)
                if writer is not None:
                    writer.add_scalar("optimization/nonfinite_recovery", nonfinite_recovery_count, step)
                    writer.add_scalar("optimization/learning_rate_after_nonfinite", new_lr, step)
                print(f"step={step:06d} recovered: nonfinite_parameters lr={new_lr:.6g}")
                continue

            active_eval_interval = effective_eval_interval(
                args.eval_interval,
                best_l1_distance=best.l1_distance,
                last_eval_l1_distance=last_eval_l1_distance,
                close_delta=args.adaptive_eval_close_delta,
                best_l1_threshold=args.adaptive_eval_best_l1_threshold,
                factor=args.adaptive_eval_factor,
                min_interval=args.min_eval_interval,
                enabled=args.adaptive_eval,
            )
            should_eval = step == 1 or step % active_eval_interval == 0 or step == args.steps
            should_log_tb = writer is not None and (step % args.tensorboard_interval == 0 or should_eval)
            should_log_image = step == 1 or ( step % (args.image_interval if step <= 100 else 50) == 0 ) or step == args.steps
            exact = None

            if should_eval:
                with torch.no_grad(), autocast_context(device, dtype):
                    eval_images = experiment.generate(fixed_latents)
                final_images = eval_images.detach().clamp(0.0, 1.0).cpu()
                exact = evaluate_exact(
                    images=eval_images,
                    target_image=target_image,
                    target_hash=target_hash,
                    certphash=certphash,
                    height=args.height,
                    width=args.width,
                )
                if is_better_result(exact, total_loss_value, best):
                    best.l1_distance = float(exact["exact_l1_distance"])
                    best.hash_distance = float(exact["exact_hash_distance"])
                    best.exact_rgb_l1_distance = float(exact["exact_rgb_l1_distance"])
                    best.mean_absolute_pixel_difference = float(exact["mean_absolute_pixel_difference"])
                    best.loss = total_loss_value
                    best.step = step
                    best.image = exact["image"]
                    best.quantized_hash = exact["quantized_hash"]
                    best.hash_details = exact["hash_details"]
                    best_embedding_state = experiment.embedding_state()
                    save_image_tensor(best.image, run_dir / "best.png")
                    save_embedding_file(run_dir / "learned_embedding.pt", experiment)
                    best_path = best_checkpoint_path(paths["checkpoints"])
                    best.checkpoint_path = str(best_path)
                    save_checkpoint(
                        best_path,
                        step=step,
                        experiment=experiment,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        best=best,
                        fixed_latents=fixed_latents,
                        config=config,
                        resume_history=resume_history,
                    )
                last_eval_l1_distance = float(exact["exact_l1_distance"])
                success = best.l1_distance <= args.l1_threshold
                print(
                    f"step={step:06d} loss={total_loss_value:.6f} "
                    f"surrogate={hash_loss_value:.6f} l1={exact['exact_l1_distance']:.2f} "
                    f"best_l1={best.l1_distance:.2f} hash={exact['exact_hash_distance']:.2f} "
                    f"mean_abs_pixel={exact['mean_absolute_pixel_difference']:.6f} "
                    f"eval_interval={active_eval_interval}"
                )
            else:
                print(
                    f"step={step:06d} loss={total_loss_value:.6f} "
                    f"surrogate={hash_loss_value:.6f} grad_norm={grad_norm_before_clip:.6f}"
                )

            runtime = time.time() - step_started
            exact_l1 = float(exact["exact_l1_distance"]) if exact is not None else math.nan
            exact_hash = float(exact["exact_hash_distance"]) if exact is not None else math.nan
            rgb_l1 = float(exact["exact_rgb_l1_distance"]) if exact is not None else math.nan
            mean_abs_pixel = float(exact["mean_absolute_pixel_difference"]) if exact is not None else math.nan
            bit_match = exact["bit_match_percentage"] if exact is not None else None
            row = {
                "step": step,
                "total_loss": total_loss_value,
                "hash_surrogate_loss": hash_loss_value,
                "normalized_l1_loss": normalized_l1_value,
                "embedding_regularization": embedding_reg_value,
                "image_range_loss": range_loss_value,
                "exact_l1_distance": exact_l1,
                "best_l1_distance": best.l1_distance,
                "exact_rgb_l1_distance": rgb_l1,
                "mean_absolute_pixel_difference": mean_abs_pixel,
                "best_mean_absolute_pixel_difference": best.mean_absolute_pixel_difference,
                "l1_threshold": args.l1_threshold,
                "l1_minus_threshold": exact_l1 - args.l1_threshold if not math.isnan(exact_l1) else math.nan,
                "exact_hash_distance": exact_hash,
                "best_hash_distance": best.hash_distance,
                "bit_match_percentage": bit_match,
                "learning_rate": optimizer_lr(optimizer),
                "gradient_norm": grad_norm_before_clip,
                "embedding_norm": experiment.embedding_norm(),
                "embedding_displacement": experiment.embedding_displacement(),
                "runtime_seconds": runtime,
                "active_eval_interval": active_eval_interval,
                "best_step": best.step,
                "success": success,
                "skipped_step": False,
                "skip_reason": "",
            }
            append_jsonl(run_dir / "metrics.jsonl", row)
            append_csv(run_dir / "metrics.csv", row)

            if should_log_tb and writer is not None:
                writer.add_scalar("loss/total", total_loss_value, step)
                writer.add_scalar("loss/hash_surrogate", hash_loss_value, step)
                writer.add_scalar("loss/normalized_l1_training", normalized_l1_value, step)
                writer.add_scalar("loss/embedding_regularization", embedding_reg_value, step)
                writer.add_scalar("loss/image_range", range_loss_value, step)
                writer.add_scalar("metrics/exact_summed_l1_distance", exact_l1, step)
                writer.add_scalar("metrics/best_real_l1_distance", best.l1_distance, step)
                writer.add_scalar("metrics/mean_absolute_pixel_difference", mean_abs_pixel, step)
                writer.add_scalar("metrics/l1_threshold", args.l1_threshold, step)
                writer.add_scalar("metrics/l1_minus_threshold", row["l1_minus_threshold"], step)
                writer.add_scalar("metrics/exact_hash_distance", exact_hash, step)
                if bit_match is not None:
                    writer.add_scalar("metrics/bit_match_percentage", bit_match, step)
                writer.add_scalar("optimization/learning_rate", optimizer_lr(optimizer), step)
                writer.add_scalar("optimization/gradient_norm", grad_norm_before_clip, step)
                writer.add_scalar("optimization/embedding_norm", row["embedding_norm"], step)
                writer.add_scalar("optimization/embedding_displacement", row["embedding_displacement"], step)
                writer.add_scalar("optimization/active_eval_interval", active_eval_interval, step)
                writer.add_scalar("runtime/seconds_per_step", runtime, step)
                log_embedding_histograms(writer, step, experiment)

            if should_log_image:
                images_to_save = (
                    final_images
                    if exact is not None and final_images is not None
                    else current_images_for_log.detach().cpu()
                )
                save_image_grid(images_to_save, paths["images"] / f"step_{step:06d}.png")
                if writer is not None:
                    log_tensorboard_images(
                        writer,
                        step,
                        target_image=target_image,
                        current_images=images_to_save,
                        best_image=best.image,
                    )

            if step % args.checkpoint_interval == 0:
                save_checkpoint(
                    next_checkpoint_path(paths["checkpoints"], step),
                    step=step,
                    experiment=experiment,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    best=best,
                    fixed_latents=fixed_latents,
                    config=config,
                    resume_history=resume_history,
                )

            if success and args.early_stopping:
                print(f"Reached L1 threshold: best_l1={best.l1_distance:.2f} <= {args.l1_threshold:.2f}")
                break
    finally:
        if writer is not None:
            writer.flush()
            writer.close()

    if final_images is None:
        with torch.no_grad(), autocast_context(device, dtype):
            eval_images = experiment.generate(fixed_latents)
        final_images = eval_images.detach().clamp(0.0, 1.0).cpu()
    save_image_tensor(final_images[:1], run_dir / "final.png")
    if best.image is None:
        best.image = final_images[:1]
        save_image_tensor(best.image, run_dir / "best.png")
    save_embedding_file(run_dir / "learned_embedding.pt", experiment)

    duration = time.time() - training_started
    final_success = best.l1_distance <= args.l1_threshold
    report = {
        "success": final_success,
        "l1_threshold": args.l1_threshold,
        "best_l1_distance": best.l1_distance,
        "best_mean_absolute_pixel_difference": best.mean_absolute_pixel_difference,
        "best_exact_rgb_l1_distance": best.exact_rgb_l1_distance,
        "best_training_step": best.step,
        "best_checkpoint_path": best.checkpoint_path,
        "target_hash": asdict(target_hash.details),
        "best_generated_image_hash": asdict(best.hash_details) if best.hash_details is not None else None,
        "exact_hash_distance": best.hash_distance,
        "bit_match_percentage": bit_match_percentage(best.hash_details.bit_string, target_hash.details.bit_string)
        if best.hash_details is not None
        else None,
        "model_configuration": {
            "model_id": args.model_id,
            "height": args.height,
            "width": args.width,
            "inference_steps": args.inference_steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "num_latent_samples": args.num_latent_samples,
            "mixed_precision": args.mixed_precision,
            "dtype": str(dtype),
        },
        "loss_weights": {
            "hash_weight": args.hash_weight,
            "l1_weight": args.l1_weight,
            "embedding_reg_weight": args.embedding_reg_weight,
            "image_range_weight": args.image_range_weight,
        },
        "training_duration_seconds": duration,
        "resume_history": resume_history,
        "software": software_report(device),
        "run_dir": str(run_dir),
        "tensorboard_command": f'tensorboard --logdir "{paths["tensorboard"]}"',
        "success_definition": "success = best_l1_distance <= l1_threshold",
        "l1_distance_definition": "Exact CertPhash byte L1 between generated image hash and target image hash.",
        "pixel_l1_definition": "exact_rgb_l1_distance is summed abs difference of rounded [0,255] RGB values.",
    }
    write_json(run_dir / "final_report.json", report)
    print(f"Final report: {run_dir / 'final_report.json'}")
    print(f"Best L1 distance: {best.l1_distance:.2f}")
    print(f"Threshold reached: {final_success}")


if __name__ == "__main__":
    main()
