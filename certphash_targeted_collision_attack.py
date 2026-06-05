from __future__ import annotations

import argparse
import base64
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


ROOT = Path(__file__).resolve().parents[1]
CERTPHASH_ATTACK = ROOT / "CertPhash" / "attack"
DEFAULT_MODEL = ROOT / "CertPhash" / "train_verify" / "saved_models" / "coco_photodna_ep8" / "ckpt_best.pth"
DEFAULT_IMAGE_DIR = ROOT / "CertPhash" / "train_verify" / "data" / "coco100x100_val"
DEFAULT_OUTPUT_DIR = ROOT / "my_work" / "results" / "certphash_collider"
DEFAULT_TENSORBOARD_DIR = ROOT / "my_work" / "results" / "tensorboard" / "certphash_collider"
DEFAULT_MODEL_INPUT_SIZE = 64
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}

sys.path.insert(0, str(CERTPHASH_ATTACK))
from models.resnet_v5 import resnet_v5  # noqa: E402


@dataclass
class HashResult:
    quantized: list[float]
    bytes: list[int]
    bit_string: str
    base64: str


@dataclass
class AttackMetrics:
    step: int
    loss: float
    target_loss: float
    source_l1: float
    target_image_l1: float
    byte_l1: float
    bit_hamming: int
    linf: float
    l2: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Targeted CertPHash collision attack. Samples two images, keeps the second image's "
            "CertPHash as the target, and optimizes a perturbation of the first image toward it."
        )
    )
    parser.add_argument("--source", type=Path, default=None, help="Image to modify. If omitted, sampled randomly.")
    parser.add_argument("--target", type=Path, default=None, help="Image whose hash should be reached.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR, help="Directory used for random sampling.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="CertPHash checkpoint path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Where outputs are saved.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for image sampling and torch.")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--optimize-size", choices=["model", "original"], default="model", help="Optimize the 64x64 model input or the original-resolution image.")
    parser.add_argument("--model-input-size", type=int, default=DEFAULT_MODEL_INPUT_SIZE, help="CertPHash model input size used before the network.")
    parser.add_argument("--steps", type=int, default=3000, help="Optimization steps.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Adam learning rate for the image perturbation.")
    parser.add_argument("--epsilon", type=float, default=16.0 / 255.0, help="L-infinity perturbation budget in [0, 1].")
    parser.add_argument("--unbounded", action="store_true", help="Disable the epsilon bound and only clamp pixels to [0, 1].")
    parser.add_argument("--check-every", type=int, default=25, help="How often to evaluate the discrete hash.")
    parser.add_argument("--success-bit-hamming", type=int, default=0, help="Stop once bit hamming distance is <= this.")
    parser.add_argument("--success-byte-l1", type=float, default=0.0, help="Stop once quantized byte L1 is <= this.")
    parser.add_argument("--similarity-weight", type=float, default=0.0, help="Optional L1 penalty to stay near the source image.")
    parser.add_argument("--target-image-weight", type=float, default=0.0, help="Optional L1 penalty to move pixels toward the target image.")
    parser.add_argument("--tv-weight", type=float, default=0.0, help="Optional L1 total-variation penalty for smoother noise.")
    parser.add_argument("--hash-loss", choices=["l1", "mse", "smooth_l1"], default="l1", help="Loss used to optimize the continuous hash output.")
    parser.add_argument("--smooth-l1-beta", type=float, default=0.02, help="Beta parameter for --hash-loss smooth_l1.")
    parser.add_argument("--hash-scale", type=float, default=255.0, help="Scale for the continuous hash loss.")
    parser.add_argument("--shrink-steps", type=int, default=20, help="Line-search steps to reduce perturbation after success.")
    parser.add_argument("--save-progress", action="store_true", help="Save the current best image when it improves.")
    parser.add_argument("--tensorboard", action="store_true", help="Write TensorBoard logs for losses, distances, and images.")
    parser.add_argument("--tb-dir", type=Path, default=DEFAULT_TENSORBOARD_DIR, help="TensorBoard log directory.")
    parser.add_argument("--tb-image-every", type=int, default=100, help="How often to log adversarial images to TensorBoard.")
    parser.add_argument("--hash-only", action="store_true", help="Only compute hashes/distances for images, then exit.")
    parser.add_argument("--hash-images", type=Path, nargs="+", default=None, help="Images to hash in --hash-only mode.")
    parser.add_argument("--hash-reference", type=Path, default=None, help="Reference image for hash distance in --hash-only mode.")
    parser.add_argument("--hash-output", type=Path, default=None, help="Optional JSON path for --hash-only results.")
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def iter_images(folder: Path) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(folder):
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path


def pick_source_and_target(args: argparse.Namespace) -> tuple[Path, Path]:
    rng = random.Random(args.seed)
    source = args.source
    target = args.target

    if source is not None and target is not None:
        return source, target

    images = sorted(iter_images(args.image_dir))
    if len(images) < 2:
        raise ValueError(f"Need at least two images in {args.image_dir}")

    if source is None and target is None:
        source, target = rng.sample(images, 2)
    elif source is None:
        choices = [path for path in images if path.resolve() != target.resolve()]
        source = rng.choice(choices)
    elif target is None:
        choices = [path for path in images if path.resolve() != source.resolve()]
        target = rng.choice(choices)

    if source.resolve() == target.resolve():
        raise ValueError("Source and target must be different images.")
    return source, target


def load_image(path: Path, device: torch.device, image_size: int | None = DEFAULT_MODEL_INPUT_SIZE) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if image_size is not None:
        image = image.resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def save_image(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor.detach().cpu().clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).numpy()
    image = (image * 255.0).round().astype(np.uint8)
    Image.fromarray(image).save(path)


def tensor_to_uint8_image(tensor: torch.Tensor) -> Image.Image:
    image = tensor.detach().cpu().clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).numpy()
    image = (image * 255.0).round().astype(np.uint8)
    return Image.fromarray(image)


def resize_tensor(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if image.shape[-2:] == size:
        return image
    return F.interpolate(image, size=size, mode="bilinear", align_corners=False)


def save_comparison(source: torch.Tensor, target: torch.Tensor, adv: torch.Tensor, path: Path) -> None:
    target = resize_tensor(target, source.shape[-2:])
    adv = resize_tensor(adv, source.shape[-2:])
    delta_visual = delta_to_visual(adv - source)
    panels = [
        tensor_to_uint8_image(source),
        tensor_to_uint8_image(target),
        tensor_to_uint8_image(adv),
        tensor_to_uint8_image(delta_visual),
    ]
    width, height = panels[0].size
    canvas = Image.new("RGB", (width * len(panels), height))
    for idx, panel in enumerate(panels):
        canvas.paste(panel, (idx * width, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_delta(delta: torch.Tensor, path: Path) -> None:
    max_abs = float(delta.detach().abs().max().cpu())
    if max_abs <= 0:
        visual = torch.full_like(delta, 0.5)
    else:
        visual = (delta / (2.0 * max_abs)) + 0.5
    save_image(visual, path)


def delta_to_visual(delta: torch.Tensor) -> torch.Tensor:
    max_abs = delta.detach().abs().amax(dim=(1, 2, 3), keepdim=True)
    return torch.where(max_abs > 0, (delta / (2.0 * max_abs)) + 0.5, torch.full_like(delta, 0.5)).clamp(0.0, 1.0)


def comparison_tensor(source: torch.Tensor, target: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
    target = resize_tensor(target, source.shape[-2:])
    adv = resize_tensor(adv, source.shape[-2:])
    panels = [
        source.detach().cpu().clamp(0.0, 1.0),
        target.detach().cpu().clamp(0.0, 1.0),
        adv.detach().cpu().clamp(0.0, 1.0),
        delta_to_visual(adv - source).detach().cpu(),
    ]
    return torch.cat(panels, dim=3).squeeze(0)


def normalize_coco(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
    return (image - mean) / std


def resize_for_certphash(image: torch.Tensor, model_input_size: int = DEFAULT_MODEL_INPUT_SIZE) -> torch.Tensor:
    return resize_tensor(image, (model_input_size, model_input_size))


def certphash_logits(
    model: torch.nn.Module,
    image: torch.Tensor,
    model_input_size: int = DEFAULT_MODEL_INPUT_SIZE,
) -> torch.Tensor:
    image_for_model = resize_for_certphash(image, model_input_size)
    return model(normalize_coco(image_for_model))


def load_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    model = resnet_v5(input_dim=64)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            checkpoint = checkpoint["model"]
    cleaned = {key.removeprefix("module."): value for key, value in checkpoint.items()}
    model.load_state_dict(cleaned)
    model.to(device)
    model.eval()
    return model


def quantize_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.relu(torch.round(logits))


def hash_from_quantized(quantized: torch.Tensor) -> HashResult:
    values = quantized.detach().cpu().view(-1).numpy()
    byte_values = values.astype(np.uint8)
    byte_array = byte_values.tobytes()
    bit_array = np.unpackbits(byte_values)
    return HashResult(
        quantized=[float(v) for v in values.tolist()],
        bytes=[int(v) for v in byte_values.tolist()],
        bit_string="".join(str(int(bit)) for bit in bit_array.tolist()),
        base64=base64.b64encode(byte_array).decode("utf-8"),
    )


@torch.no_grad()
def certphash(
    model: torch.nn.Module,
    image: torch.Tensor,
    model_input_size: int = DEFAULT_MODEL_INPUT_SIZE,
) -> tuple[torch.Tensor, HashResult]:
    logits = certphash_logits(model, image, model_input_size)
    quantized = quantize_logits(logits)
    return quantized, hash_from_quantized(quantized)


def hash_distances(current_quantized: torch.Tensor, target_quantized: torch.Tensor, target_hash: HashResult) -> tuple[float, int]:
    current_hash = hash_from_quantized(current_quantized)
    current_quant = np.array(current_hash.quantized, dtype=np.float32)
    target_quant = np.array(target_hash.quantized, dtype=np.float32)
    byte_l1 = float(np.abs(current_quant - target_quant).sum())
    bit_hamming = sum(a != b for a, b in zip(current_hash.bit_string, target_hash.bit_string))
    return byte_l1, bit_hamming


def total_variation(image: torch.Tensor) -> torch.Tensor:
    horizontal = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]).mean()
    vertical = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]).mean()
    return horizontal + vertical


def hash_target_loss(logits: torch.Tensor, target_quantized: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    current = logits / args.hash_scale
    target = target_quantized / args.hash_scale
    if args.hash_loss == "l1":
        return F.l1_loss(current, target)
    if args.hash_loss == "mse":
        return F.mse_loss(current, target)
    if args.hash_loss == "smooth_l1":
        return F.smooth_l1_loss(current, target, beta=args.smooth_l1_beta)
    raise ValueError(f"Unsupported hash loss: {args.hash_loss}")


def measure(
    model: torch.nn.Module,
    source: torch.Tensor,
    adv: torch.Tensor,
    target_quantized: torch.Tensor,
    target_hash: HashResult,
    step: int,
    loss: float,
    target_loss: float,
    source_l1: float,
    target_image_l1: float,
    model_input_size: int = DEFAULT_MODEL_INPUT_SIZE,
) -> AttackMetrics:
    with torch.no_grad():
        logits = certphash_logits(model, adv, model_input_size)
        current_quantized = quantize_logits(logits)
        byte_l1, bit_hamming = hash_distances(current_quantized, target_quantized, target_hash)
        diff = adv - source
        linf = float(diff.abs().max().cpu())
        l2 = float(torch.linalg.vector_norm(diff).cpu())
    return AttackMetrics(
        step=step,
        loss=loss,
        target_loss=target_loss,
        source_l1=source_l1,
        target_image_l1=target_image_l1,
        byte_l1=byte_l1,
        bit_hamming=bit_hamming,
        linf=linf,
        l2=l2,
    )


def is_success(metrics: AttackMetrics, args: argparse.Namespace) -> bool:
    return metrics.bit_hamming <= args.success_bit_hamming and metrics.byte_l1 <= args.success_byte_l1


def clamp_delta(delta: torch.Tensor, source: torch.Tensor, args: argparse.Namespace) -> None:
    if not args.unbounded:
        delta.clamp_(-args.epsilon, args.epsilon)
    delta.copy_(torch.clamp(source + delta, 0.0, 1.0) - source)


def maybe_shrink(
    model: torch.nn.Module,
    source: torch.Tensor,
    adv: torch.Tensor,
    target_quantized: torch.Tensor,
    target_hash: HashResult,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, AttackMetrics | None]:
    best = adv.detach().clone()
    best_metrics: AttackMetrics | None = None
    low = 0.0
    high = 1.0

    for idx in range(args.shrink_steps):
        alpha = (low + high) / 2.0
        candidate = torch.clamp(source + alpha * (adv - source), 0.0, 1.0)
        metrics = measure(
            model,
            source,
            candidate,
            target_quantized,
            target_hash,
            -idx - 1,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            args.model_input_size,
        )
        if is_success(metrics, args):
            best = candidate.detach().clone()
            best_metrics = metrics
            high = alpha
        else:
            low = alpha

    return best, best_metrics


def default_hash_images(args: argparse.Namespace) -> list[Path]:
    candidates = [
        args.output_dir / "source.png",
        args.output_dir / "target.png",
        args.output_dir / "adversarial.png",
    ]
    return [path for path in candidates if path.exists()]


def run_hash_report(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    model = load_model(args.model, device)
    images = args.hash_images if args.hash_images is not None else default_hash_images(args)
    if not images:
        raise ValueError("No images to hash. Pass --hash-images or run an attack first so output images exist.")

    reference_path = args.hash_reference
    if reference_path is None and (args.output_dir / "target.png").exists():
        reference_path = args.output_dir / "target.png"

    reference_quantized = None
    reference_hash = None
    if reference_path is not None:
        reference_image = load_image(reference_path, device, image_size=None)
        reference_quantized, reference_hash = certphash(model, reference_image, args.model_input_size)

    rows = []
    for image_path in images:
        image = load_image(image_path, device, image_size=None)
        quantized, image_hash = certphash(model, image, args.model_input_size)
        row = {
            "image": str(image_path),
            "hash_base64": image_hash.base64,
            "hash_bits": image_hash.bit_string,
            "hash_quantized": image_hash.quantized,
        }
        if reference_quantized is not None and reference_hash is not None:
            byte_l1, bit_hamming = hash_distances(quantized, reference_quantized, reference_hash)
            row["reference"] = str(reference_path)
            row["byte_l1_to_reference"] = byte_l1
            row["bit_hamming_to_reference"] = bit_hamming
        rows.append(row)

    if reference_path is not None:
        print(f"Reference image: {reference_path}")
    for row in rows:
        print(row["image"])
        print(f"  hash base64: {row['hash_base64']}")
        if "byte_l1_to_reference" in row:
            print(f"  byte L1 to reference: {row['byte_l1_to_reference']:.1f}")
            print(f"  bit Hamming to reference: {row['bit_hamming_to_reference']}")

    output_path = args.hash_output or (args.output_dir / "hash_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    print(f"Saved hash report to {output_path}")


def run_attack(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = choose_device(args.device)
    source_path, target_path = pick_source_and_target(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Source image: {source_path}")
    print(f"Target image: {target_path}")
    print(f"Model: {args.model}")

    model = load_model(args.model, device)
    image_size = None if args.optimize_size == "original" else args.model_input_size
    source = load_image(source_path, device, image_size=image_size)
    target = load_image(target_path, device, image_size=image_size)
    target_for_pixels = resize_tensor(target, source.shape[-2:])
    writer: SummaryWriter | None = None

    with torch.no_grad():
        source_quantized, source_hash = certphash(model, source, args.model_input_size)
        target_quantized, target_hash = certphash(model, target, args.model_input_size)

    print(f"Initial source hash base64: {source_hash.base64}")
    print(f"Target hash base64:         {target_hash.base64}")
    initial_byte_l1, initial_bit_hamming = hash_distances(source_quantized, target_quantized, target_hash)
    print(f"Initial byte L1: {initial_byte_l1:.1f}")
    print(f"Initial bit Hamming: {initial_bit_hamming}")

    delta = torch.zeros_like(source, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=args.lr)
    best_adv = source.detach().clone()
    best_metrics: AttackMetrics | None = None
    success_metrics: AttackMetrics | None = None

    if args.tensorboard:
        run_name = f"{source_path.stem}_to_{target_path.stem}_seed{args.seed}"
        writer = SummaryWriter(log_dir=str(args.tb_dir / run_name))
        writer.add_text("paths/source", str(source_path), 0)
        writer.add_text("paths/target", str(target_path), 0)
        writer.add_text("paths/model", str(args.model), 0)
        writer.add_text("hash/source_base64", source_hash.base64, 0)
        writer.add_text("hash/target_base64", target_hash.base64, 0)
        writer.add_image("images/source", source.detach().cpu().squeeze(0), 0)
        writer.add_image("images/target", target.detach().cpu().squeeze(0), 0)
        writer.add_image("slider/adversarial", source.detach().cpu().squeeze(0), 0)
        writer.add_image("slider/comparison", comparison_tensor(source, target, source), 0)
        writer.add_scalar("hash/byte_l1", initial_byte_l1, 0)
        writer.add_scalar("hash/bit_hamming", initial_bit_hamming, 0)
        writer.add_scalar("perturbation/linf_checked", 0.0, 0)
        writer.add_scalar("perturbation/l2_checked", 0.0, 0)
        writer.add_hparams(
            {
                "lr": args.lr,
                "steps": args.steps,
                "epsilon": -1.0 if args.unbounded else args.epsilon,
                "similarity_weight": args.similarity_weight,
                "target_image_weight": args.target_image_weight,
                "tv_weight": args.tv_weight,
                "hash_loss": args.hash_loss,
                "smooth_l1_beta": args.smooth_l1_beta,
                "hash_scale": args.hash_scale,
                "optimize_size": args.optimize_size,
                "model_input_size": args.model_input_size,
            },
            {},
        )
        print(f"TensorBoard logs: {args.tb_dir / run_name}")

    try:
        for step in range(1, args.steps + 1):
            adv = torch.clamp(source + delta, 0.0, 1.0)
            logits = certphash_logits(model, adv, args.model_input_size)
            target_loss = hash_target_loss(logits, target_quantized, args)
            source_l1 = F.l1_loss(adv, source)
            target_image_l1 = F.l1_loss(adv, target_for_pixels)
            tv_loss = total_variation(adv) if args.tv_weight > 0 else torch.zeros((), device=device)
            loss = target_loss + args.similarity_weight * source_l1 + args.target_image_weight * target_image_l1 + args.tv_weight * tv_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                clamp_delta(delta, source, args)

            if writer is not None:
                writer.add_scalar("loss/total", float(loss.detach().cpu()), step)
                writer.add_scalar("loss/hash_l1", float(target_loss.detach().cpu()), step)
                writer.add_scalar("loss/source_l1", float(source_l1.detach().cpu()), step)
                writer.add_scalar("loss/target_image_l1", float(target_image_l1.detach().cpu()), step)
                writer.add_scalar("loss/total_variation", float(tv_loss.detach().cpu()), step)
                writer.add_scalar("perturbation/linf_live", float(delta.detach().abs().max().cpu()), step)
                writer.add_scalar("perturbation/l2_live", float(torch.linalg.vector_norm(delta.detach()).cpu()), step)

            should_check = step == 1 or step % args.check_every == 0 or step == args.steps
            if not should_check:
                if writer is not None and args.tb_image_every > 0 and step % args.tb_image_every == 0:
                    current_adv = torch.clamp(source + delta, 0.0, 1.0)
                    writer.add_image("images/adversarial_live", current_adv.detach().cpu().squeeze(0), step)
                    writer.add_image("images/perturbation_live", delta_to_visual(delta.detach()).cpu().squeeze(0), step)
                    writer.add_image("slider/adversarial", current_adv.detach().cpu().squeeze(0), step)
                    writer.add_image("slider/comparison", comparison_tensor(source, target, current_adv), step)
                continue

            adv = torch.clamp(source + delta, 0.0, 1.0).detach()
            metrics = measure(
                model,
                source,
                adv,
                target_quantized,
                target_hash,
                step,
                float(loss.detach().cpu()),
                float(target_loss.detach().cpu()),
                float(source_l1.detach().cpu()),
                float(target_image_l1.detach().cpu()),
                args.model_input_size,
            )
            if writer is not None:
                writer.add_scalar("hash/byte_l1", metrics.byte_l1, step)
                writer.add_scalar("hash/bit_hamming", metrics.bit_hamming, step)
                writer.add_scalar("perturbation/linf_checked", metrics.linf, step)
                writer.add_scalar("perturbation/l2_checked", metrics.l2, step)
                if args.tb_image_every > 0 and (step == 1 or step % args.tb_image_every == 0 or step == args.steps):
                    writer.add_image("images/adversarial_checked", adv.cpu().squeeze(0), step)
                    writer.add_image("images/perturbation_checked", delta_to_visual(adv - source).cpu().squeeze(0), step)
                    writer.add_image("slider/adversarial", adv.cpu().squeeze(0), step)
                    writer.add_image("slider/comparison", comparison_tensor(source, target, adv), step)

            improved = (
                best_metrics is None
                or metrics.bit_hamming < best_metrics.bit_hamming
                or (metrics.bit_hamming == best_metrics.bit_hamming and metrics.byte_l1 < best_metrics.byte_l1)
            )
            if improved:
                best_adv = adv.detach().clone()
                best_metrics = metrics
                if writer is not None:
                    writer.add_scalar("best/byte_l1", metrics.byte_l1, step)
                    writer.add_scalar("best/bit_hamming", metrics.bit_hamming, step)
                    writer.add_image("images/best_adversarial", best_adv.cpu().squeeze(0), step)
                if args.save_progress:
                    save_image(best_adv, args.output_dir / "best_progress.png")

            print(
                "step={step:5d} loss={loss:.6f} target={target:.6f} "
                "byte_l1={byte_l1:.1f} bit_hamming={bits} linf={linf:.5f}".format(
                    step=metrics.step,
                    loss=metrics.loss,
                    target=metrics.target_loss,
                    byte_l1=metrics.byte_l1,
                    bits=metrics.bit_hamming,
                    linf=metrics.linf,
                )
            )

            if is_success(metrics, args):
                success_metrics = metrics
                best_adv = adv.detach().clone()
                if writer is not None:
                    writer.add_scalar("success/step", step, step)
                print("Reached the requested target-hash threshold.")
                break

        if success_metrics is not None and args.shrink_steps > 0:
            shrunk, shrink_metrics = maybe_shrink(model, source, best_adv, target_quantized, target_hash, args)
            if shrink_metrics is not None:
                best_adv = shrunk
                best_metrics = shrink_metrics
                if writer is not None:
                    writer.add_scalar("shrink/linf", shrink_metrics.linf, args.steps)
                    writer.add_scalar("shrink/l2", shrink_metrics.l2, args.steps)
                    writer.add_image("images/adversarial_shrunk", best_adv.cpu().squeeze(0), args.steps)
                print(f"Shrank successful perturbation to linf={shrink_metrics.linf:.5f}, l2={shrink_metrics.l2:.5f}.")

        final_quantized, final_hash = certphash(model, best_adv, args.model_input_size)
        final_byte_l1, final_bit_hamming = hash_distances(final_quantized, target_quantized, target_hash)
        final_diff = best_adv - source
        final_metrics = {
            "source": str(source_path),
            "target": str(target_path),
            "model": str(args.model),
            "device": str(device),
            "loss_objective": f"hash_{args.hash_loss}",
            "optimize_size": args.optimize_size,
            "source_tensor_shape": list(source.shape),
            "target_tensor_shape": list(target.shape),
            "model_input_size": args.model_input_size,
            "steps_requested": args.steps,
            "epsilon": None if args.unbounded else args.epsilon,
            "similarity_weight": args.similarity_weight,
            "target_image_weight": args.target_image_weight,
            "tv_weight": args.tv_weight,
            "hash_loss": args.hash_loss,
            "smooth_l1_beta": args.smooth_l1_beta,
            "hash_scale": args.hash_scale,
            "success_bit_hamming": args.success_bit_hamming,
            "success_byte_l1": args.success_byte_l1,
            "initial_byte_l1": initial_byte_l1,
            "initial_bit_hamming": initial_bit_hamming,
            "final_byte_l1": final_byte_l1,
            "final_bit_hamming": final_bit_hamming,
            "final_linf": float(final_diff.abs().max().cpu()),
            "final_l2": float(torch.linalg.vector_norm(final_diff).cpu()),
            "best_checked_metrics": asdict(best_metrics) if best_metrics else None,
            "source_hash": asdict(source_hash),
            "target_hash": asdict(target_hash),
            "adversarial_hash": asdict(final_hash),
        }

        if writer is not None:
            writer.add_scalar("final/byte_l1", final_byte_l1, args.steps)
            writer.add_scalar("final/bit_hamming", final_bit_hamming, args.steps)
            writer.add_scalar("final/linf", final_metrics["final_linf"], args.steps)
            writer.add_scalar("final/l2", final_metrics["final_l2"], args.steps)
            writer.add_text("hash/adversarial_base64", final_hash.base64, args.steps)
            writer.add_image("images/final_adversarial", best_adv.cpu().squeeze(0), args.steps)
            writer.add_image("images/final_perturbation", delta_to_visual(final_diff).cpu().squeeze(0), args.steps)
            writer.add_image("slider/adversarial", best_adv.cpu().squeeze(0), args.steps)
            writer.add_image("slider/comparison", comparison_tensor(source, target, best_adv), args.steps)

        save_image(source, args.output_dir / "source.png")
        save_image(target, args.output_dir / "target.png")
        save_image(best_adv, args.output_dir / "adversarial.png")
        save_delta(best_adv - source, args.output_dir / "perturbation_visualized.png")
        save_comparison(source, target, best_adv, args.output_dir / "comparison.png")
        with open(args.output_dir / "attack_metadata.json", "w", encoding="utf-8") as handle:
            json.dump(final_metrics, handle, indent=2)

        print(f"Saved outputs to {args.output_dir}")
        print(f"Initial -> final byte L1: {initial_byte_l1:.1f} -> {final_byte_l1:.1f}")
        print(f"Initial -> final bit Hamming: {initial_bit_hamming} -> {final_bit_hamming}")
        print(f"Final byte L1: {final_byte_l1:.1f}")
        print(f"Final bit Hamming: {final_bit_hamming}")
    finally:
        if writer is not None:
            writer.flush()
            writer.close()


def main() -> None:
    args = parse_args()
    if args.hash_only:
        run_hash_report(args)
    else:
        run_attack(args)


if __name__ == "__main__":
    main()
