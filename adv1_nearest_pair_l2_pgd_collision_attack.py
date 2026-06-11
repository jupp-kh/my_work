from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
CERTPHASH_ATTACK = ROOT / "CertPhash" / "attack"
DEFAULT_MODEL = ROOT / "CertPhash" / "train_verify" / "saved_models" / "coco_photodna_ep8" / "ckpt_best.pth"
DEFAULT_PAIRS_CSV = ROOT / "my_work" / "results" / "cocoanalysis" / "coco100x100_val_hashes_coco_photodna_ep8_nearest_pairs.csv"
DEFAULT_OUTPUT_DIR = ROOT / "my_work" / "results" / "adv1_nearest_pair_l2_pgd_collision"

sys.path.insert(0, str(CERTPHASH_ATTACK))
from models.resnet_v5 import resnet_v5  # noqa: E402
from utils.image_processing import load_and_preprocess_img, normalize, save_images  # noqa: E402


def parse_l2_epsilons(value: str) -> list[float]:
    epsilons = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not epsilons:
        raise argparse.ArgumentTypeError("At least one L2 epsilon value is required.")
    if any(epsilon <= 0 for epsilon in epsilons):
        raise argparse.ArgumentTypeError("All L2 epsilon values must be positive.")
    return epsilons


def parse_positive_floats(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required.")
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("All values must be positive.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an L2-projected PGD CertPHash collision attack using a precomputed nearest-pairs CSV. "
            "Each CSV row is treated as one source image and one target image."
        )
    )
    parser.add_argument("--nearest-pairs-csv", type=Path, default=DEFAULT_PAIRS_CSV)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--sample-limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--l2-epsilons",
        type=parse_l2_epsilons,
        default=parse_l2_epsilons("8.0"),
        help="Comma-separated L2 perturbation budgets in image [0, 1] space.",
    )
    parser.add_argument(
        "--pgd-step-size",
        type=parse_positive_floats,
        default=parse_positive_floats("0.25"),
        help="Comma-separated L2-normalized PGD step sizes.",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--check-interval", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=1800.0)
    parser.add_argument("--random-start", action="store_true", help="Start from a random point inside the L2 ball.")
    parser.add_argument("--save-examples", type=int, default=0, help="0 saves no PNGs, -1 saves every success.")
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle CSV rows before applying --sample-limit.")
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    return torch.device(value)


def resolve_csv_image(path_value: str, csv_path: Path) -> Path:
    raw_path = Path(path_value)
    if raw_path.is_absolute():
        return raw_path

    stripped_parts = list(raw_path.parts)
    while stripped_parts and stripped_parts[0] == "..":
        stripped_parts.pop(0)
    stripped_path = Path(*stripped_parts) if stripped_parts else raw_path

    candidates = [
        Path.cwd() / raw_path,
        CERTPHASH_ATTACK / raw_path,
        csv_path.resolve().parent / raw_path,
        ROOT / "CertPhash" / stripped_path,
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved

    return (CERTPHASH_ATTACK / raw_path).resolve()


def load_pairs(csv_path: Path, sample_limit: int, seed: int, shuffle: bool) -> list[dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if shuffle:
        random.Random(seed).shuffle(rows)
    if sample_limit > 0:
        rows = rows[:sample_limit]
    return rows


def load_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    model = resnet_v5(input_dim=64)
    try:
        weights = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        weights = torch.load(model_path, map_location=device)
    if isinstance(weights, dict):
        if "state_dict" in weights:
            weights = weights["state_dict"]
        elif "model" in weights and isinstance(weights["model"], dict):
            weights = weights["model"]
    weights = {key.removeprefix("module."): value for key, value in weights.items()}
    model.load_state_dict(weights)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def rounded_hash(model: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    return torch.relu(torch.round(model(normalize(image, "coco"))))


@torch.no_grad()
def hash_l1(model: torch.nn.Module, image: torch.Tensor, target_hash: torch.Tensor) -> float:
    current_hash = rounded_hash(model, image).int()
    return float(torch.nn.functional.l1_loss(current_hash.float(), target_hash.float(), reduction="sum").cpu())


def project_l2_ball(delta: torch.Tensor, epsilon: float) -> torch.Tensor:
    flat_delta = delta.flatten(start_dim=1)
    norms = torch.linalg.vector_norm(flat_delta, ord=2, dim=1).clamp_min(1e-12)
    scale = torch.clamp(epsilon / norms, max=1.0)
    return delta * scale.view(-1, 1, 1, 1)


def random_l2_delta(source: torch.Tensor, epsilon: float) -> torch.Tensor:
    delta = torch.randn_like(source)
    delta = project_l2_ball(delta, 1.0)
    radius = torch.rand(source.shape[0], device=source.device).view(-1, 1, 1, 1)
    delta = delta * radius * epsilon
    delta = (source + delta).clamp(0.0, 1.0) - source
    return project_l2_ball(delta, epsilon)


def should_save_example(saved_count: int, save_examples: int) -> bool:
    if save_examples < 0:
        return True
    return saved_count < save_examples


def attack_pair(
    model: torch.nn.Module,
    source_path: Path,
    target_path: Path,
    device: torch.device,
    args: argparse.Namespace,
    l2_epsilon: float,
    pgd_step_size: float,
) -> dict[str, object]:
    source = load_and_preprocess_img(str(source_path), device, "coco")
    target = load_and_preprocess_img(str(target_path), device, "coco")
    source_orig = source.clone()

    with torch.no_grad():
        target_hash = rounded_hash(model, target)
        initial_hash_l1 = hash_l1(model, source, target_hash)

    if args.random_start:
        delta = random_l2_delta(source, l2_epsilon).detach().requires_grad_(True)
    else:
        delta = torch.zeros_like(source, requires_grad=True)

    mse_loss = torch.nn.MSELoss()
    final_step = args.steps
    final_hash_l1 = initial_hash_l1
    success = False
    current_img = source_orig

    iterator = tqdm(range(args.steps), disable=args.disable_progress, leave=False)
    for step in iterator:
        current_img = source + delta
        outputs_source = model(normalize(current_img, "coco"))
        target_loss = mse_loss(outputs_source, target_hash)

        target_loss.backward()

        with torch.no_grad():
            grad = delta.grad
            grad_norm = torch.linalg.vector_norm(grad.flatten(start_dim=1), ord=2, dim=1).clamp_min(1e-12)
            normalized_grad = grad / grad_norm.view(-1, 1, 1, 1)

            delta.sub_(pgd_step_size * normalized_grad)
            delta.copy_(project_l2_ball(delta, l2_epsilon))
            delta.copy_((source + delta).clamp(0.0, 1.0) - source)
            delta.copy_(project_l2_ball(delta, l2_epsilon))
            delta.grad.zero_()

        if step % args.check_interval != 0:
            continue

        with torch.no_grad():
            current_img = source + delta
            final_hash_l1 = hash_l1(model, current_img, target_hash)
            if final_hash_l1 < args.threshold:
                final_step = step + 1
                success = True
                break

    with torch.no_grad():
        current_img = (source + delta).clamp(0.0, 1.0)
        diff = current_img - source_orig
        l2_distance = float(torch.norm(diff, p=2).cpu())
        linf_distance = float(torch.norm(diff, p=float("inf")).cpu())
        if not success:
            final_hash_l1 = hash_l1(model, current_img, target_hash)

    return {
        "source": str(source_path),
        "target": str(target_path),
        "success": success,
        "initial_hash_l1": initial_hash_l1,
        "final_hash_l1": final_hash_l1,
        "l2_budget": l2_epsilon,
        "pgd_step_size": pgd_step_size,
        "l2": l2_distance,
        "l_inf": linf_distance,
        "steps": final_step,
        "adversarial": current_img.detach(),
        "source_tensor": source_orig.detach(),
        "target_tensor": target.detach(),
        "delta": diff.detach(),
    }


def write_results(results_path: Path, rows: list[dict[str, object]]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "target",
        "nearest_distance",
        "tie_count",
        "l2_budget",
        "pgd_step_size",
        "success",
        "initial_hash_l1",
        "final_hash_l1",
        "l2",
        "l_inf",
        "steps",
        "adversarial_file",
    ]
    with open(results_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_float_label(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.l2_epsilons = sorted(set(args.l2_epsilons))
    args.pgd_step_size = sorted(set(args.pgd_step_size))

    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(args.nearest_pairs_csv, args.sample_limit, args.seed, args.shuffle)
    model = load_model(args.model_path, device)

    print(f"Device: {device}")
    print(f"Loaded {len(pairs)} nearest pairs from {args.nearest_pairs_csv}")
    print(f"Model: {args.model_path}")
    print(f"L2 PGD budgets: {', '.join(f'{epsilon:g}' for epsilon in args.l2_epsilons)}")
    print(f"PGD step sizes: {', '.join(f'{step:g}' for step in args.pgd_step_size)}")

    saved_examples = 0
    for pgd_step_size in args.pgd_step_size:
        for l2_epsilon in args.l2_epsilons:
            result_rows = []
            print(f"\n=== PGD step size {pgd_step_size:g}, L2 epsilon {l2_epsilon:g} ===")
            for pair_index, pair in enumerate(pairs, start=1):
                source_path = resolve_csv_image(pair["image1"], args.nearest_pairs_csv)
                target_path = resolve_csv_image(pair["image2"], args.nearest_pairs_csv)

                adversarial_file = ""
                result = attack_pair(model, source_path, target_path, device, args, l2_epsilon, pgd_step_size)

                if result["success"] and should_save_example(saved_examples, args.save_examples):
                    saved_examples += 1
                    budget_label = f"l2_{l2_epsilon:g}".replace(".", "p")
                    step_label = f"step_{pgd_step_size:g}".replace(".", "p")
                    example_prefix = (
                        f"{step_label}_{budget_label}_example_{saved_examples:03d}_"
                        f"{source_path.stem}_to_{target_path.stem}"
                    )
                    save_images(result["source_tensor"], str(args.output_dir), f"{example_prefix}_source")
                    save_images(result["target_tensor"], str(args.output_dir), f"{example_prefix}_target")
                    save_images(result["adversarial"], str(args.output_dir), f"{example_prefix}_adversarial")
                    save_images(result["delta"], str(args.output_dir), f"{example_prefix}_delta")
                    adversarial_file = str(args.output_dir / f"{example_prefix}_adversarial.png")

                row = {
                    "source": result["source"],
                    "target": result["target"],
                    "nearest_distance": pair.get("distance", ""),
                    "tie_count": pair.get("tie_count", ""),
                    "l2_budget": result["l2_budget"],
                    "pgd_step_size": result["pgd_step_size"],
                    "success": result["success"],
                    "initial_hash_l1": result["initial_hash_l1"],
                    "final_hash_l1": result["final_hash_l1"],
                    "l2": result["l2"],
                    "l_inf": result["l_inf"],
                    "steps": result["steps"],
                    "adversarial_file": adversarial_file,
                }
                result_rows.append(row)

                status = "success" if result["success"] else "fail"
                print(
                    f"[{pair_index}/{len(pairs)}] {status} "
                    f"pgd_step_size={result['pgd_step_size']:.4f} "
                    f"l2_budget={result['l2_budget']:.4f} "
                    f"final_hash_l1={result['final_hash_l1']:.1f} "
                    f"l2={result['l2']:.4f} l_inf={result['l_inf']:.4f} steps={result['steps']}"
                )

            step_label = format_float_label(pgd_step_size)
            budget_label = format_float_label(l2_epsilon)
            results_path = (
                args.output_dir / f"nearest_pair_l2_pgd_step_{step_label}_budget_{budget_label}_collision_results.csv"
            )
            write_results(results_path, result_rows)

            print(f"\nSaved results to {results_path}")
            budget_rows = result_rows
            successes = [row for row in budget_rows if row["success"]]
            success_rate = (len(successes) / len(budget_rows) * 100.0) if budget_rows else 0.0
            print(
                f"PGD step size {pgd_step_size:g}, L2 epsilon {l2_epsilon:g} "
                f"Collision Rate: {success_rate:.2f}% ({len(successes)}/{len(budget_rows)})"
            )
            if successes:
                print(f"  Mean l2: {np.mean([float(row['l2']) for row in successes]):.4f}")
                print(f"  Mean l_inf: {np.mean([float(row['l_inf']) for row in successes]):.4f}")
                print(f"  Mean steps: {np.mean([float(row['steps']) for row in successes]):.2f}")


if __name__ == "__main__":
    main()
