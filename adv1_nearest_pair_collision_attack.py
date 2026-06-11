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
DEFAULT_OUTPUT_DIR = ROOT / "my_work" / "results" / "adv1_nearest_pair_collision"

sys.path.insert(0, str(CERTPHASH_ATTACK))
from models.resnet_v5 import resnet_v5  # noqa: E402
from utils.image_processing import load_and_preprocess_img, normalize, save_images  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the adv1 CertPHash collision attack using a precomputed nearest-pairs CSV. "
            "Each CSV row is treated as one source image and one target image."
        )
    )
    parser.add_argument("--nearest-pairs-csv", type=Path, default=DEFAULT_PAIRS_CSV)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--sample-limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epsilon", type=float, default=16.0 / 255.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--check-interval", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=1800.0)
    parser.add_argument("--save-examples", type=int, default=2, help="0 saves no PNGs, -1 saves every success.")
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle CSV rows before applying --sample-limit.")
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
) -> dict[str, object]:
    source = load_and_preprocess_img(str(source_path), device, "coco")
    target = load_and_preprocess_img(str(target_path), device, "coco")
    source_orig = source.clone()

    with torch.no_grad():
        target_hash = rounded_hash(model, target)
        initial_hash_l1 = hash_l1(model, source, target_hash)

    delta = torch.zeros_like(source, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=args.learning_rate)
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

        optimizer.zero_grad()
        target_loss.backward()
        optimizer.step()

        with torch.no_grad():
            delta.clamp_(-args.epsilon, args.epsilon)

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


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(args.nearest_pairs_csv, args.sample_limit, args.seed, args.shuffle)
    model = load_model(args.model_path, device)

    print(f"Device: {device}")
    print(f"Loaded {len(pairs)} nearest pairs from {args.nearest_pairs_csv}")
    print(f"Model: {args.model_path}")

    saved_examples = 0
    result_rows = []
    for pair_index, pair in enumerate(pairs, start=1):
        source_path = resolve_csv_image(pair["image1"], args.nearest_pairs_csv)
        target_path = resolve_csv_image(pair["image2"], args.nearest_pairs_csv)
        result = attack_pair(model, source_path, target_path, device, args)

        adversarial_file = ""
        if result["success"] and should_save_example(saved_examples, args.save_examples):
            saved_examples += 1
            example_prefix = f"example_{saved_examples:03d}_{source_path.stem}_to_{target_path.stem}"
            save_images(result["source_tensor"], str(args.output_dir), f"{example_prefix}_source")
            save_images(result["target_tensor"], str(args.output_dir), f"{example_prefix}_target")
            save_images(result["adversarial"], str(args.output_dir), f"{example_prefix}_adversarial")
            save_images(result["delta"], str(args.output_dir), f"{example_prefix}_delta")
            adversarial_file = str(args.output_dir / f"{example_prefix}_adversarial.png")

        result_rows.append(
            {
                "source": result["source"],
                "target": result["target"],
                "nearest_distance": pair.get("distance", ""),
                "tie_count": pair.get("tie_count", ""),
                "success": result["success"],
                "initial_hash_l1": result["initial_hash_l1"],
                "final_hash_l1": result["final_hash_l1"],
                "l2": result["l2"],
                "l_inf": result["l_inf"],
                "steps": result["steps"],
                "adversarial_file": adversarial_file,
            }
        )

        status = "success" if result["success"] else "fail"
        print(
            f"[{pair_index}/{len(pairs)}] {status} "
            f"final_hash_l1={result['final_hash_l1']:.1f} "
            f"l2={result['l2']:.4f} l_inf={result['l_inf']:.4f} steps={result['steps']}"
        )

    results_path = args.output_dir / "nearest_pair_collision_results.csv"
    write_results(results_path, result_rows)

    successes = [row for row in result_rows if row["success"]]
    success_rate = (len(successes) / len(result_rows) * 100.0) if result_rows else 0.0
    print(f"Saved results to {results_path}")
    print(f"Collision Rate: {success_rate:.2f}% ({len(successes)}/{len(result_rows)})")
    if successes:
        print(f"Mean l2: {np.mean([float(row['l2']) for row in successes]):.4f}")
        print(f"Mean l_inf: {np.mean([float(row['l_inf']) for row in successes]):.4f}")
        print(f"Mean steps: {np.mean([float(row['steps']) for row in successes]):.2f}")


if __name__ == "__main__":
    main()
