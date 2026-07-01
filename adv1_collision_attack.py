import argparse
import concurrent.futures
import csv
import random
import shutil
import sys
import threading
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
ATTACK_ROOT = REPO_ROOT / "CertPhash" / "attack"
if str(ATTACK_ROOT) not in sys.path:
    sys.path.insert(0, str(ATTACK_ROOT))

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from tqdm import tqdm

from models.resnet_v5 import resnet_v5
from utils.image_processing import load_and_preprocess_img, normalize, save_images
from utils.logger import Logger


DEFAULT_COCO_ROOT = REPO_ROOT / "CertPhash" / "train_verify" / "data" / "coco100x100_val"
DEFAULT_MODEL_PATH = (
    REPO_ROOT
    / "CertPhash"
    / "train_verify"
    / "saved_models"
    / "coco_photodna_ep8"
    / "ckpt_best.pth"
)
DEFAULT_NEAREST_PAIRS = (
    REPO_ROOT
    / "my_work"
    / "results"
    / "cocoanalysis"
    / "coco100x100_val_hashes_coco_photodna_ep8_nearest_pairs.csv"
)


def resolve_image_path(path_value, dataset_root):
    path = Path(str(path_value).replace("\\", "/"))
    if path.is_absolute():
        return path.resolve()

    candidates = [
        (Path.cwd() / path),
        (REPO_ROOT / path),
        (ATTACK_ROOT / path),
        (dataset_root / path.name),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (dataset_root / path.name).resolve()


def image_keys(path):
    resolved = Path(path).resolve()
    return {
        str(resolved),
        resolved.as_posix(),
        resolved.name,
    }


def load_nearest_pairs(csv_path, dataset_root):
    pairs = {}
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            source_path = resolve_image_path(row["image1"], dataset_root)
            target_path = resolve_image_path(row["image2"], dataset_root)
            for key in image_keys(source_path):
                pairs[key] = str(target_path)
    return pairs


def get_precomputed_target_path(img, nearest_pairs):
    for key in image_keys(img):
        if key in nearest_pairs:
            return nearest_pairs[key]
    raise RuntimeError(f"No precomputed nearest-neighbor target found for {img}")


def project_delta(delta, epsilon, projection):
    if projection == "linf":
        delta.clamp_(-epsilon, epsilon)
    elif projection == "l2":
        norm_delta = torch.norm(delta, p=2)
        if norm_delta > epsilon:
            delta.mul_(epsilon / norm_delta)
    else:
        raise RuntimeError(f"{projection} is no valid projection. Use linf or l2.")


class ImageSaveLimiter:
    def __init__(self, limit):
        self.limit = limit
        self.saved = 0
        self.lock = threading.Lock()

    def reserve(self):
        if self.limit == 0:
            return None
        with self.lock:
            if self.limit > 0 and self.saved >= self.limit:
                return None
            self.saved += 1
            return self.saved


def optimization_thread(
    url_list,
    device,
    logger,
    args,
    model_path,
    epsilon,
    nearest_pairs,
    image_save_limiter,
):
    print(f"Process {threading.get_ident()} started")
    model = resnet_v5(input_dim=64)
    theshold = 1800
    model_weights = torch.load(model_path, map_location=device)
    model.load_state_dict(model_weights)
    model.eval()
    model.to(device)

    # Start optimizing images
    while url_list != []:
        print(len(url_list))
        img = url_list.pop(0)  # Pop the first image from the original list
        target_path = get_precomputed_target_path(img, nearest_pairs)
        target_tensor = load_and_preprocess_img(target_path, device, "coco")
        with torch.no_grad():
            target_hash = torch.relu(torch.round(model(normalize(target_tensor, "coco")))).float()

        # Store and reload source image to avoid image changes due to different formats
        source = load_and_preprocess_img(img, device, "coco")
        example_index = image_save_limiter.reserve() if args.output_folder != "" else None
        example_dir = None
        if example_index is not None:
            example_dir = Path(args.output_folder) / f"example_{example_index:06d}"
            example_dir.mkdir(parents=True, exist_ok=True)
            save_images(source, str(example_dir), "source")
            save_images(target_tensor, str(example_dir), "target")
        source_orig = source.clone()
        delta = torch.zeros_like(source, requires_grad=True)

        with torch.no_grad():
            l1_loss = torch.nn.L1Loss(reduction="sum")
            l2_loss = torch.nn.MSELoss()
            source_hash_initial = torch.relu(torch.round(model(normalize(source, "coco")))).int()
            initial_hash_l1 = l1_loss(source_hash_initial, target_hash).item()

            if args.edges_only:
                from skimage import feature

                # Compute edge mask
                transform = T.Compose([T.ToPILImage(), T.Grayscale(), T.ToTensor()])
                image_gray = transform(source.squeeze()).squeeze()
                image_gray = image_gray.cpu().numpy()
                edges = feature.canny(image_gray, sigma=3).astype(int)
                edge_mask = torch.from_numpy(edges).to(device)

        # Apply attack
        if args.optimizer == "Adam":
            optimizer = torch.optim.Adam(params=[delta], lr=args.learning_rate)
        elif args.optimizer == "SGD":
            optimizer = torch.optim.SGD(params=[delta], lr=args.learning_rate)
        else:
            raise RuntimeError(
                f"{args.optimizer} is no valid optimizer class. Please select --optimizer out of [Adam, SGD]"
            )

        for i in tqdm(range(1000)):
            outputs_source = model(normalize(source + delta, "coco"))
            target_loss = l2_loss(outputs_source, target_hash)
            total_loss = target_loss

            optimizer.zero_grad()
            total_loss.backward()
            if args.edges_only:
                optimizer.param_groups[0]["params"][0].grad *= edge_mask

            optimizer.step()

            with torch.no_grad():
                project_delta(delta, epsilon, args.projection)

                # Check for hash changes
                if i % args.check_interval == 0:
                    with torch.no_grad():
                        current_img = source + delta
                        check_output = model(normalize(current_img, "coco"))
                        source_hash_hex = torch.relu(torch.round(check_output)).int()
                        final_hash_l1 = l1_loss(source_hash_hex, target_hash).item()
                        if final_hash_l1 < theshold:
                            # Compute metrics in the [0, 1] space
                            l2_distance = torch.norm(current_img - source_orig, p=2)
                            linf_distance = torch.norm(
                                current_img - source_orig, p=float("inf")
                            )
                            print(
                                f"Finishing after {i + 1} steps - L2 distance: {l2_distance:.4f} - L-Inf distance: {linf_distance:.4f}"
                            )

                            if example_dir is not None:
                                save_images(source + delta, str(example_dir), "adversarial")
                                save_images(delta, str(example_dir), "delta")
                            logger_data = [
                                img,
                                target_path,
                                initial_hash_l1,
                                True,
                                final_hash_l1,
                                l2_distance.item(),
                                linf_distance.item(),
                                i + 1,
                            ]

                            logger.add_line(logger_data)
                            break
        else:
            with torch.no_grad():
                current_img = source + delta
                check_output = model(normalize(current_img, "coco"))
                source_hash_hex = torch.relu(torch.round(check_output)).int()
                final_hash_l1 = l1_loss(source_hash_hex, target_hash).item()
                l2_distance = torch.norm(current_img - source_orig, p=2)
                linf_distance = torch.norm(current_img - source_orig, p=float("inf"))
                success = final_hash_l1 < theshold
            if example_dir is not None:
                save_images(source + delta, str(example_dir), "adversarial")
                save_images(delta, str(example_dir), "delta")
            logger.add_line(
                [
                    img,
                    target_path,
                    initial_hash_l1,
                    success,
                    final_hash_l1,
                    l2_distance.item(),
                    linf_distance.item(),
                    1000,
                ]
            )


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Perform neural collision attack.")
    parser.add_argument(
        "--source",
        dest="source",
        type=str,
        default=str(DEFAULT_COCO_ROOT),
        help="image or directory to manipulate",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="path of model weight",
    )
    parser.add_argument(
        "--nearest_pairs",
        type=str,
        default=str(DEFAULT_NEAREST_PAIRS),
        help="precomputed nearest-neighbor csv file path for coco",
    )
    parser.add_argument(
        "--learning_rate",
        dest="learning_rate",
        default=1e-3,
        type=float,
        help="step size of PGD optimization step",
    )
    parser.add_argument("--optimizer", dest="optimizer", default="Adam", type=str, help="kind of optimizer")
    parser.add_argument(
        "--ssim_weight",
        dest="ssim_weight",
        default=10,
        type=float,
        help="weight of ssim loss",
    )
    parser.add_argument(
        "--experiment_name",
        dest="experiment_name",
        default="preimage_attack",
        type=str,
        help="name of the experiment and logging file",
    )
    parser.add_argument(
        "--output_folder",
        dest="output_folder",
        default=str(REPO_ROOT / "my_work" / "results" / "adv1_collision_attack_outputs_robust"),
        type=str,
        help="folder to save optimized images in",
    )
    parser.add_argument(
        "--save_examples",
        default=0,
        type=int,
        help="Number of attempted examples to save as images. Use 0 to disable image saving, -1 to save all.",
    )
    parser.add_argument("--edges_only", dest="edges_only", action="store_true", help="Change only pixels of edges")
    parser.add_argument(
        "--sample_limit",
        dest="sample_limit",
        default=10000,
        type=int,
        help="Maximum of images to be processed",
    )
    parser.add_argument(
        "--threads",
        dest="num_threads",
        default=1,
        type=int,
        help="Number of parallel threads",
    )
    parser.add_argument(
        "--check_interval",
        dest="check_interval",
        default=10,
        type=int,
        help="Hash change interval checking",
    )
    parser.add_argument("--epsilon", default=16.0 / 255, type=float)
    parser.add_argument(
        "--projection",
        choices=["linf", "l2"],
        default="linf",
        help="PGD projection norm: linf clips each pixel delta, l2 rescales the full perturbation.",
    )

    args = parser.parse_args()

    model_path = str(Path(args.model_path).expanduser().resolve())
    # Load and prepare components
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dataset_root = Path(args.source).expanduser()
    if dataset_root.is_file():
        dataset_root = dataset_root.parent
    dataset_root = dataset_root.resolve()
    nearest_pairs = load_nearest_pairs(Path(args.nearest_pairs).expanduser().resolve(), dataset_root)

    if args.output_folder != "":
        output_folder = Path(args.output_folder).expanduser()
        try:
            output_folder.mkdir(parents=True)
        except FileExistsError:
            if not any(output_folder.iterdir()):
                print(f"Folder {output_folder} already exists and is empty.")
            else:
                print(f"Folder {output_folder} already exists and is not empty, delete it")
                shutil.rmtree(output_folder)
                output_folder.mkdir(parents=True)
        args.output_folder = str(output_folder)

    # Prepare logging
    logging_header = [
        "source",
        "target",
        "initial_hash_l1",
        "success",
        "final_hash_l1",
        "l2",
        "l_inf",
        "steps",
    ]
    logger = Logger(args.experiment_name, logging_header, output_dir=str(Path(args.output_folder) / "logs"))

    # Load images
    source = Path(args.source).expanduser()
    if source.is_file():
        images = [str(source.resolve())]
    elif source.is_dir():
        images = sorted(str(path.resolve()) for path in source.iterdir() if path.is_file())
    else:
        raise RuntimeError(f"{args.source} is neither a file nor a directory.")

    if len(images) > args.sample_limit:
        images = random.sample(images, args.sample_limit)

    threads_args = (
        images,
        device,
        logger,
        args,
        model_path,
        args.epsilon,
        nearest_pairs,
        ImageSaveLimiter(args.save_examples),
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        for t in range(args.num_threads):
            executor.submit(lambda p: optimization_thread(*p), threads_args)

    logger.finish_logging()

    """
    Statistic
    """
    log_file = Path(args.output_folder) / "logs" / f"{args.experiment_name}.csv"
    df = pd.read_csv(log_file, skiprows=1)
    if not df.empty:
        df["l2"] = df["l2"].astype(float)
        df["l_inf"] = df["l_inf"].astype(float)
        df["steps"] = df["steps"].astype(int)
        df["success"] = df["success"].astype(str).str.lower() == "true"
        print(f'l2: {df["l2"].mean()}')
        print(f'l_inf: {df["l_inf"].mean()}')
        print(f'steps: {df["steps"].mean()}')
        projection_metric = "l_inf" if args.projection == "linf" else "l2"
        epsilon = "{:.4f}".format(max(df[projection_metric]))
        print(
            f"===>Collision Rate under {projection_metric}={epsilon}: {(df['success'].mean() * 100)}%"
        )
    else:
        print("No successful collisions were logged.")


if __name__ == "__main__":
    Path("./temp").mkdir(exist_ok=True)
    main()
