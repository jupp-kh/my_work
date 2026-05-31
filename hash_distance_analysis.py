from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def parse_hash_bin(value: str) -> np.ndarray:
    value = value.strip()
    if set(value).issubset({"0", "1"}):
        return np.fromiter((1 if ch == "1" else 0 for ch in value), dtype=np.float32)

    cleaned = value.replace("[", " ").replace("]", " ").replace(",", " ")
    parts = cleaned.split()
    if not parts:
        raise ValueError("hash_bin is empty after parsing")
    try:
        numbers = [float(item) for item in parts]
    except ValueError as exc:
        raise ValueError(f"Invalid hash_bin: {value[:64]}...") from exc
    return np.array(numbers, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="nearest-pair search (L1).")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(
            r"C:\Users\youss\Desktop\Uni\Master\S3\lab-visual-computing\CertPhash\dataset_hashes\coco100x100_hashes_coco_photodna_ep8.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            r"C:\Users\youss\Desktop\Uni\Master\S3\lab-visual-computing\my_work\results\cocoanalysis"
        ),
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--show-hist", action="store_true", default=True)
    parser.add_argument("--no-show-hist", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.csv)
    required_cols = {"image", "hash_bin", "hash_hex"}
    if required_cols.issubset(df.columns):
        image_col, hash_bin_col = "image", "hash_bin"
    else:
        columns = list(df.columns)
        if len(columns) < 3:
            raise ValueError("CSV must have at least 3 columns: image, hash_bin, hash_hex")
        image_col = columns[1] if columns[0].lower().startswith("unnamed") else columns[0]
        hash_bin_col = columns[2]

    df = df.dropna(subset=[image_col, hash_bin_col]).reset_index(drop=True)
    df[hash_bin_col] = df[hash_bin_col].astype(str)

    if args.max_rows is not None:
        df = df.head(args.max_rows).reset_index(drop=True)

    hash_arrays = df[hash_bin_col].map(parse_hash_bin).to_list()
    lengths = {arr.size for arr in hash_arrays}
    if len(lengths) != 1:
        raise ValueError(f"hash_bin lengths differ: {sorted(lengths)}")

    hashes = np.stack(hash_arrays, axis=0)
    images: List[str] = df[image_col].tolist()

    device = torch.device(args.device)
    data = torch.tensor(hashes, dtype=torch.float32, device=device)
    n = data.shape[0]

    best_dist = torch.full((n,), float("inf"), device=device)
    best_idx = torch.full((n,), -1, dtype=torch.long, device=device)
    best_tie_count = torch.zeros((n,), dtype=torch.long, device=device)

    batch_size = args.batch_size
    block_size = args.block_size

    for i_start in range(0, n, batch_size):
        i_end = min(n, i_start + batch_size)
        left = data[i_start:i_end]

        for j_start in range(0, n, block_size):
            j_end = min(n, j_start + block_size)
            right = data[j_start:j_end]

            # L1 distances for the batch: [B, J]
            dists = torch.cdist(left, right, p=1)

            # Mask self-distances when blocks overlap
            if j_start <= i_end and j_end >= i_start:
                for local_i, global_i in enumerate(range(i_start, i_end)):
                    if j_start <= global_i < j_end:
                        dists[local_i, global_i - j_start] = float("inf")

            min_vals, min_idx = torch.min(dists, dim=1)
            global_idx = min_idx + j_start
            min_counts = (dists == min_vals.unsqueeze(1)).sum(dim=1)

            current_best = best_dist[i_start:i_end]
            better_mask = min_vals < current_best
            equal_mask = torch.isclose(min_vals, current_best, atol=1e-6)

            best_dist[i_start:i_end] = torch.where(better_mask, min_vals, current_best)
            best_idx[i_start:i_end] = torch.where(better_mask, global_idx, best_idx[i_start:i_end])
            best_tie_count[i_start:i_end] = torch.where(
                better_mask,
                min_counts,
                torch.where(equal_mask, best_tie_count[i_start:i_end] + min_counts, best_tie_count[i_start:i_end]),
            )

        if args.progress_every > 0 and ((i_end // batch_size) % args.progress_every == 0 or i_end == n):
            print(f"Processed {i_end} / {n} rows")

    best_dist_cpu = best_dist.cpu().numpy()
    best_idx_cpu = best_idx.cpu().numpy()
    best_tie_count_cpu = best_tie_count.cpu().numpy()

    nearest_rows: List[Tuple[str, str, float, int]] = []
    for i, j in enumerate(best_idx_cpu):
        if j < 0:
            continue
        nearest_rows.append(
            (images[i], images[int(j)], float(best_dist_cpu[i]), int(best_tie_count_cpu[i]))
        )

    nearest_df = pd.DataFrame(nearest_rows, columns=["image1", "image2", "distance", "tie_count"])
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.csv.stem
    nearest_path = output_dir / f"{prefix}_nearest_pairs.csv"
    nearest_df.to_csv(nearest_path, index=False)
    print(f"Saved nearest pairs to {nearest_path}")

    if not nearest_df.empty:
        closest_row = nearest_df.sort_values("distance", ascending=True).head(1)
        closest_path = output_dir / f"{prefix}_closest_pair.csv"
        closest_row.to_csv(closest_path, index=False)
        print("Closest pair:")
        print(closest_row)
        mean_closest = float(nearest_df["distance"].mean())
        print(f"Mean closest distance for {len(nearest_df)}: {mean_closest:.6f}")

    show_hist = args.show_hist
    if args.no_show_hist:
        show_hist = False

    if show_hist and not nearest_df.empty:
        plt.figure(figsize=(8, 4))
        plt.hist(nearest_df["distance"], bins=30, color="#2a9d8f", edgecolor="#1f1f1f")
        plt.title("Histogram of Nearest-Neighbor L1 Distances")
        plt.xlabel("Distance")
        plt.ylabel("Count")
        plt.tight_layout()
        hist_path = output_dir / f"{prefix}_nearest_hist.png"
        plt.savefig(hist_path, dpi=150)
        print(f"Saved histogram to {hist_path}")
        plt.show()

        if "tie_count" in nearest_df.columns:
            plt.figure(figsize=(8, 4))
            plt.hist(nearest_df["tie_count"], bins=30, color="#264653", edgecolor="#1f1f1f")
            plt.title("Histogram of Tie Counts")
            plt.xlabel("Tie Count")
            plt.ylabel("Count")
            plt.tight_layout()
            tie_hist_path = output_dir / f"{prefix}_tie_count_hist.png"
            plt.savefig(tie_hist_path, dpi=150)
            print(f"Saved tie-count histogram to {tie_hist_path}")
            plt.show()


if __name__ == "__main__":
    main()
