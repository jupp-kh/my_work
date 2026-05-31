from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def resolve_image_path(images_dir: Path, image_name: str) -> Path:
    image_name = image_name.strip()
    image_path = Path(image_name)
    if image_path.is_absolute() and image_path.exists():
        return image_path

    candidate = images_dir / image_name
    if candidate.exists():
        return candidate

    parts = list(image_path.parts)
    if "coco100x100" in parts:
        idx = parts.index("coco100x100")
        tail = Path(*parts[idx + 1 :])
        candidate = images_dir / tail
        if candidate.exists():
            return candidate

    if image_path.exists():
        return image_path

    basename_candidate = images_dir / image_path.name
    if basename_candidate.exists():
        return basename_candidate

    return candidate


def load_image(images_dir: Path, image_name: str) -> Image.Image:
    image_path = resolve_image_path(images_dir, image_name)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path)


def compute_mean_distance(df: pd.DataFrame) -> float:
    if "distance" not in df.columns:
        raise KeyError("Missing 'distance' column in CSV")
    return float(df["distance"].mean())


def top_min_distances(df: pd.DataFrame, count: int = 10) -> pd.DataFrame:
    if "distance" not in df.columns:
        raise KeyError("Missing 'distance' column in CSV")
    return df.nsmallest(count, "distance").reset_index(drop=True)


def unique_top_min_distances(df: pd.DataFrame, count: int) -> pd.DataFrame:
    if df.empty:
        return df
    if not {"image1", "image2"}.issubset(df.columns):
        raise KeyError("Missing 'image1' or 'image2' column in CSV")
    rows = df.sort_values("distance", ascending=True).reset_index(drop=True)
    seen: set[str] = set()
    kept_rows = []
    for _, row in rows.iterrows():
        img1 = str(row["image1"])
        img2 = str(row["image2"])
        if img1 in seen or img2 in seen:
            continue
        kept_rows.append(row)
        seen.add(img1)
        seen.add(img2)
        if len(kept_rows) >= count:
            break
    if not kept_rows:
        return rows.head(0)
    return pd.DataFrame(kept_rows).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a nearest image pair.")
    parser.add_argument(
        "--nearest",
        type=Path,
        default=Path(
            r"C:\Users\youss\Desktop\Uni\Master\S3\lab-visual-computing\my_work\results\cocoanalysis\coco100x100_hashes_coco_photodna_ep8_nearest_pairs.csv"
        ),
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path(
            r"C:\Users\youss\Desktop\Uni\Master\S3\lab-visual-computing\CertPhash\train_verify\data\coco100x100"
        ),
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--random-count",
        type=int,
        default=0,
        help="Show this many random pairs (overrides --index when > 0).",
    )
    parser.add_argument(
        "--top-min",
        type=int,
        default=0,
        help="Visualize N smallest distances (0 to disable).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.nearest.exists():
        raise FileNotFoundError(f"Nearest-pairs CSV not found: {args.nearest}")

    df = pd.read_csv(args.nearest)
    if df.empty:
        raise ValueError("Nearest-pairs CSV is empty")

    zero_count = int((df["distance"] == 0).sum())
    df_nonzero = df[df["distance"] > 1500].reset_index(drop=True)
    if df_nonzero.empty:
        raise ValueError("All distances are 0; nothing to visualize.")

    mean_distance = compute_mean_distance(df_nonzero)

    print(f"Mean distance: {mean_distance:.2f}")
    print(f"min distance: {df_nonzero['distance'].min():.2f}")
    print(f"max distance: {df_nonzero['distance'].max():.2f}")
    print(f"number of 0 distances: {zero_count}")

    if args.top_min > 0:
        rows = unique_top_min_distances(df_nonzero, count=args.top_min)
        print(f"Top {len(rows)} minimum distances:")
        print(rows[["image1", "image2", "distance"]].to_string(index=False))
    elif args.random_count > 0:
        sample_count = min(args.random_count, len(df_nonzero))
        rows = df_nonzero.sample(n=sample_count)
    else:
        if args.index < 0 or args.index >= len(df_nonzero):
            raise IndexError(f"index {args.index} is out of range")
        rows = df_nonzero.iloc[[args.index]]

    fig, axes = plt.subplots(
        len(rows),
        3,
        figsize=(7, 3 * len(rows)),
        gridspec_kw={"width_ratios": [1, 0.3, 1]},
    )
    if len(rows) == 1:
        axes = [axes]

    for row, row_axes in zip(rows.itertuples(index=False), axes):
        row_data = row._asdict()
        left_ax, mid_ax, right_ax = row_axes
        for ax, key in [(left_ax, "image1"), (right_ax, "image2")]:
            try:
                img = load_image(args.images_dir, row_data[key])
                ax.imshow(img)
            except FileNotFoundError as exc:
                ax.text(0.5, 0.5, str(exc), ha="center", va="center", wrap=True)
            ax.axis("off")

        mid_ax.text(
            0.5,
            0.5,
            f"{int(round(row_data['distance']))}",
            ha="center",
            va="center",
            fontsize=12,
        )
        mid_ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
