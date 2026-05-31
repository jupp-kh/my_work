import argparse
import base64
import concurrent.futures
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ORIGINAL_MAP = None


def parse_hash_bin(hash_bin_str: str) -> np.ndarray:
    """Parse a hash_bin string like "[ 1  2  3 ...]" into a numpy array."""
    if pd.isna(hash_bin_str):
        raise ValueError("hash_bin is NaN")
    values = re.findall(r"-?\d+", str(hash_bin_str))
    if not values:
        raise ValueError(f"No integers found in hash_bin: {hash_bin_str!r}")
    return np.array([int(v) for v in values], dtype=np.int64)


def parse_hash_binary(binary_str: str) -> np.ndarray:
    """Parse a binary string like "010101" or "0 1 0 1" into a numpy array."""
    if pd.isna(binary_str):
        raise ValueError("hash_binary is NaN")
    text = re.sub(r"\s+", "", str(binary_str))
    if not re.fullmatch(r"[01]+", text):
        raise ValueError(f"Invalid binary string: {binary_str!r}")
    return np.fromiter((1 if ch == "1" else 0 for ch in text), dtype=np.int64)


def parse_hash_hex_or_base64(hash_str: str) -> np.ndarray:
    """Parse a hex or base64 string into a uint8 array."""
    if pd.isna(hash_str):
        raise ValueError("hash_hex is NaN")
    text = str(hash_str).strip()
    hex_like = re.fullmatch(r"[0-9a-fA-F]+", text) and len(text) % 2 == 0
    if hex_like:
        raw = bytes.fromhex(text)
    else:
        try:
            raw = base64.b64decode(text, validate=True)
        except Exception as exc:
            raise ValueError(f"Invalid hex/base64 string: {hash_str!r}") from exc
    return np.frombuffer(raw, dtype=np.uint8)


def l1_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Compute L1 distance between two hash vectors."""
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    return int(np.abs(a - b).sum())


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Compute Hamming distance between two vectors."""
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    if a.dtype == np.uint8 and b.dtype == np.uint8:
        xor = np.bitwise_xor(a, b)
        bit_counts = np.unpackbits(xor).sum()
        return int(bit_counts)

    a_int = a.astype(np.int64, copy=False)
    b_int = b.astype(np.int64, copy=False)
    if a_int.min() >= 0 and a_int.max() <= 1 and b_int.min() >= 0 and b_int.max() <= 1:
        return int(np.bitwise_xor(a_int, b_int).sum())

    return int((a_int != b_int).sum())


def parse_hash_value(row: pd.Series, hash_field: str) -> np.ndarray:
    """Parse a hash value based on the selected field."""
    if hash_field == "hash_bin":
        return parse_hash_bin(row["hash_bin"])
    if hash_field == "hash_binary":
        return parse_hash_binary(row["hash_binary"])
    if hash_field == "hash_hex":
        return parse_hash_hex_or_base64(row["hash_hex"])
    raise ValueError(f"Unsupported hash field: {hash_field}")


def read_hash_csv(csv_path: Path) -> pd.DataFrame:
    """Read a hash CSV with multiline hash values."""
    df = pd.read_csv(csv_path, engine="python")
    if "index" in df.columns:
        idx_col = "index"
    elif "Unnamed: 0" in df.columns:
        idx_col = "Unnamed: 0"
    else:
        idx_col = df.columns[0]
    df = df.rename(columns={idx_col: "index"})
    return df


def build_original_map(original_csv: Path, hash_field: str) -> dict:
    """Build a map from index to hash vector for originals."""
    df = read_hash_csv(original_csv)
    original_map = {}
    for _, row in df.iterrows():
        idx = int(row["index"])
        original_map[idx] = parse_hash_value(row, hash_field)
    return original_map


def discover_transformation_csvs(logs_dir: Path, original_name: str) -> list:
    """Find all transformation CSVs under logs_dir (excluding original)."""
    csvs = [p for p in logs_dir.rglob("*.csv") if p.name != original_name]
    return sorted(csvs)


def parse_transformation_info(csv_path: Path) -> tuple[str, str]:
    """Parse transformation name and value from filename."""
    name = csv_path.stem  # e.g. coco_val_brightness_1.0
    parts = name.split("_")
    if len(parts) >= 4:
        transformation = parts[2]
        value = "_".join(parts[3:])
    else:
        transformation = csv_path.parent.name
        value = ""
    return transformation, value


def compute_distances(original_map: dict, csv_path: Path, hash_field: str, metric: str) -> pd.DataFrame:
    """Compute distances for one transformation CSV."""
    df = read_hash_csv(csv_path)
    if hash_field not in df.columns:
        raise ValueError(f"Missing {hash_field} column in {csv_path}")
    distances = []
    missing = 0
    for _, row in df.iterrows():
        idx = int(row["index"])
        if idx not in original_map:
            missing += 1
            continue
        transformed = parse_hash_value(row, hash_field)
        if metric == "l1":
            dist = l1_distance(original_map[idx], transformed)
        elif metric == "hamming":
            dist = hamming_distance(original_map[idx], transformed)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        distances.append({"index": idx, "distance": dist})
    if missing:
        print(f"Warning: {missing} indices missing in originals for {csv_path.name}")
    return pd.DataFrame(distances)


def summarize_distances(dist_df: pd.DataFrame) -> dict:
    """Summarize distance statistics for a transformation."""
    if dist_df.empty:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": int(dist_df["distance"].count()),
        "mean": float(dist_df["distance"].mean()),
        "median": float(dist_df["distance"].median()),
        "min": int(dist_df["distance"].min()),
        "max": int(dist_df["distance"].max()),
    }


def summarize_by_transformation(summary_df: pd.DataFrame, mean_threshold: float) -> pd.DataFrame:
    """Compute per-transformation mean stats after filtering by mean threshold."""
    filtered = summary_df[summary_df["mean"] >= mean_threshold].copy()
    if filtered.empty:
        return pd.DataFrame(columns=["transformation", "count", "mean", "median", "min", "max"])
    grouped = filtered.groupby("transformation", as_index=False)
    return grouped.agg(
        count=("mean", "count"),
        mean=("mean", "mean"),
        median=("median", "median"),
        min=("min", "min"),
        max=("max", "max"),
    )


def init_worker(original_map: dict) -> None:
    """Initializer for worker processes."""
    global ORIGINAL_MAP
    ORIGINAL_MAP = original_map


def process_transformation(csv_path: Path, logs_dir: Path, hash_field: str, metric: str) -> dict:
    """Compute distances and return summary for one CSV."""
    transformation, value = parse_transformation_info(csv_path)
    dist_df = compute_distances(ORIGINAL_MAP, csv_path, hash_field, metric)

    summary = summarize_distances(dist_df)
    summary.update(
        {
            "transformation": transformation,
            "value": value,
            "csv": str(csv_path.relative_to(logs_dir)),
            "distances": dist_df["distance"].values,
        }
    )
    return summary


def log_tensorboard(summary_rows: list, output_dir: Path, log_dir: Path) -> None:
    """Write TensorBoard scalars and histograms for each transformation."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:
        raise RuntimeError(
            "TensorBoard logging requires torch and tensorboard. "
            "Install with: pip install torch tensorboard"
        ) from exc

    writer = SummaryWriter(log_dir=str(log_dir))
    try:
        grouped = {}
        for row in summary_rows:
            grouped.setdefault(row["transformation"], []).append(row)

        for transformation, rows in grouped.items():
            rows = sorted(rows, key=lambda r: _value_sort_key(r.get("value", "")))
            for step_idx, row in enumerate(rows):
                step = step_idx
                value_label = row.get("value", "")
                tag_base = f"{transformation}"

                if row["count"]:
                    writer.add_scalar(f"{tag_base}/mean", row["mean"], step)
                    writer.add_scalar(f"{tag_base}/median", row["median"], step)
                    writer.add_scalar(f"{tag_base}/min", row["min"], step)
                    writer.add_scalar(f"{tag_base}/max", row["max"], step)

                    distances = row.get("distances")
                    if distances is not None and len(distances):
                        writer.add_histogram(f"{tag_base}/distance", distances, step)

                writer.add_text(f"{tag_base}/value_label", f"step={step} value={value_label}", step)

            table_lines = ["| value | count | mean | median | min | max |", "| --- | --- | --- | --- | --- | --- |"]
            for row in rows:
                table_lines.append(
                    f"| {row.get('value', '')} | {row.get('count', 0)} | {row.get('mean', '')} | "
                    f"{row.get('median', '')} | {row.get('min', '')} | {row.get('max', '')} |"
                )
            writer.add_text(f"{transformation}/summary", "\n".join(table_lines), 0)
    finally:
        writer.close()


def _value_sort_key(value: str):
    if value is None:
        return (1, "")
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def plot_transformation_histograms(summary_rows: list, output_dir: Path, metric: str) -> None:
    """Plot histograms (small multiples) and a combined CDF per transformation."""
    grouped = {}
    for row in summary_rows:
        grouped.setdefault(row["transformation"], []).append(row)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for transformation, rows in grouped.items():
        rows = sorted(rows, key=lambda r: _value_sort_key(r.get("value", "")))
        series_list = []
        labels = []
        for row in rows:
            distances = row.get("distances")
            if distances is None or len(distances) == 0:
                continue
            series_list.append(distances)
            labels.append(row.get("value", "") or "default")

        if not series_list:
            continue

        all_values = np.concatenate(series_list)
        bins = min(50, max(10, int(np.sqrt(len(all_values)))) )
        min_val = int(all_values.min())
        max_val = int(all_values.max())
        if min_val == max_val:
            max_val = min_val + 1
        bin_edges = np.linspace(min_val, max_val, bins)

        cols = 3
        rows_count = int(math.ceil(len(series_list) / cols))
        total_rows = rows_count + 1  # +1 for CDF row

        fig = plt.figure(figsize=(12, 3 * total_rows))
        for idx, (values, label) in enumerate(zip(series_list, labels), start=1):
            ax = plt.subplot(total_rows, cols, idx)
            ax.hist(values, bins=bin_edges, color="#4C72B0", alpha=0.85)
            ax.set_title(f"{transformation}: {label}")
            ax.set_xlabel(f"{metric} distance")
            ax.set_ylabel("Count")

        cdf_ax = plt.subplot(total_rows, 1, total_rows)
        colors = plt.cm.tab10(np.linspace(0, 1, len(series_list)))
        for values, label, color in zip(series_list, labels, colors):
            sorted_vals = np.sort(values)
            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            cdf_ax.plot(sorted_vals, cdf, label=str(label), color=color, linewidth=1.5)
        cdf_ax.set_title(f"{transformation}: CDF (all values)")
        cdf_ax.set_xlabel(f"{metric} distance")
        cdf_ax.set_ylabel("CDF")
        cdf_ax.legend(loc="lower right", ncol=2, fontsize=8)

        plt.tight_layout()
        plot_path = plots_dir / f"{metric}_{transformation}_hist_cdf.png"
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute  distances for hash transformations.")
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("CertPhash/attack/func_logs/coco_val_photodna_nn_cert_ep8"),
        help="Directory containing original and transformation CSV files.",
    )
    parser.add_argument(
        "--original-csv",
        type=Path,
        default=None,
        help="Path to the original CSV. Defaults to logs-dir/coco_val_original.csv.",
    )
    parser.add_argument(
        "--metric",
        choices=["l1", "hamming"],
        default="l1",
        help="Distance metric to compute.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for outputs (CSV summaries and plots).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plotting distance visualizations.",
    )

    parser.add_argument(
        "--hash-field",
        choices=["hash_bin", "hash_hex", "hash_binary"],
        default="hash_bin",
        help="Hash field to use from the CSV.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for parallel processing.",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging.",
    )
    parser.add_argument(
        "--tensorboard-logdir",
        type=Path,
        default=None,
        help="Directory for TensorBoard logs.",
    )
    parser.add_argument(
        "--summary-mean-threshold",
        type=float,
        default=1.0,
        help="Filter out rows with mean below this value when aggregating summary.",
    )
    args = parser.parse_args()

    # Resolve relative paths against the script location to avoid CWD surprises.
    script_root = Path(__file__).resolve().parent.parent
    logs_dir = args.logs_dir
    if not logs_dir.is_absolute():
        logs_dir = script_root / logs_dir
    original_csv = args.original_csv or (logs_dir / "coco_val_original.csv")
    if args.original_csv and not original_csv.is_absolute():
        original_csv = script_root / original_csv
    output_dir = args.output_dir or Path(f"my_work/results/{args.metric}_distances")
    if not output_dir.is_absolute():
        output_dir = script_root / output_dir
    tb_logdir = args.tensorboard_logdir or (output_dir / "tensorboard")
    if not tb_logdir.is_absolute():
        tb_logdir = script_root / tb_logdir
    output_dir.mkdir(parents=True, exist_ok=True)

    original_map = build_original_map(original_csv, args.hash_field)
    transform_csvs = discover_transformation_csvs(logs_dir, original_csv.name)

    summary_rows = []
    if args.workers > 1:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, initializer=init_worker, initargs=(original_map,)
        ) as executor:
            futures = [
                executor.submit(process_transformation, p, logs_dir, args.hash_field, args.metric)
                for p in transform_csvs
            ]
            for future in concurrent.futures.as_completed(futures):
                summary_rows.append(future.result())
    else:
        init_worker(original_map)
        for csv_path in transform_csvs:
            summary_rows.append(
                process_transformation(csv_path, logs_dir, args.hash_field, args.metric)
            )


    summary_df = pd.DataFrame(
        [
            {k: v for k, v in row.items() if k != "distances"}
            for row in summary_rows
        ]
    )
    summary_df.to_csv(output_dir / f"{args.metric}_distance_summary.csv", index=False)
    print(f"Summary written to {output_dir / f'{args.metric}_distance_summary.csv'}")

    agg_df = summarize_by_transformation(summary_df, args.summary_mean_threshold)
    agg_path = output_dir / f"{args.metric}_summary_by_transformation.csv"
    agg_df.to_csv(agg_path, index=False)
    print("_"*60)
    print(f"Aggregated summary written to {agg_path}")
    print("Aggregated summary table (all transformations):")
    print("_"*60)
    print(agg_df.to_string(index=False))
    print("_"*60)
    if not agg_df.empty:
        overall_mean = float(agg_df["mean"].mean())
        print(f"Overall mean across transformations: {overall_mean:.6f}")
    print()


    if not args.no_plots:
        plot_transformation_histograms(summary_rows, output_dir, args.metric)

    if args.tensorboard:
        tb_logdir.mkdir(parents=True, exist_ok=True)
        log_tensorboard(summary_rows, output_dir, tb_logdir)
        print(f"TensorBoard logs written to {tb_logdir}")


if __name__ == "__main__":
    main()
