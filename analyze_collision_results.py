from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CSV = Path("my_work/results/adv1_nearest_pair_coco_val/nearest_pair_collision_results.csv")
LINF_PLOT_POINTS = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze collision results using final_hash_l1 and l_inf thresholds. "
            "Reports the percentage of the original CSV that satisfies both "
            "filters, plus mean l_inf/l2/steps for those rows."
        )
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("my_work/results/collision_csv_analysis"),
    )
    parser.add_argument(
        "--final-hash-l1-max",
        type=float,
        default=1800,
        help="A row is counted as a collision when final_hash_l1 <= this value.",
    )
    parser.add_argument(
        "--l-inf-max",
        type=float,
        default=0.12,
        help="A row is counted as a collision only when l_inf <= this value too.",
    )
    return parser.parse_args()


def require_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = ["final_hash_l1", "l_inf", "l2", "steps"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.dropna(subset=required).reset_index(drop=True)


def collision_rate(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def filter_collisions(
    df: pd.DataFrame,
    final_hash_l1_max: float,
    l_inf_max: float,
) -> pd.DataFrame:
    return df[(df["final_hash_l1"] <= final_hash_l1_max) & (df["l_inf"] <= l_inf_max)].copy()


def make_linf_curve(
    df: pd.DataFrame,
    final_hash_l1_max: float,
    max_linf_threshold: float,
    count: int,
) -> pd.DataFrame:
    min_linf = float(df["l_inf"].min())
    thresholds = np.linspace(min_linf, max_linf_threshold, max(count, 2))
    total_rows = len(df)

    rows = []
    for threshold in thresholds:
        collisions = filter_collisions(df, final_hash_l1_max, threshold)
        rows.append(
            {
                "l_inf_threshold": threshold,
                "total_original_rows": total_rows,
                "collisions": int(len(collisions)),
                "percent_original_data_collided": collision_rate(len(collisions), total_rows),
                "mean_l_inf_collisions": float(collisions["l_inf"].mean()) if len(collisions) else np.nan,
                "mean_l2_collisions": float(collisions["l2"].mean()) if len(collisions) else np.nan,
                "mean_steps_collisions": float(collisions["steps"].mean()) if len(collisions) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def plot_collision_rate_vs_linf(curve: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(
        curve["l_inf_threshold"],
        curve["percent_original_data_collided"],
        marker="o",
        markersize=3,
        linewidth=2.0,
        color="#2f6f73",
    )
    ax.set_title("Original Data Collided vs l_inf Threshold")
    ax.set_xlabel("l_inf threshold")
    ax.set_ylabel("original data collided (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = require_numeric_columns(df)

    collisions = filter_collisions(df, args.final_hash_l1_max, args.l_inf_max)
    total_rows = len(df)
    collision_count = len(collisions)
    cr_percent = collision_rate(collision_count, total_rows)

    summary = pd.DataFrame(
        [
            {
                "final_hash_l1_threshold": args.final_hash_l1_max,
                "l_inf_threshold": args.l_inf_max,
                "total_rows": total_rows,
                "collisions": collision_count,
                "percent_original_data_collided": cr_percent,
                "mean_l_inf_collisions": float(collisions["l_inf"].mean()) if collision_count else np.nan,
                "mean_l2_collisions": float(collisions["l2"].mean()) if collision_count else np.nan,
                "mean_steps_collisions": float(collisions["steps"].mean()) if collision_count else np.nan,
            }
        ]
    )

    curve = make_linf_curve(
        df,
        args.final_hash_l1_max,
        args.l_inf_max,
        LINF_PLOT_POINTS,
    )

    summary_path = args.output_dir / "collision_summary.csv"
    curve_path = args.output_dir / "original_data_collided_vs_linf.csv"
    plot_path = args.output_dir / "original_data_collided_vs_linf.png"

    summary.to_csv(summary_path, index=False)
    curve.to_csv(curve_path, index=False)
    plot_collision_rate_vs_linf(curve, plot_path)

    print(f"Loaded rows: {total_rows}")
    print(
        "Collision definition: "
        f"final_hash_l1 <= {args.final_hash_l1_max:g} "
        f"and l_inf <= {args.l_inf_max:g}"
    )
    print(f"Collisions: {collision_count}/{total_rows}")
    print(f"Original data collided: {cr_percent:.2f}%")
    print(f"Mean l_inf of collisions: {summary.loc[0, 'mean_l_inf_collisions']:.6f}")
    print(f"Mean l2 of collisions: {summary.loc[0, 'mean_l2_collisions']:.6f}")
    print(f"Mean steps of collisions: {summary.loc[0, 'mean_steps_collisions']:.2f}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote l_inf curve: {curve_path}")
    print(f"Wrote plot: {plot_path}")


if __name__ == "__main__":
    main()
