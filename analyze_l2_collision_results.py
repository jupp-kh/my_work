from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_FINAL_HASH_L1_MAX = 1800.0
DEFAULT_L_INF_MAX = 0.12
PLOT_POINTS = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze one L2-PGD collision result CSV. "
            "The analyzer filters by final_hash_l1, l_inf, and l2, then saves outputs "
            "in a subdirectory named after the CSV."
        )
    )
    parser.add_argument("--csv", type=Path, default=None, help="One L2-PGD collision result CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
             default=Path("/results/adv1_nearest_pair_l2_pgd_collision/"),
        help="Optional output directory. Default: <csv parent>/<csv stem>/",
    )
    parser.add_argument(
        "--final-hash-l1-max",
        type=float,
        default=DEFAULT_FINAL_HASH_L1_MAX,
        help="A row is counted as a collision only when final_hash_l1 <= this value.",
    )
    parser.add_argument(
        "--l-inf-max",
        type=float,
        default=DEFAULT_L_INF_MAX,
        help="A row is counted as a collision only when l_inf <= this value too.",
    )
    parser.add_argument(
        "--l2-max",
        type=float,
        default=None,
        help="A row is counted as a collision only when l2 <= this value too. Defaults to l2_budget from the CSV.",
    )
    parser.add_argument("--plot-points", type=int, default=PLOT_POINTS)
    return parser.parse_args()


def ask_path(prompt: str) -> Path:
    while True:
        value = input(prompt).strip().strip('"')
        if value:
            return Path(value)


def ask_float(prompt: str, default: float | None) -> float:
    while True:
        suffix = f" [{default:g}]" if default is not None else ""
        value = input(f"{prompt}{suffix}: ").strip()
        if not value and default is not None:
            return default
        try:
            parsed = float(value)
        except ValueError:
            print("Please enter a number.")
            continue
        if parsed < 0:
            print("Please enter a non-negative number.")
            continue
        return parsed


def resolve_interactive_args(args: argparse.Namespace) -> argparse.Namespace:
    interactive = args.csv is None
    if args.csv is None:
        args.csv = ask_path("CSV path: ")

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV does not exist: {args.csv}")
    if args.csv.is_dir():
        raise IsADirectoryError(f"Please pass one CSV file, not a directory: {args.csv}")

    preview = pd.read_csv(args.csv, nrows=20)
    default_l2 = infer_default_l2_max(preview)

    if interactive:
        print("\nPress Enter to keep each default.")
        args.final_hash_l1_max = ask_float("final_hash_l1 max", args.final_hash_l1_max)
        args.l_inf_max = ask_float("l_inf max", args.l_inf_max)
        if args.l2_max is None:
            args.l2_max = ask_float("l2 max", default_l2)
        else:
            args.l2_max = ask_float("l2 max", args.l2_max)

    if args.output_dir is None:
        args.output_dir = args.csv.parent / args.csv.stem
    return args


def infer_default_l2_max(df: pd.DataFrame) -> float | None:
    if "l2_budget" not in df.columns:
        return None
    budgets = pd.to_numeric(df["l2_budget"], errors="coerce").dropna().unique()
    if len(budgets) == 1:
        return float(budgets[0])
    return None


def require_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = ["final_hash_l1", "l_inf", "l2", "steps"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    optional = ["pgd_step_size", "l2_budget", "initial_hash_l1"]
    for column in required + optional:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.dropna(subset=required).reset_index(drop=True)


def collision_rate(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def filter_collisions(
    df: pd.DataFrame,
    final_hash_l1_max: float,
    l_inf_max: float,
    l2_max: float,
) -> pd.DataFrame:
    return df[
        (df["final_hash_l1"] <= final_hash_l1_max)
        & (df["l_inf"] <= l_inf_max)
        & (df["l2"] <= l2_max)
    ].copy()


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else np.nan


def make_summary(
    df: pd.DataFrame,
    collisions: pd.DataFrame,
    final_hash_l1_max: float,
    l_inf_max: float,
    l2_max: float,
) -> pd.DataFrame:
    hash_collisions = df[df["final_hash_l1"] <= final_hash_l1_max]
    l_inf_filtered = hash_collisions[hash_collisions["l_inf"] <= l_inf_max]
    l2_filtered = hash_collisions[hash_collisions["l2"] <= l2_max]
    total_rows = len(df)

    row = {
        "final_hash_l1_threshold": final_hash_l1_max,
        "l_inf_threshold": l_inf_max,
        "l2_threshold": l2_max,
        "total_rows": total_rows,
        "hash_collisions": int(len(hash_collisions)),
        "hash_collision_rate_percent": collision_rate(len(hash_collisions), total_rows),
        "hash_and_l_inf_collisions": int(len(l_inf_filtered)),
        "hash_and_l_inf_collision_rate_percent": collision_rate(len(l_inf_filtered), total_rows),
        "hash_and_l2_collisions": int(len(l2_filtered)),
        "hash_and_l2_collision_rate_percent": collision_rate(len(l2_filtered), total_rows),
        "filtered_collisions": int(len(collisions)),
        "filtered_collision_rate_percent": collision_rate(len(collisions), total_rows),
        "mean_l_inf_filtered_collisions": safe_mean(collisions["l_inf"]),
        "mean_L_inf_all_rows": safe_mean(df["l_inf"]),  
        "mean_l2_all_rows": safe_mean(df["l2"]),
        "mean_l2_filtered_collisions": safe_mean(collisions["l2"]),
        "mean_steps_filtered_collisions": safe_mean(collisions["steps"]),
    }

    for column in ["pgd_step_size", "l2_budget"]:
        if column in df.columns:
            values = df[column].dropna().unique()
            if len(values) == 1:
                row[column] = float(values[0])

    return pd.DataFrame([row])


def make_threshold_curve(
    df: pd.DataFrame,
    varied_column: str,
    threshold_column: str,
    fixed_l_inf_max: float,
    fixed_l2_max: float,
    final_hash_l1_max: float,
    max_threshold: float,
    count: int,
) -> pd.DataFrame:
    min_threshold = float(df[varied_column].min())
    thresholds = np.linspace(min_threshold, max_threshold, max(count, 2))
    total_rows = len(df)

    rows = []
    for threshold in thresholds:
        l_inf_max = threshold if varied_column == "l_inf" else fixed_l_inf_max
        l2_max = threshold if varied_column == "l2" else fixed_l2_max
        collisions = filter_collisions(df, final_hash_l1_max, l_inf_max, l2_max)
        rows.append(
            {
                threshold_column: threshold,
                "total_original_rows": total_rows,
                "collisions": int(len(collisions)),
                "percent_original_data_collided": collision_rate(len(collisions), total_rows),
                "mean_l_inf_collisions": safe_mean(collisions["l_inf"]),
                "mean_l2_collisions": safe_mean(collisions["l2"]),
                "mean_steps_collisions": safe_mean(collisions["steps"]),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = resolve_interactive_args(parse_args())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = require_numeric_columns(df)

    if args.l2_max is None:
        default_l2 = infer_default_l2_max(df)
        if default_l2 is None:
            raise ValueError("Could not infer --l2-max. Pass --l2-max explicitly.")
        args.l2_max = default_l2
    collisions = filter_collisions(df, args.final_hash_l1_max, args.l_inf_max, args.l2_max)
    summary = make_summary(df, collisions, args.final_hash_l1_max, args.l_inf_max, args.l2_max)

    summary_path = args.output_dir/ f"l2_collision_summary_l2max_{args.l2_max}_linf_{args.l_inf_max}.csv"

    summary.to_csv(summary_path, index=False)


    print(f"\nLoaded rows: {len(df)}")
    print(
        "Collision definition: "
        f"final_hash_l1 <= {args.final_hash_l1_max:g}, "
        f"l_inf <= {args.l_inf_max:g}, "
        f"l2 <= {args.l2_max:g}"
    )
    print(f"Filtered collisions: {len(collisions)}/{len(df)}")
    print(f"Original data collided: {collision_rate(len(collisions), len(df)):.2f}%")
    print(f"Mean l_inf of filtered collisions: {summary.loc[0, 'mean_l_inf_filtered_collisions']:.6f}")
    print(f"Mean l2 of filtered collisions: {summary.loc[0, 'mean_l2_filtered_collisions']:.6f}")
    print(f"Mean l_inf of all rows: {summary.loc[0, 'mean_L_inf_all_rows']:.6f}")   
    print(f"Mean l2 of all rows: {summary.loc[0, 'mean_l2_all_rows']:.6f}")
    print(f"Mean steps of filtered collisions: {summary.loc[0, 'mean_steps_filtered_collisions']:.2f}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
