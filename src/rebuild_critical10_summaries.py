# -*- coding: utf-8 -*-
"""
rebuild_critical10_summaries.py

Rebuilds summary CSV files from completed critical 10-seed experiment folders.

This script does not rerun experiments. It reads existing output folders matching:

- critical10_reference_free_transformer_L12_N64_seed_*
- critical10_reference_free_transformer_L12_N128_seed_*

and reconstructs:

- critical10_all_runs_FIXED.csv
- critical10_group_summary_FIXED.csv
- critical10_delta_raw_FIXED.csv
- critical10_delta_summary_FIXED.csv

Example command:
python src/rebuild_critical10_summaries.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
SUMMARY_DIR = RESULTS_DIR / "critical_10seed_summaries"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

FOLDER_PATTERNS = [
    "critical10_reference_free_transformer_L12_N64_seed_*",
    "critical10_reference_free_transformer_L12_N128_seed_*",
]


def parse_folder_name(folder_name: str):
    """
    Example:
    critical10_reference_free_transformer_L12_N64_seed_20260428
    """

    parts = folder_name.split("_")

    length = None
    size = None
    seed = None

    for part in parts:
        if part.startswith("L") and part[1:].isdigit():
            length = int(part[1:])
        elif part.startswith("N") and part[1:].isdigit():
            size = int(part[1:])

    if "seed" in parts:
        seed_index = parts.index("seed")
        if seed_index + 1 < len(parts):
            seed = int(parts[seed_index + 1])

    if length is None or size is None or seed is None:
        raise ValueError(f"Could not parse folder name: {folder_name}")

    return length, size, seed


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def read_proposed_from_best_summary(folder: Path):
    path = folder / "best_summary.json"

    if not path.exists():
        raise FileNotFoundError(f"Missing best_summary.json: {folder}")

    data = read_json(path)

    best = data.get("best_metrics", {})
    pool = data.get("candidate_pool_summary", {})

    runtime_seconds = data.get("runtime_seconds", np.nan)
    pool_runtime = pool.get("runtime_total_seconds", np.nan)

    if np.isnan(runtime_seconds) or np.isnan(pool_runtime):
        total_runtime = np.nan
    else:
        total_runtime = runtime_seconds + pool_runtime

    return {
        "method": "Proposed",
        "fitness": best.get("fitness", np.nan),
        "min_hamming": best.get("min_hamming", np.nan),
        "avg_hamming": best.get("avg_hamming", np.nan),
        "gc_penalty": best.get("gc_penalty", np.nan),
        "homopolymer_penalty": best.get("homopolymer_penalty", np.nan),
        "collision_pairs_radius1": best.get("collision_pairs_radius1", np.nan),
        "duplicate_count": best.get("duplicate_count", np.nan),
        "kmer_entropy": best.get("kmer_entropy", np.nan),
        "valid_gc_fraction": best.get("valid_gc_fraction", np.nan),
        "valid_hp_fraction": best.get("valid_hp_fraction", np.nan),
        "runtime_seconds": runtime_seconds,
        "candidate_pool_runtime_seconds": pool_runtime,
        "total_runtime_seconds": total_runtime,
    }


def read_abc_only_from_comparison_or_log(folder: Path):
    comparison_path = folder / "comparison_summary.csv"
    log_path = folder / "run_log_abc_only_control.csv"

    # Prefer comparison_summary.csv if it contains ABC-only metrics.
    if comparison_path.exists():
        df = pd.read_csv(comparison_path)

        method_col = None
        for col in df.columns:
            if col.lower() == "method":
                method_col = col
                break

        if method_col is not None:
            mask = df[method_col].astype(str).str.lower().str.contains("abc")
            abc_rows = df[mask]

            if len(abc_rows) > 0:
                row = abc_rows.iloc[0].to_dict()

                return {
                    "method": "ABC-only",
                    "fitness": row.get("fitness", np.nan),
                    "min_hamming": row.get("min_hamming", np.nan),
                    "avg_hamming": row.get("avg_hamming", np.nan),
                    "gc_penalty": row.get("gc_penalty", np.nan),
                    "homopolymer_penalty": row.get("homopolymer_penalty", np.nan),
                    "collision_pairs_radius1": row.get(
                        "collision_pairs_radius1", np.nan
                    ),
                    "duplicate_count": row.get("duplicate_count", np.nan),
                    "kmer_entropy": row.get("kmer_entropy", np.nan),
                    "valid_gc_fraction": row.get("valid_gc_fraction", np.nan),
                    "valid_hp_fraction": row.get("valid_hp_fraction", np.nan),
                    "runtime_seconds": row.get("runtime_seconds", np.nan),
                    "candidate_pool_runtime_seconds": np.nan,
                    "total_runtime_seconds": row.get("runtime_seconds", np.nan),
                }

    # Fallback: use the last row of ABC-only run log.
    if not log_path.exists():
        raise FileNotFoundError(f"Missing ABC-only log: {folder}")

    df = pd.read_csv(log_path)

    if df.empty:
        raise ValueError(f"Empty ABC-only log: {folder}")

    row = df.iloc[-1].to_dict()

    return {
        "method": "ABC-only",
        "fitness": row.get("best_fitness", row.get("fitness", np.nan)),
        "min_hamming": row.get("min_hamming", row.get("minD", np.nan)),
        "avg_hamming": row.get("avg_hamming", row.get("avgD", np.nan)),
        "gc_penalty": np.nan,
        "homopolymer_penalty": np.nan,
        "collision_pairs_radius1": row.get(
            "collision_pairs_radius1",
            row.get("coll", row.get("collision", np.nan)),
        ),
        "duplicate_count": np.nan,
        "kmer_entropy": np.nan,
        "valid_gc_fraction": np.nan,
        "valid_hp_fraction": np.nan,
        "runtime_seconds": np.nan,
        "candidate_pool_runtime_seconds": np.nan,
        "total_runtime_seconds": np.nan,
    }


def collect_all_runs():
    folders = []

    search_dirs = [
        RESULTS_DIR,
        RESULTS_DIR / "raw_runs",
        BASE_DIR,
    ]

    for pattern in FOLDER_PATTERNS:
        for search_dir in search_dirs:
            if search_dir.exists():
                folders.extend(sorted(search_dir.glob(pattern)))

    # Remove possible duplicates while preserving order.
    unique_folders = []
    seen = set()

    for folder in folders:
        resolved = folder.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_folders.append(folder)

    folders = unique_folders

    if not folders:
        raise FileNotFoundError(
            "No critical 10-seed output folders found. "
            "Check RESULTS_DIR, raw run folders, and folder names."
        )

    rows = []

    print("=" * 80)
    print("FOUND OUTPUT FOLDERS")
    print("=" * 80)

    for folder in folders:
        if not folder.is_dir():
            continue

        print(folder.name)

        length, size, seed = parse_folder_name(folder.name)

        proposed = read_proposed_from_best_summary(folder)
        abc_only = read_abc_only_from_comparison_or_log(folder)

        for row in [proposed, abc_only]:
            row.update(
                {
                    "length": length,
                    "size": size,
                    "seed": seed,
                    "output_folder": folder.name,
                }
            )
            rows.append(row)

    df = pd.DataFrame(rows)

    first_cols = ["length", "size", "seed", "method"]
    other_cols = [col for col in df.columns if col not in first_cols]
    df = df[first_cols + other_cols]

    return df


def make_group_summary(df: pd.DataFrame):
    metrics = [
        "fitness",
        "min_hamming",
        "avg_hamming",
        "collision_pairs_radius1",
        "duplicate_count",
        "kmer_entropy",
        "valid_gc_fraction",
        "valid_hp_fraction",
        "runtime_seconds",
        "candidate_pool_runtime_seconds",
        "total_runtime_seconds",
    ]

    available = [metric for metric in metrics if metric in df.columns]

    summary = (
        df.groupby(["length", "size", "method"])[available]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    summary.columns = [
        "_".join([str(x) for x in col if str(x) != ""])
        for col in summary.columns
    ]

    return summary


def make_delta_summary(df: pd.DataFrame):
    proposed = df[df["method"] == "Proposed"].copy()
    abc_only = df[df["method"] == "ABC-only"].copy()

    merged = proposed.merge(
        abc_only,
        on=["length", "size", "seed"],
        suffixes=("_proposed", "_abc_only"),
        how="inner",
    )

    metrics = [
        "fitness",
        "min_hamming",
        "avg_hamming",
        "collision_pairs_radius1",
    ]

    for metric in metrics:
        proposed_col = f"{metric}_proposed"
        abc_col = f"{metric}_abc_only"

        if proposed_col in merged.columns and abc_col in merged.columns:
            merged[f"delta_{metric}"] = (
                merged[proposed_col] - merged[abc_col]
            )

    if "delta_fitness" in merged.columns:
        merged["proposed_better_fitness"] = (
            merged["delta_fitness"] > 0
        ).astype(int)
    else:
        merged["proposed_better_fitness"] = np.nan

    delta_cols = [
        "delta_fitness",
        "delta_min_hamming",
        "delta_avg_hamming",
        "delta_collision_pairs_radius1",
        "proposed_better_fitness",
    ]

    available_delta_cols = [col for col in delta_cols if col in merged.columns]

    summary = (
        merged.groupby(["length", "size"])[available_delta_cols]
        .agg(["mean", "std", "min", "max", "sum"])
        .reset_index()
    )

    summary.columns = [
        "_".join([str(x) for x in col if str(x) != ""])
        for col in summary.columns
    ]

    return merged, summary


def main():
    all_runs = collect_all_runs()

    all_runs_path = SUMMARY_DIR / "critical10_all_runs_FIXED.csv"
    group_path = SUMMARY_DIR / "critical10_group_summary_FIXED.csv"
    delta_raw_path = SUMMARY_DIR / "critical10_delta_raw_FIXED.csv"
    delta_summary_path = SUMMARY_DIR / "critical10_delta_summary_FIXED.csv"

    all_runs.to_csv(all_runs_path, index=False, encoding="utf-8-sig")

    group = make_group_summary(all_runs)
    group.to_csv(group_path, index=False, encoding="utf-8-sig")

    delta_raw, delta_summary = make_delta_summary(all_runs)
    delta_raw.to_csv(delta_raw_path, index=False, encoding="utf-8-sig")
    delta_summary.to_csv(delta_summary_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("FIXED SUMMARIES CREATED")
    print("=" * 80)
    print(all_runs_path)
    print(group_path)
    print(delta_raw_path)
    print(delta_summary_path)

    print("\nGROUP SUMMARY")
    print(group.to_string(index=False))

    print("\nDELTA SUMMARY")
    print(delta_summary.to_string(index=False))


if __name__ == "__main__":
    main()
