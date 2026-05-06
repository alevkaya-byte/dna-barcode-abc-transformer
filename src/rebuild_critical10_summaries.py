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


def parse_folder_name(folder_name):
    """
    Example:
    critical10_reference_free_transformer_L12_N64_seed_20260428
    """

    parts = folder_name.split("_")

    length = None
    size = None
    seed = None

    for p in parts:
        if p.startswith("L") and p[1:].isdigit():
            length = int(p[1:])
        if p.startswith("N") and p[1:].isdigit():
            size = int(p[1:])

    if "seed" in parts:
        idx = parts.index("seed")
        seed = int(parts[idx + 1])

    if length is None or size is None or seed is None:
        raise ValueError(f"Could not parse folder name: {folder_name}")

    return length, size, seed


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_proposed_from_best_summary(folder):
    path = folder / "best_summary.json"

    if not path.exists():
        raise FileNotFoundError(f"Missing best_summary.json: {folder}")

    js = read_json(path)

    best = js.get("best_metrics", {})
    pool = js.get("candidate_pool_summary", {})

    row = {
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
        "runtime_seconds": js.get("runtime_seconds", np.nan),
        "candidate_pool_runtime_seconds": pool.get("runtime_total_seconds", np.nan),
        "total_runtime_seconds": (
            js.get("runtime_seconds", 0.0)
            + pool.get("runtime_total_seconds", 0.0)
        ),
    }

    return row


def read_abc_only_from_comparison_or_log(folder):
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
                r = abc_rows.iloc[0].to_dict()

                return {
                    "method": "ABC-only",
                    "fitness": r.get("fitness", np.nan),
                    "min_hamming": r.get("min_hamming", np.nan),
                    "avg_hamming": r.get("avg_hamming", np.nan),
                    "gc_penalty": r.get("gc_penalty", np.nan),
                    "homopolymer_penalty": r.get("homopolymer_penalty", np.nan),
                    "collision_pairs_radius1": r.get(
                        "collision_pairs_radius1", np.nan
                    ),
                    "duplicate_count": r.get("duplicate_count", np.nan),
                    "kmer_entropy": r.get("kmer_entropy", np.nan),
                    "valid_gc_fraction": r.get("valid_gc_fraction", np.nan),
                    "valid_hp_fraction": r.get("valid_hp_fraction", np.nan),
                    "runtime_seconds": r.get("runtime_seconds", np.nan),
                    "candidate_pool_runtime_seconds": np.nan,
                    "total_runtime_seconds": r.get("runtime_seconds", np.nan),
                }

    # Fallback: use last row of ABC-only run log.
    if not log_path.exists():
        raise FileNotFoundError(f"Missing ABC-only log: {folder}")

    df = pd.read_csv(log_path)

    if df.empty:
        raise ValueError(f"Empty ABC-only log: {folder}")

    r = df.iloc[-1].to_dict()

    return {
        "method": "ABC-only",
        "fitness": r.get("best_fitness", r.get("fitness", np.nan)),
        "min_hamming": r.get("min_hamming", r.get("minD", np.nan)),
        "avg_hamming": r.get("avg_hamming", r.get("avgD", np.nan)),
        "gc_penalty": np.nan,
        "homopolymer_penalty": np.nan,
        "collision_pairs_radius1": r.get(
            "collision_pairs_radius1",
            r.get("coll", r.get("collision", np.nan)),
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
            row.update({
                "length": length,
                "size": size,
                "seed": seed,
                "output_folder": folder.name,
            })

            rows.append(row)

    df = pd.DataFrame(rows)

    first_cols = ["length", "size", "seed", "method"]
    other_cols = [c for c in df.columns if c not in first_cols]
    df = df[first_cols + other_cols]

    return df

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
            row.update({
                "length": length,
                "size": size,
                "seed": seed,
                "output_folder": folder.name,
            })

            rows.append(row)

    df = pd.DataFrame(rows)

    # Order columns
    first_cols = ["length", "size", "seed", "method"]
    other_cols = [c for c in df.columns if c not in first_cols]
    df = df[first_cols + other_cols]

    return df


def make_group_summary(df):
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

    available = [m for m in metrics if m in df.columns]

    summary = (
        df
        .groupby(["length", "size", "method"])[available]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    summary.columns = [
        "_".join([str(x) for x in col if str(x) != ""])
        for col in summary.columns
    ]

    return summary


def make_delta_summary(df):
    proposed = df[df["method"] == "Proposed"].copy()
    abc = df[df["method"] == "ABC-only"].copy()

    merged = proposed.merge(
        abc,
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

    for m in metrics:
        p = f"{m}_proposed"
        a = f"{m}_abc_only"

        if p in merged.columns and a in merged.columns:
            merged[f"delta_{m}"] = merged[p] - merged[a]

    merged["proposed_better_fitness"] = (
        merged["delta_fitness"] > 0
    ).astype(int)

    delta_cols = [
        "delta_fitness",
        "delta_min_hamming",
        "delta_avg_hamming",
        "delta_collision_pairs_radius1",
        "proposed_better_fitness",
    ]

    summary = (
        merged
        .groupby(["length", "size"])[delta_cols]
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
