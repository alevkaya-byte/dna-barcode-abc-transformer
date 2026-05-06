# -*- coding: utf-8 -*-
"""
run_scalability_L16_N256.py

Scalability stress test for the DNA barcode/index library study.

Experiment setting:
- L = 16
- N = 256
- 3 independent seeds
- Proposed reference-free Transformer candidate generation + ABC
- ABC-only control

Purpose:
This is not a million-scale barcode generation test. It is a moderate-scale
optimization stress test to evaluate whether the proposed library-level
optimization remains usable beyond N=128.

Required script:
- src/barcode_abc_reference_free_transformer.py

Example command:
python src/run_scalability_L16_N256.py
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
RESULTS_DIR = BASE_DIR / "results"
RAW_DIR = RESULTS_DIR / "raw_runs" / "scalability_L16_N256"

MAIN_SCRIPT = SRC_DIR / "barcode_abc_reference_free_transformer.py"

SUMMARY_DIR = RESULTS_DIR / "scalability_L16_N256_summaries"

RAW_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Experiment setting
# ============================================================

LENGTH = 16
SIZE = 256

SEEDS = [
    20260501,
    20260502,
    20260503,
]

COMMON_ARGS = {
    "iters": 20,
    "foods": 12,
    "raw_pool_size": 4096,
    "seed_pool_size": 1024,
    "filter_sample_size": 300,
}

SKIP_EXISTING = True


# ============================================================
# Helpers
# ============================================================

def run_one(seed: int) -> Path:
    out_name = f"scalability_reference_free_transformer_L{LENGTH}_N{SIZE}_seed_{seed}"
    out_dir = RAW_DIR / out_name

    expected_files = [
        out_dir / "best_summary.json",
        out_dir / "run_log.csv",
        out_dir / "run_log_abc_only_control.csv",
        out_dir / "comparison_summary.csv",
        out_dir / "pairwise_distances.csv",
    ]

    if SKIP_EXISTING and all(path.exists() for path in expected_files):
        print(f"[SKIP] Existing complete result found: {out_name}")
        return out_dir

    cmd = [
        sys.executable,
        str(MAIN_SCRIPT),
        "--length", str(LENGTH),
        "--size", str(SIZE),
        "--iters", str(COMMON_ARGS["iters"]),
        "--foods", str(COMMON_ARGS["foods"]),
        "--raw-pool-size", str(COMMON_ARGS["raw_pool_size"]),
        "--seed-pool-size", str(COMMON_ARGS["seed_pool_size"]),
        "--filter-sample-size", str(COMMON_ARGS["filter_sample_size"]),
        "--run-abc-control",
        "--seed", str(seed),
        "--out", str(out_dir),
    ]

    print("=" * 90)
    print(f"Running scalability stress test: L={LENGTH}, N={SIZE}, seed={seed}")
    print(f"Output folder: {out_dir}")
    print("=" * 90)

    subprocess.run(cmd, cwd=str(BASE_DIR), check=True)

    return out_dir


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def read_proposed(folder: Path) -> dict:
    data = read_json(folder / "best_summary.json")

    best = data.get("best_metrics", {}) or {}
    pool = data.get("candidate_pool_summary", {}) or {}

    runtime_seconds = data.get("runtime_seconds", np.nan)
    pool_runtime = pool.get("runtime_total_seconds", np.nan)

    if pd.isna(runtime_seconds) or pd.isna(pool_runtime):
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
        "abc_runtime_seconds": runtime_seconds,
        "candidate_pool_runtime_seconds": pool_runtime,
        "total_runtime_seconds": total_runtime,
    }


def read_abc_only(folder: Path) -> dict:
    comparison_path = folder / "comparison_summary.csv"
    log_path = folder / "run_log_abc_only_control.csv"

    if comparison_path.exists():
        df = pd.read_csv(comparison_path)

        if "method" in df.columns:
            mask = df["method"].astype(str).str.lower().str.contains("abc")
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
                    "abc_runtime_seconds": row.get("runtime_seconds", np.nan),
                    "candidate_pool_runtime_seconds": np.nan,
                    "total_runtime_seconds": row.get("runtime_seconds", np.nan),
                }

    if not log_path.exists():
        raise FileNotFoundError(f"Missing ABC-only log: {log_path}")

    df = pd.read_csv(log_path)

    if df.empty:
        raise ValueError(f"Empty ABC-only log: {log_path}")

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
        "abc_runtime_seconds": np.nan,
        "candidate_pool_runtime_seconds": np.nan,
        "total_runtime_seconds": np.nan,
    }


def build_summaries(rows):
    df = pd.DataFrame(rows)

    all_runs_path = SUMMARY_DIR / "scalability_L16_N256_all_runs.csv"
    group_summary_path = SUMMARY_DIR / "scalability_L16_N256_group_summary.csv"
    delta_raw_path = SUMMARY_DIR / "scalability_L16_N256_delta_raw.csv"
    delta_summary_path = SUMMARY_DIR / "scalability_L16_N256_delta_summary.csv"

    df.to_csv(all_runs_path, index=False, encoding="utf-8-sig")

    numeric_cols = [
        "fitness",
        "min_hamming",
        "avg_hamming",
        "collision_pairs_radius1",
        "duplicate_count",
        "kmer_entropy",
        "valid_gc_fraction",
        "valid_hp_fraction",
        "abc_runtime_seconds",
        "candidate_pool_runtime_seconds",
        "total_runtime_seconds",
    ]

    available_numeric_cols = [
        col for col in numeric_cols if col in df.columns
    ]

    group_summary = (
        df
        .groupby(["length", "size", "method"])[available_numeric_cols]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    group_summary.columns = [
        "_".join([str(x) for x in col if str(x) != ""])
        for col in group_summary.columns
    ]

    group_summary.to_csv(group_summary_path, index=False, encoding="utf-8-sig")

    proposed = df[df["method"] == "Proposed"].copy()
    abc_only = df[df["method"] == "ABC-only"].copy()

    merged = proposed.merge(
        abc_only,
        on=["length", "size", "seed"],
        suffixes=("_proposed", "_abc_only"),
        how="inner",
    )

    for metric in [
        "fitness",
        "min_hamming",
        "avg_hamming",
        "collision_pairs_radius1",
    ]:
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

    merged.to_csv(delta_raw_path, index=False, encoding="utf-8-sig")

    delta_cols = [
        col for col in merged.columns
        if col.startswith("delta_") or col == "proposed_better_fitness"
    ]

    if delta_cols:
        delta_summary = (
            merged
            .groupby(["length", "size"])[delta_cols]
            .agg(["mean", "std", "min", "max", "sum"])
            .reset_index()
        )

        delta_summary.columns = [
            "_".join([str(x) for x in col if str(x) != ""])
            for col in delta_summary.columns
        ]
    else:
        delta_summary = pd.DataFrame()

    delta_summary.to_csv(delta_summary_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 90)
    print("SCALABILITY L16_N256 SUMMARY FILES CREATED")
    print("=" * 90)
    print(all_runs_path)
    print(group_summary_path)
    print(delta_raw_path)
    print(delta_summary_path)

    print("\nGROUP SUMMARY")
    print(group_summary.to_string(index=False))

    print("\nDELTA SUMMARY")
    print(delta_summary.to_string(index=False))


def main():
    if not MAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Main script not found: {MAIN_SCRIPT}")

    rows = []

    for seed in SEEDS:
        folder = run_one(seed)

        proposed_row = read_proposed(folder)
        abc_row = read_abc_only(folder)

        for row in [proposed_row, abc_row]:
            row.update(
                {
                    "length": LENGTH,
                    "size": SIZE,
                    "seed": seed,
                    "output_folder": str(folder.relative_to(BASE_DIR)),
                }
            )
            rows.append(row)

    build_summaries(rows)


if __name__ == "__main__":
    main()
