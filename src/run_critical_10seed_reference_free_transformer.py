# -*- coding: utf-8 -*-
"""
run_critical_10seed_reference_free_transformer.py

Runs the critical 10-seed experiments for the DNA barcode/index library study.

Experiment settings:
- L=12, N=64
- L=12, N=128

Each setting is evaluated over 10 independent seeds using:
- Proposed reference-free Transformer candidate generation + ABC
- ABC-only control

Required script:
- src/barcode_abc_reference_free_transformer.py

Example command:
python src/run_critical_10seed_reference_free_transformer.py
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
RAW_DIR = RESULTS_DIR / "raw_runs" / "critical_10seed"

MAIN_SCRIPT = SRC_DIR / "barcode_abc_reference_free_transformer.py"

SUMMARY_DIR = RESULTS_DIR / "critical_10seed_summaries"

RAW_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Experiment settings
# ============================================================

EXPERIMENTS = [
    {"length": 12, "size": 64},
    {"length": 12, "size": 128},
]

SEEDS = [
    20260428,
    20260429,
    20260430,
    20260431,
    20260432,
    20260433,
    20260434,
    20260435,
    20260436,
    20260437,
]

COMMON_ARGS = {
    "iters": 20,
    "foods": 12,
    "raw_pool_size": 1024,
    "seed_pool_size": 256,
    "filter_sample_size": 100,
}

SKIP_EXISTING = True


# ============================================================
# Helpers
# ============================================================

def run_one_experiment(length: int, size: int, seed: int) -> Path:
    out_name = f"critical10_reference_free_transformer_L{length}_N{size}_seed_{seed}"
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
        "--length", str(length),
        "--size", str(size),
        "--iters", str(COMMON_ARGS["iters"]),
        "--foods", str(COMMON_ARGS["foods"]),
        "--raw-pool-size", str(COMMON_ARGS["raw_pool_size"]),
        "--seed-pool-size", str(COMMON_ARGS["seed_pool_size"]),
        "--filter-sample-size", str(COMMON_ARGS["filter_sample_size"]),
        "--run-abc-control",
        "--seed", str(seed),
        "--out", str(out_dir),
    ]

    print("=" * 80)
    print(f"Running L={length}, N={size}, seed={seed}")
    print(f"Output folder: {out_dir}")
    print("=" * 80)

    subprocess.run(cmd, cwd=str(BASE_DIR), check=True)

    return out_dir


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def safe_get(data: dict, key: str, default=np.nan):
    return data.get(key, default) if isinstance(data, dict) else default


def read_proposed_summary(out_dir: Path) -> dict:
    summary = read_json(out_dir / "best_summary.json")

    best = summary.get("best_metrics", {})
    pool = summary.get("candidate_pool_summary", {})

    runtime_seconds = summary.get("runtime_seconds", np.nan)
    pool_runtime = pool.get("runtime_total_seconds", np.nan)

    if pd.isna(runtime_seconds) or pd.isna(pool_runtime):
        total_runtime = np.nan
    else:
        total_runtime = runtime_seconds + pool_runtime

    return {
        "method": "Proposed",
        "fitness": safe_get(best, "fitness"),
        "min_hamming": safe_get(best, "min_hamming"),
        "avg_hamming": safe_get(best, "avg_hamming"),
        "gc_penalty": safe_get(best, "gc_penalty"),
        "homopolymer_penalty": safe_get(best, "homopolymer_penalty"),
        "collision_pairs_radius1": safe_get(best, "collision_pairs_radius1"),
        "duplicate_count": safe_get(best, "duplicate_count"),
        "kmer_entropy": safe_get(best, "kmer_entropy"),
        "valid_gc_fraction": safe_get(best, "valid_gc_fraction"),
        "valid_hp_fraction": safe_get(best, "valid_hp_fraction"),
        "runtime_seconds": runtime_seconds,
        "candidate_pool_runtime_seconds": pool_runtime,
        "total_runtime_seconds": total_runtime,
    }


def read_abc_only_summary(out_dir: Path) -> dict:
    comparison_path = out_dir / "comparison_summary.csv"
    log_path = out_dir / "run_log_abc_only_control.csv"

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
                    "collision_pairs_radius1": row.get("collision_pairs_radius1", np.nan),
                    "duplicate_count": row.get("duplicate_count", np.nan),
                    "kmer_entropy": row.get("kmer_entropy", np.nan),
                    "valid_gc_fraction": row.get("valid_gc_fraction", np.nan),
                    "valid_hp_fraction": row.get("valid_hp_fraction", np.nan),
                    "runtime_seconds": row.get("runtime_seconds", np.nan),
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
        "runtime_seconds": np.nan,
        "candidate_pool_runtime_seconds": np.nan,
        "total_runtime_seconds": np.nan,
    }


def collect_one_result(out_dir: Path, length: int, size: int, seed: int) -> list:
    rows = []

    proposed = read_proposed_summary(out_dir)
    abc_only = read_abc_only_summary(out_dir)

    for row in [proposed, abc_only]:
        row.update(
            {
                "length": length,
                "size": size,
                "seed": seed,
                "output_folder": str(out_dir.relative_to(BASE_DIR)),
            }
        )
        rows.append(row)

    return rows


def make_group_summary(all_runs_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
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

    available_numeric_cols = [
        col for col in numeric_cols if col in all_runs_df.columns
    ]

    summary = (
        all_runs_df
        .groupby(["length", "size", "method"])[available_numeric_cols]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    summary.columns = [
        "_".join([str(x) for x in col if str(x) != ""])
        for col in summary.columns.values
    ]

    return summary


def make_delta_summary(all_runs_df: pd.DataFrame):
    proposed = all_runs_df[all_runs_df["method"] == "Proposed"].copy()
    abc_only = all_runs_df[all_runs_df["method"] == "ABC-only"].copy()

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

    delta_cols = [
        "delta_fitness",
        "delta_min_hamming",
        "delta_avg_hamming",
        "delta_collision_pairs_radius1",
        "proposed_better_fitness",
    ]

    available_delta_cols = [
        col for col in delta_cols if col in merged.columns
    ]

    delta_summary = (
        merged
        .groupby(["length", "size"])[available_delta_cols]
        .agg(["mean", "std", "min", "max", "sum"])
        .reset_index()
    )

    delta_summary.columns = [
        "_".join([str(x) for x in col if str(x) != ""])
        for col in delta_summary.columns.values
    ]

    return merged, delta_summary


# ============================================================
# Main
# ============================================================

def main():
    if not MAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Main script not found: {MAIN_SCRIPT}")

    all_rows = []

    for experiment in EXPERIMENTS:
        length = experiment["length"]
        size = experiment["size"]

        for seed in SEEDS:
            out_dir = run_one_experiment(length, size, seed)
            all_rows.extend(collect_one_result(out_dir, length, size, seed))

    all_runs_df = pd.DataFrame(all_rows)

    all_runs_path = SUMMARY_DIR / "critical10_all_runs.csv"
    group_summary_path = SUMMARY_DIR / "critical10_group_summary.csv"
    delta_raw_path = SUMMARY_DIR / "critical10_delta_raw.csv"
    delta_summary_path = SUMMARY_DIR / "critical10_delta_summary.csv"

    all_runs_df.to_csv(all_runs_path, index=False, encoding="utf-8-sig")

    group_summary = make_group_summary(all_runs_df)
    group_summary.to_csv(group_summary_path, index=False, encoding="utf-8-sig")

    delta_raw, delta_summary = make_delta_summary(all_runs_df)
    delta_raw.to_csv(delta_raw_path, index=False, encoding="utf-8-sig")
    delta_summary.to_csv(delta_summary_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("CRITICAL 10-SEED EXPERIMENTS COMPLETED")
    print("=" * 80)
    print(f"Saved: {all_runs_path}")
    print(f"Saved: {group_summary_path}")
    print(f"Saved: {delta_raw_path}")
    print(f"Saved: {delta_summary_path}")

    print("\nQuick group summary:")
    print(group_summary.to_string(index=False))

    print("\nQuick delta summary:")
    print(delta_summary.to_string(index=False))


if __name__ == "__main__":
    main()
