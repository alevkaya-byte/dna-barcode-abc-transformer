# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:15:06 2026

@author: kaya-
"""

# -*- coding: utf-8 -*-
"""
run_scalability_L16_N256.py

Scalability stress test for the DNA barcode/index library manuscript.

Runs:
- L = 16
- N = 256
- 3 independent seeds
- Proposed reference-free Transformer + ABC
- ABC-only control

Purpose:
This is NOT a million-scale barcode generation test.
It is a moderate-scale optimization stress test to show whether the
proposed library-level ABC optimization remains usable beyond N=128.

Required file in the same folder:
barcode_abc_reference_free_transformer.py

Spyder run:
runfile(
    'C:/Users/kaya-/Desktop/ABC/run_scalability_L16_N256.py',
    wdir='C:/Users/kaya-/Desktop/ABC'
)
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

BASE_DIR = Path(r"C:/Users/kaya-/Desktop/ABC")
MAIN_SCRIPT = BASE_DIR / "barcode_abc_reference_free_transformer.py"

SUMMARY_DIR = BASE_DIR / "scalability_L16_N256_summaries"
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

def run_one(seed):
    out_name = f"scalability_reference_free_transformer_L{LENGTH}_N{SIZE}_seed_{seed}"
    out_dir = BASE_DIR / out_name

    expected_files = [
        out_dir / "best_summary.json",
        out_dir / "run_log.csv",
        out_dir / "run_log_abc_only_control.csv",
        out_dir / "comparison_summary.csv",
        out_dir / "pairwise_distances.csv",
    ]

    if SKIP_EXISTING and all(p.exists() for p in expected_files):
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
        "--out", out_name,
    ]

    print("=" * 90)
    print(f"Running scalability stress test: L={LENGTH}, N={SIZE}, seed={seed}")
    print(f"Output folder: {out_name}")
    print("=" * 90)

    subprocess.run(cmd, cwd=str(BASE_DIR), check=True)

    return out_dir


def read_json(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_proposed(folder):
    js = read_json(folder / "best_summary.json")

    best = js.get("best_metrics", {})
    pool = js.get("candidate_pool_summary", {})

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
        "abc_runtime_seconds": js.get("runtime_seconds", np.nan),
        "candidate_pool_runtime_seconds": pool.get("runtime_total_seconds", np.nan),
        "total_runtime_seconds": (
            js.get("runtime_seconds", 0.0)
            + pool.get("runtime_total_seconds", 0.0)
        ),
    }


def read_abc_only(folder):
    comparison_path = folder / "comparison_summary.csv"
    log_path = folder / "run_log_abc_only_control.csv"

    # Prefer comparison_summary.csv if possible.
    if comparison_path.exists():
        df = pd.read_csv(comparison_path)

        if "method" in df.columns:
            mask = df["method"].astype(str).str.lower().str.contains("abc")
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
                    "collision_pairs_radius1": r.get("collision_pairs_radius1", np.nan),
                    "duplicate_count": r.get("duplicate_count", np.nan),
                    "kmer_entropy": r.get("kmer_entropy", np.nan),
                    "valid_gc_fraction": r.get("valid_gc_fraction", np.nan),
                    "valid_hp_fraction": r.get("valid_hp_fraction", np.nan),
                    "abc_runtime_seconds": r.get("runtime_seconds", np.nan),
                    "candidate_pool_runtime_seconds": np.nan,
                    "total_runtime_seconds": r.get("runtime_seconds", np.nan),
                }

    # Fallback to ABC-only run log.
    if not log_path.exists():
        return {
            "method": "ABC-only",
            "fitness": np.nan,
            "min_hamming": np.nan,
            "avg_hamming": np.nan,
            "gc_penalty": np.nan,
            "homopolymer_penalty": np.nan,
            "collision_pairs_radius1": np.nan,
            "duplicate_count": np.nan,
            "kmer_entropy": np.nan,
            "valid_gc_fraction": np.nan,
            "valid_hp_fraction": np.nan,
            "abc_runtime_seconds": np.nan,
            "candidate_pool_runtime_seconds": np.nan,
            "total_runtime_seconds": np.nan,
        }

    df = pd.read_csv(log_path)
    if df.empty:
        return {
            "method": "ABC-only",
            "fitness": np.nan,
            "min_hamming": np.nan,
            "avg_hamming": np.nan,
            "gc_penalty": np.nan,
            "homopolymer_penalty": np.nan,
            "collision_pairs_radius1": np.nan,
            "duplicate_count": np.nan,
            "kmer_entropy": np.nan,
            "valid_gc_fraction": np.nan,
            "valid_hp_fraction": np.nan,
            "abc_runtime_seconds": np.nan,
            "candidate_pool_runtime_seconds": np.nan,
            "total_runtime_seconds": np.nan,
        }

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

    group_summary = (
        df
        .groupby(["length", "size", "method"])[numeric_cols]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    group_summary.columns = [
        "_".join([str(x) for x in col if str(x) != ""])
        for col in group_summary.columns
    ]

    group_summary.to_csv(group_summary_path, index=False, encoding="utf-8-sig")

    proposed = df[df["method"] == "Proposed"].copy()
    abc = df[df["method"] == "ABC-only"].copy()

    merged = proposed.merge(
        abc,
        on=["length", "size", "seed"],
        suffixes=("_proposed", "_abc_only"),
        how="inner",
    )

    for metric in ["fitness", "min_hamming", "avg_hamming", "collision_pairs_radius1"]:
        p = f"{metric}_proposed"
        a = f"{metric}_abc_only"

        if p in merged.columns and a in merged.columns:
            merged[f"delta_{metric}"] = merged[p] - merged[a]

    if "delta_fitness" in merged.columns:
        merged["proposed_better_fitness"] = (merged["delta_fitness"] > 0).astype(int)

    merged.to_csv(delta_raw_path, index=False, encoding="utf-8-sig")

    delta_cols = [
        c for c in merged.columns
        if c.startswith("delta_") or c == "proposed_better_fitness"
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

        delta_summary.to_csv(delta_summary_path, index=False, encoding="utf-8-sig")
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
            row.update({
                "length": LENGTH,
                "size": SIZE,
                "seed": seed,
                "output_folder": folder.name,
            })
            rows.append(row)

    build_summaries(rows)


if __name__ == "__main__":
    main()