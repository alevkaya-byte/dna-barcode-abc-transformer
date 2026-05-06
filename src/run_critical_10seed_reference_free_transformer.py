# -*- coding: utf-8 -*-
"""
Created on Mon May  4 18:54:30 2026

@author: kaya-
"""

# -*- coding: utf-8 -*-
"""
run_critical_10seed_reference_free_transformer.py

Critical 10-seed experiments for the DNA barcode/index library manuscript.

Runs:
1) L=12, N=64
2) L=12, N=128

Each setting is evaluated over 10 independent seeds with:
- Proposed reference-free Transformer candidate generation + ABC
- ABC-only control

Required existing script in the same folder:
barcode_abc_reference_free_transformer.py

Spyder run:
runfile(
    'C:/Users/kaya-/Desktop/ABC/run_critical_10seed_reference_free_transformer.py',
    wdir='C:/Users/kaya-/Desktop/ABC'
)
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(r"C:/Users/kaya-/Desktop/ABC")
MAIN_SCRIPT = BASE_DIR / "barcode_abc_reference_free_transformer.py"

# Main output summary folder
SUMMARY_DIR = BASE_DIR / "critical_10seed_summaries"
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

# If True, completed output folders are not re-run.
SKIP_EXISTING = True


# ============================================================
# Helpers
# ============================================================

def run_one_experiment(length, size, seed):
    out_name = f"critical10_reference_free_transformer_L{length}_N{size}_seed_{seed}"
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
        "--length", str(length),
        "--size", str(size),
        "--iters", str(COMMON_ARGS["iters"]),
        "--foods", str(COMMON_ARGS["foods"]),
        "--raw-pool-size", str(COMMON_ARGS["raw_pool_size"]),
        "--seed-pool-size", str(COMMON_ARGS["seed_pool_size"]),
        "--filter-sample-size", str(COMMON_ARGS["filter_sample_size"]),
        "--run-abc-control",
        "--seed", str(seed),
        "--out", out_name,
    ]

    print("=" * 80)
    print(f"Running L={length}, N={size}, seed={seed}")
    print("Output:", out_name)
    print("=" * 80)

    subprocess.run(cmd, cwd=str(BASE_DIR), check=True)

    return out_dir


def read_json(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_get(d, key, default=None):
    return d.get(key, default) if isinstance(d, dict) else default


def read_last_log_metrics(path):
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    if df.empty:
        return {}

    last = df.iloc[-1].to_dict()

    # Flexible column handling
    return {
        "fitness": last.get("best_fitness", last.get("fitness", None)),
        "min_hamming": last.get("min_hamming", last.get("minD", None)),
        "avg_hamming": last.get("avg_hamming", last.get("avgD", None)),
        "collision_pairs_radius1": last.get(
            "collision_pairs_radius1",
            last.get("coll", last.get("collision", None)),
        ),
    }


def collect_one_result(out_dir, length, size, seed):
    rows = []

    # Proposed summary
    proposed_summary = read_json(out_dir / "best_summary.json")

    rows.append({
        "length": length,
        "size": size,
        "seed": seed,
        "method": "Proposed",
        "fitness": safe_get(proposed_summary, "fitness"),
        "min_hamming": safe_get(proposed_summary, "min_hamming"),
        "avg_hamming": safe_get(proposed_summary, "avg_hamming"),
        "gc_penalty": safe_get(proposed_summary, "gc_penalty"),
        "homopolymer_penalty": safe_get(proposed_summary, "homopolymer_penalty"),
        "collision_pairs_radius1": safe_get(proposed_summary, "collision_pairs_radius1"),
        "duplicate_count": safe_get(proposed_summary, "duplicate_count"),
        "kmer_entropy": safe_get(proposed_summary, "kmer_entropy"),
        "valid_gc_fraction": safe_get(proposed_summary, "valid_gc_fraction"),
        "valid_hp_fraction": safe_get(proposed_summary, "valid_hp_fraction"),
        "runtime_seconds": safe_get(proposed_summary, "runtime_seconds"),
        "seed_pool_runtime_seconds": safe_get(proposed_summary, "seed_pool_runtime_seconds"),
        "total_runtime_seconds": safe_get(proposed_summary, "total_runtime_seconds"),
        "output_folder": out_dir.name,
    })

    # ABC-only control from run log
    abc_metrics = read_last_log_metrics(out_dir / "run_log_abc_only_control.csv")

    rows.append({
        "length": length,
        "size": size,
        "seed": seed,
        "method": "ABC-only",
        "fitness": abc_metrics.get("fitness"),
        "min_hamming": abc_metrics.get("min_hamming"),
        "avg_hamming": abc_metrics.get("avg_hamming"),
        "gc_penalty": None,
        "homopolymer_penalty": None,
        "collision_pairs_radius1": abc_metrics.get("collision_pairs_radius1"),
        "duplicate_count": None,
        "kmer_entropy": None,
        "valid_gc_fraction": None,
        "valid_hp_fraction": None,
        "runtime_seconds": None,
        "seed_pool_runtime_seconds": None,
        "total_runtime_seconds": None,
        "output_folder": out_dir.name,
    })

    return rows


def make_group_summary(all_runs_df):
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
        "seed_pool_runtime_seconds",
        "total_runtime_seconds",
    ]

    available_numeric_cols = [c for c in numeric_cols if c in all_runs_df.columns]

    summary = (
        all_runs_df
        .groupby(["length", "size", "method"])[available_numeric_cols]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    # Flatten multi-index columns
    summary.columns = [
        "_".join([str(x) for x in col if str(x) != ""])
        for col in summary.columns.values
    ]

    return summary


def make_delta_summary(all_runs_df):
    proposed = all_runs_df[all_runs_df["method"] == "Proposed"].copy()
    abc = all_runs_df[all_runs_df["method"] == "ABC-only"].copy()

    merge_cols = ["length", "size", "seed"]

    merged = proposed.merge(
        abc,
        on=merge_cols,
        suffixes=("_proposed", "_abc_only"),
        how="inner",
    )

    for metric in ["fitness", "min_hamming", "avg_hamming", "collision_pairs_radius1"]:
        p = f"{metric}_proposed"
        a = f"{metric}_abc_only"
        if p in merged.columns and a in merged.columns:
            merged[f"delta_{metric}"] = merged[p] - merged[a]

    delta_cols = [
        "delta_fitness",
        "delta_min_hamming",
        "delta_avg_hamming",
        "delta_collision_pairs_radius1",
    ]

    available_delta_cols = [c for c in delta_cols if c in merged.columns]

    delta_summary = (
        merged
        .groupby(["length", "size"])[available_delta_cols]
        .agg(["mean", "std", "min", "max"])
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

    for exp in EXPERIMENTS:
        length = exp["length"]
        size = exp["size"]

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