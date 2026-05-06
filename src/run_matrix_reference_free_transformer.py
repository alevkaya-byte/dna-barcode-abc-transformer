# -*- coding: utf-8 -*-
"""
run_matrix_reference_free_transformer.py

Problem-size matrix runner for the DNA barcode/index library optimization study.

This script runs several barcode-design problem sizes using:

- Proposed reference-free Transformer candidate generation + ABC
- ABC-only control

Problem-size matrix:
- L=12, N=64
- L=12, N=96
- L=12, N=128
- L=16, N=64
- L=16, N=96

Each setting is evaluated with 3 independent seeds.

Required script:
- src/barcode_abc_reference_free_transformer.py

Example command:
python src/run_matrix_reference_free_transformer.py
"""

import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, stdev


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
RESULTS_DIR = BASE_DIR / "results"
RAW_DIR = RESULTS_DIR / "raw_runs" / "problem_size_matrix"

MAIN_SCRIPT = SRC_DIR / "barcode_abc_reference_free_transformer.py"

RAW_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Experiment settings
# ============================================================

SEEDS = [
    20260428,
    20260429,
    20260430,
]

EXPERIMENTS = [
    {
        "tag": "L12_N64",
        "length": 12,
        "size": 64,
        "raw_pool_size": 512,
        "seed_pool_size": 256,
        "filter_sample_size": 80,
    },
    {
        "tag": "L12_N96",
        "length": 12,
        "size": 96,
        "raw_pool_size": 768,
        "seed_pool_size": 384,
        "filter_sample_size": 80,
    },
    {
        "tag": "L12_N128",
        "length": 12,
        "size": 128,
        "raw_pool_size": 1024,
        "seed_pool_size": 512,
        "filter_sample_size": 80,
    },
    {
        "tag": "L16_N64",
        "length": 16,
        "size": 64,
        "raw_pool_size": 512,
        "seed_pool_size": 256,
        "filter_sample_size": 80,
    },
    {
        "tag": "L16_N96",
        "length": 16,
        "size": 96,
        "raw_pool_size": 768,
        "seed_pool_size": 384,
        "filter_sample_size": 80,
    },
]

COMMON_ARGS = [
    "--iters", "20",
    "--foods", "12",
    "--random-fill-ratio", "0.40",
    "--stream-oversample", "4",
    "--pool-use-start", "0.60",
    "--pool-use-end", "0.15",
    "--run-abc-control",
]

BASE_OUT = "matrix_reference_free_transformer"

# If False, completed folders are reused.
FORCE_RERUN = False


# ============================================================
# Helper functions
# ============================================================

def read_csv_rows(path: Path):
    with open(path, "r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def to_float(value):
    try:
        return float(value)
    except Exception:
        return None


def safe_stdev(values):
    if len(values) <= 1:
        return 0.0
    return stdev(values)


def write_csv(path: Path, rows):
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def collect_run_rows(out_path: Path, exp: dict, seed: int):
    comparison_path = out_path / "comparison_summary.csv"
    best_summary_path = out_path / "best_summary.json"

    if not comparison_path.exists():
        raise FileNotFoundError(f"Missing comparison file: {comparison_path}")

    if not best_summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {best_summary_path}")

    comparison_rows = read_csv_rows(comparison_path)
    best_summary = read_json(best_summary_path)

    pool_summary = best_summary.get("candidate_pool_summary", {}) or {}

    collected = []

    for row in comparison_rows:
        method = row.get("method", "")

        collected.append(
            {
                "experiment": exp["tag"],
                "length": exp["length"],
                "size": exp["size"],
                "seed": seed,
                "method": method,
                "fitness": row.get("fitness", ""),
                "min_hamming": row.get("min_hamming", ""),
                "avg_hamming": row.get("avg_hamming", ""),
                "gc_penalty": row.get("gc_penalty", ""),
                "homopolymer_penalty": row.get("homopolymer_penalty", ""),
                "collision_pairs_radius1": row.get("collision_pairs_radius1", ""),
                "duplicate_count": row.get("duplicate_count", ""),
                "kmer_entropy": row.get("kmer_entropy", ""),
                "valid_gc_fraction": row.get("valid_gc_fraction", ""),
                "valid_hp_fraction": row.get("valid_hp_fraction", ""),
                "runtime_seconds": row.get("runtime_seconds", ""),
                "candidate_pool_runtime_seconds": row.get(
                    "candidate_pool_runtime_seconds", ""
                ),
                "total_runtime_seconds": row.get("total_runtime_seconds", ""),
                "raw_pool_size": exp["raw_pool_size"],
                "seed_pool_size": exp["seed_pool_size"],
                "filter_sample_size": exp["filter_sample_size"],
                "raw_generated_unique": pool_summary.get("raw_generated_unique", ""),
                "selected_pool_size": pool_summary.get("selected_pool_size", ""),
                "transformer_candidates": pool_summary.get(
                    "transformer_candidates", ""
                ),
                "random_fill_candidates": pool_summary.get(
                    "random_fill_candidates", ""
                ),
                "transformer_stream_bases": pool_summary.get(
                    "transformer_stream_bases", ""
                ),
                "raw_kmer_entropy": pool_summary.get("raw_kmer_entropy", ""),
                "selected_kmer_entropy": pool_summary.get(
                    "selected_kmer_entropy", ""
                ),
                "selected_valid_gc_fraction": pool_summary.get(
                    "valid_gc_fraction_selected", ""
                ),
                "selected_valid_hp_fraction": pool_summary.get(
                    "valid_hp_fraction_selected", ""
                ),
                "out_dir": str(out_path.relative_to(BASE_DIR)),
            }
        )

    return collected


def summarize_by_group(rows):
    fields = [
        "fitness",
        "min_hamming",
        "avg_hamming",
        "kmer_entropy",
        "collision_pairs_radius1",
        "duplicate_count",
        "valid_gc_fraction",
        "valid_hp_fraction",
        "runtime_seconds",
        "candidate_pool_runtime_seconds",
        "total_runtime_seconds",
    ]

    groups = {}

    for row in rows:
        key = (
            row["experiment"],
            int(row["length"]),
            int(row["size"]),
            row["method"],
        )
        groups.setdefault(key, []).append(row)

    summary_rows = []

    for key, group_rows in groups.items():
        experiment, length, size, method = key

        out = {
            "experiment": experiment,
            "length": length,
            "size": size,
            "method": method,
            "n_seeds": len(group_rows),
        }

        for field in fields:
            values = [to_float(row.get(field, "")) for row in group_rows]
            values = [value for value in values if value is not None]

            if not values:
                out[f"{field}_mean"] = ""
                out[f"{field}_std"] = ""
                out[f"{field}_min"] = ""
                out[f"{field}_max"] = ""
            else:
                out[f"{field}_mean"] = mean(values)
                out[f"{field}_std"] = safe_stdev(values)
                out[f"{field}_min"] = min(values)
                out[f"{field}_max"] = max(values)

        summary_rows.append(out)

    summary_rows.sort(key=lambda item: (item["length"], item["size"], item["method"]))

    return summary_rows


def summarize_deltas(rows):
    proposed_name = "Reference-free Transformer-generated pool + ABC"
    control_name = "ABC-only control"

    by_exp_seed = {}

    for row in rows:
        key = (
            row["experiment"],
            int(row["length"]),
            int(row["size"]),
            int(row["seed"]),
        )
        by_exp_seed.setdefault(key, {})[row["method"]] = row

    delta_raw_rows = []

    for key, methods in by_exp_seed.items():
        experiment, length, size, seed = key

        if control_name not in methods or proposed_name not in methods:
            continue

        control = methods[control_name]
        proposed = methods[proposed_name]

        fitness_control = to_float(control.get("fitness", ""))
        fitness_proposed = to_float(proposed.get("fitness", ""))

        avg_control = to_float(control.get("avg_hamming", ""))
        avg_proposed = to_float(proposed.get("avg_hamming", ""))

        min_control = to_float(control.get("min_hamming", ""))
        min_proposed = to_float(proposed.get("min_hamming", ""))

        collision_control = to_float(control.get("collision_pairs_radius1", ""))
        collision_proposed = to_float(proposed.get("collision_pairs_radius1", ""))

        if fitness_control is None or fitness_proposed is None:
            continue

        delta_raw_rows.append(
            {
                "experiment": experiment,
                "length": length,
                "size": size,
                "seed": seed,
                "fitness_abc_only": fitness_control,
                "fitness_proposed": fitness_proposed,
                "delta_fitness": fitness_proposed - fitness_control,
                "min_hamming_abc_only": min_control,
                "min_hamming_proposed": min_proposed,
                "delta_min_hamming": (
                    min_proposed - min_control
                    if min_control is not None and min_proposed is not None
                    else ""
                ),
                "avg_hamming_abc_only": avg_control,
                "avg_hamming_proposed": avg_proposed,
                "delta_avg_hamming": (
                    avg_proposed - avg_control
                    if avg_control is not None and avg_proposed is not None
                    else ""
                ),
                "collision_pairs_abc_only": collision_control,
                "collision_pairs_proposed": collision_proposed,
                "delta_collision_pairs_radius1": (
                    collision_proposed - collision_control
                    if collision_control is not None and collision_proposed is not None
                    else ""
                ),
                "proposed_better": int(fitness_proposed > fitness_control),
            }
        )

    grouped = {}

    for row in delta_raw_rows:
        key = (row["experiment"], row["length"], row["size"])
        grouped.setdefault(key, []).append(row)

    delta_summary = []

    for key, group_rows in grouped.items():
        experiment, length, size = key

        delta_fitness = [to_float(row["delta_fitness"]) for row in group_rows]
        delta_fitness = [value for value in delta_fitness if value is not None]

        delta_min_hamming = [
            to_float(row["delta_min_hamming"]) for row in group_rows
        ]
        delta_min_hamming = [
            value for value in delta_min_hamming if value is not None
        ]

        delta_avg_hamming = [
            to_float(row["delta_avg_hamming"]) for row in group_rows
        ]
        delta_avg_hamming = [
            value for value in delta_avg_hamming if value is not None
        ]

        delta_collision = [
            to_float(row["delta_collision_pairs_radius1"]) for row in group_rows
        ]
        delta_collision = [
            value for value in delta_collision if value is not None
        ]

        wins = sum(int(row["proposed_better"]) for row in group_rows)

        delta_summary.append(
            {
                "experiment": experiment,
                "length": length,
                "size": size,
                "n_seeds": len(group_rows),
                "proposed_wins": wins,
                "abc_only_wins": len(group_rows) - wins,
                "delta_fitness_mean": mean(delta_fitness) if delta_fitness else "",
                "delta_fitness_std": (
                    safe_stdev(delta_fitness) if delta_fitness else ""
                ),
                "delta_fitness_min": min(delta_fitness) if delta_fitness else "",
                "delta_fitness_max": max(delta_fitness) if delta_fitness else "",
                "delta_min_hamming_mean": (
                    mean(delta_min_hamming) if delta_min_hamming else ""
                ),
                "delta_min_hamming_std": (
                    safe_stdev(delta_min_hamming) if delta_min_hamming else ""
                ),
                "delta_avg_hamming_mean": (
                    mean(delta_avg_hamming) if delta_avg_hamming else ""
                ),
                "delta_avg_hamming_std": (
                    safe_stdev(delta_avg_hamming) if delta_avg_hamming else ""
                ),
                "delta_collision_pairs_radius1_mean": (
                    mean(delta_collision) if delta_collision else ""
                ),
            }
        )

    delta_summary.sort(key=lambda item: (item["length"], item["size"]))

    return delta_raw_rows, delta_summary


# ============================================================
# Main
# ============================================================

def main():
    if not MAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Main script not found: {MAIN_SCRIPT}")

    all_rows = []

    print("=" * 80)
    print("PROBLEM-SIZE MATRIX: REFERENCE-FREE TRANSFORMER + ABC")
    print("=" * 80)
    print(f"Main script: {MAIN_SCRIPT}")
    print(f"Raw output directory: {RAW_DIR}")
    print(f"Summary directory: {RESULTS_DIR}")
    print(f"Seeds: {SEEDS}")
    print("Experiments:")

    for exp in EXPERIMENTS:
        print(
            f"  {exp['tag']}: "
            f"length={exp['length']}, size={exp['size']}, "
            f"raw_pool={exp['raw_pool_size']}, "
            f"seed_pool={exp['seed_pool_size']}, "
            f"filter_sample={exp['filter_sample_size']}"
        )

    print("=" * 80)

    global_start = time.time()

    for exp in EXPERIMENTS:
        for seed in SEEDS:
            out_name = f"{BASE_OUT}_{exp['tag']}_seed_{seed}"
            out_path = RAW_DIR / out_name

            comparison_path = out_path / "comparison_summary.csv"
            best_summary_path = out_path / "best_summary.json"

            if (
                not FORCE_RERUN
                and comparison_path.exists()
                and best_summary_path.exists()
            ):
                print("\n" + "-" * 80)
                print(f"Skipping completed run: {out_name}")
                print("-" * 80)

                collected = collect_run_rows(out_path, exp, seed)
                all_rows.extend(collected)
                continue

            cmd = [
                sys.executable,
                str(MAIN_SCRIPT),
                "--length", str(exp["length"]),
                "--size", str(exp["size"]),
                "--raw-pool-size", str(exp["raw_pool_size"]),
                "--seed-pool-size", str(exp["seed_pool_size"]),
                "--filter-sample-size", str(exp["filter_sample_size"]),
                *COMMON_ARGS,
                "--seed", str(seed),
                "--out", str(out_path),
            ]

            print("\n" + "-" * 80)
            print(
                f"Running {exp['tag']} | "
                f"length={exp['length']} size={exp['size']} seed={seed}"
            )
            print(f"Output folder: {out_path}")
            print("-" * 80)

            completed = subprocess.run(
                cmd,
                cwd=str(BASE_DIR),
                text=True,
                capture_output=True,
            )

            print(completed.stdout)

            if completed.returncode != 0:
                print(completed.stderr)
                raise RuntimeError(
                    f"Run failed for {exp['tag']} seed={seed}"
                )

            collected = collect_run_rows(out_path, exp, seed)
            all_rows.extend(collected)

            partial_csv = RESULTS_DIR / "matrix_reference_free_transformer_all_runs_partial.csv"
            write_csv(partial_csv, all_rows)

    all_runs_csv = RESULTS_DIR / "matrix_reference_free_transformer_all_runs.csv"
    group_summary_csv = RESULTS_DIR / "matrix_reference_free_transformer_group_summary.csv"
    delta_raw_csv = RESULTS_DIR / "matrix_reference_free_transformer_delta_raw.csv"
    delta_summary_csv = RESULTS_DIR / "matrix_reference_free_transformer_delta_summary.csv"

    write_csv(all_runs_csv, all_rows)

    group_summary = summarize_by_group(all_rows)
    write_csv(group_summary_csv, group_summary)

    delta_raw, delta_summary = summarize_deltas(all_rows)
    write_csv(delta_raw_csv, delta_raw)
    write_csv(delta_summary_csv, delta_summary)

    total_time = time.time() - global_start

    print("\n" + "=" * 80)
    print("MATRIX RUNS COMPLETED")
    print("=" * 80)
    print(f"Saved: {all_runs_csv}")
    print(f"Saved: {group_summary_csv}")
    print(f"Saved: {delta_raw_csv}")
    print(f"Saved: {delta_summary_csv}")
    print(f"Total elapsed seconds: {total_time:.2f}")

    print("\nDelta summary: proposed - ABC-only")
    print("-" * 80)
    print(
        f"{'experiment':12s} {'n':>3s} {'wins':>6s} "
        f"{'d_fitness_mean':>16s} {'d_avgD_mean':>14s}"
    )

    for row in delta_summary:
        print(
            f"{str(row['experiment']):12s} "
            f"{int(row['n_seeds']):3d} "
            f"{int(row['proposed_wins']):6d} "
            f"{float(row['delta_fitness_mean']):16.6f} "
            f"{float(row['delta_avg_hamming_mean']):14.6f}"
        )


if __name__ == "__main__":
    main()



Also mention if you want to be extra robust, add a line in README "run_matrix..." writes raw outputs to `results/raw_runs/problem_size_matrix`. 
Let's final. 
