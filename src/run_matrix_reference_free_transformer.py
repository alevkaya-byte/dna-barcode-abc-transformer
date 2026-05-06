# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 14:40:28 2026

@author: kaya-
"""

# -*- coding: utf-8 -*-
"""
run_matrix_reference_free_transformer.py

Problem-size matrix runner for:

Reference-free Transformer-generated candidate pool + ABC
for DNA barcode library optimization.

This script assumes that barcode_abc_reference_free_transformer.py
is in the same folder.

Purpose
-------
This is a Q1-oriented generalization experiment.

It runs several barcode-design problem sizes:

    (length=12, size=64)
    (length=12, size=96)
    (length=12, size=128)
    (length=16, size=64)
    (length=16, size=96)

Each setting is tested with 3 independent seeds.

Outputs
-------
- matrix_reference_free_transformer_all_runs.csv
- matrix_reference_free_transformer_group_summary.csv
- matrix_reference_free_transformer_delta_summary.csv
- one output folder for each length-size-seed setting

Recommended Spyder run
----------------------
runfile(
    'C:/Users/kaya-/Desktop/ABC/run_matrix_reference_free_transformer.py',
    wdir='C:/Users/kaya-/Desktop/ABC'
)
"""

import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, stdev


SCRIPT_NAME = "barcode_abc_reference_free_transformer.py"

# First screening: 3 seeds.
# If this looks good, later we repeat selected settings with 5 or 10 seeds.
SEEDS = [
    20260428,
    20260429,
    20260430,
]

# Problem-size matrix.
# raw_pool and seed_pool are scaled with the target library size.
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

BASE_OUT = "matrix_reference_free_transformer"

# ABC settings.
# Keep these stable for fair comparison.
COMMON_ARGS = [
    "--iters", "20",
    "--foods", "12",
    "--random-fill-ratio", "0.40",
    "--stream-oversample", "4",
    "--pool-use-start", "0.60",
    "--pool-use-end", "0.15",
    "--run-abc-control",
]

# If False, completed folders are reused.
# This is useful if the script is interrupted; just run it again.
FORCE_RERUN = False


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_float(value):
    try:
        return float(value)
    except Exception:
        return None


def safe_stdev(values):
    if len(values) <= 1:
        return 0.0
    return stdev(values)


def write_csv(path, rows):
    if not rows:
        return

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def collect_run_rows(root, out_dir, exp, seed):
    out_path = root / out_dir
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
                "out_dir": out_dir,
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

    for r in rows:
        key = (
            r["experiment"],
            int(r["length"]),
            int(r["size"]),
            r["method"],
        )

        groups.setdefault(key, []).append(r)

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
            values = [to_float(r.get(field, "")) for r in group_rows]
            values = [v for v in values if v is not None]

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

    summary_rows.sort(key=lambda x: (x["length"], x["size"], x["method"]))

    return summary_rows


def summarize_deltas(rows):
    """
    Per experiment and seed:
        delta = proposed fitness - ABC-only fitness
    """

    proposed_name = "Reference-free Transformer-generated pool + ABC"
    control_name = "ABC-only control"

    by_exp_seed = {}

    for r in rows:
        key = (
            r["experiment"],
            int(r["length"]),
            int(r["size"]),
            int(r["seed"]),
        )

        by_exp_seed.setdefault(key, {})[r["method"]] = r

    delta_raw_rows = []

    for key, methods in by_exp_seed.items():
        experiment, length, size, seed = key

        if control_name not in methods or proposed_name not in methods:
            continue

        control = methods[control_name]
        proposed = methods[proposed_name]

        f_control = to_float(control.get("fitness", ""))
        f_proposed = to_float(proposed.get("fitness", ""))

        avg_control = to_float(control.get("avg_hamming", ""))
        avg_proposed = to_float(proposed.get("avg_hamming", ""))

        if f_control is None or f_proposed is None:
            continue

        delta_raw_rows.append(
            {
                "experiment": experiment,
                "length": length,
                "size": size,
                "seed": seed,
                "fitness_abc_only": f_control,
                "fitness_proposed": f_proposed,
                "delta_fitness": f_proposed - f_control,
                "avg_hamming_abc_only": avg_control,
                "avg_hamming_proposed": avg_proposed,
                "delta_avg_hamming": (
                    avg_proposed - avg_control
                    if avg_control is not None and avg_proposed is not None
                    else ""
                ),
                "proposed_better": int(f_proposed > f_control),
            }
        )

    grouped = {}

    for r in delta_raw_rows:
        key = (r["experiment"], r["length"], r["size"])
        grouped.setdefault(key, []).append(r)

    delta_summary = []

    for key, group_rows in grouped.items():
        experiment, length, size = key

        d_fit = [to_float(r["delta_fitness"]) for r in group_rows]
        d_fit = [v for v in d_fit if v is not None]

        d_avg = [to_float(r["delta_avg_hamming"]) for r in group_rows]
        d_avg = [v for v in d_avg if v is not None]

        wins = sum(int(r["proposed_better"]) for r in group_rows)

        delta_summary.append(
            {
                "experiment": experiment,
                "length": length,
                "size": size,
                "n_seeds": len(group_rows),
                "proposed_wins": wins,
                "abc_only_wins": len(group_rows) - wins,
                "delta_fitness_mean": mean(d_fit) if d_fit else "",
                "delta_fitness_std": safe_stdev(d_fit) if d_fit else "",
                "delta_fitness_min": min(d_fit) if d_fit else "",
                "delta_fitness_max": max(d_fit) if d_fit else "",
                "delta_avg_hamming_mean": mean(d_avg) if d_avg else "",
                "delta_avg_hamming_std": safe_stdev(d_avg) if d_avg else "",
                "delta_avg_hamming_min": min(d_avg) if d_avg else "",
                "delta_avg_hamming_max": max(d_avg) if d_avg else "",
            }
        )

    delta_summary.sort(key=lambda x: (x["length"], x["size"]))

    return delta_raw_rows, delta_summary


def main():
    root = Path.cwd()
    script_path = root / SCRIPT_NAME

    if not script_path.exists():
        raise FileNotFoundError(
            f"{SCRIPT_NAME} not found in {root}. "
            "Place this runner script in the same folder as "
            "barcode_abc_reference_free_transformer.py."
        )

    all_rows = []

    print("=" * 80)
    print("PROBLEM-SIZE MATRIX: REFERENCE-FREE TRANSFORMER + ABC")
    print("=" * 80)
    print(f"Script : {script_path}")
    print(f"Seeds  : {SEEDS}")
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

    global_t0 = time.time()

    for exp in EXPERIMENTS:
        for seed in SEEDS:
            out_dir = f"{BASE_OUT}_{exp['tag']}_seed_{seed}"

            out_path = root / out_dir
            comparison_path = out_path / "comparison_summary.csv"
            best_summary_path = out_path / "best_summary.json"

            if (
                not FORCE_RERUN
                and comparison_path.exists()
                and best_summary_path.exists()
            ):
                print("\n" + "-" * 80)
                print(f"Skipping completed run: {out_dir}")
                print("-" * 80)

                collected = collect_run_rows(root, out_dir, exp, seed)
                all_rows.extend(collected)
                continue

            cmd = [
                sys.executable,
                str(script_path),
                "--length", str(exp["length"]),
                "--size", str(exp["size"]),
                "--raw-pool-size", str(exp["raw_pool_size"]),
                "--seed-pool-size", str(exp["seed_pool_size"]),
                "--filter-sample-size", str(exp["filter_sample_size"]),
                *COMMON_ARGS,
                "--seed", str(seed),
                "--out", out_dir,
            ]

            print("\n" + "-" * 80)
            print(
                f"Running {exp['tag']} | "
                f"length={exp['length']} size={exp['size']} seed={seed}"
            )
            print(f"Output folder = {out_dir}")
            print("-" * 80)

            completed = subprocess.run(
                cmd,
                cwd=str(root),
                text=True,
                capture_output=True,
            )

            print(completed.stdout)

            if completed.returncode != 0:
                print(completed.stderr)
                raise RuntimeError(
                    f"Run failed for {exp['tag']} seed={seed}"
                )

            collected = collect_run_rows(root, out_dir, exp, seed)
            all_rows.extend(collected)

            # Save partial progress after every run.
            partial_csv = root / "matrix_reference_free_transformer_all_runs_partial.csv"
            write_csv(partial_csv, all_rows)

    all_runs_csv = root / "matrix_reference_free_transformer_all_runs.csv"
    group_summary_csv = root / "matrix_reference_free_transformer_group_summary.csv"
    delta_raw_csv = root / "matrix_reference_free_transformer_delta_raw.csv"
    delta_summary_csv = root / "matrix_reference_free_transformer_delta_summary.csv"

    write_csv(all_runs_csv, all_rows)

    group_summary = summarize_by_group(all_rows)
    write_csv(group_summary_csv, group_summary)

    delta_raw, delta_summary = summarize_deltas(all_rows)
    write_csv(delta_raw_csv, delta_raw)
    write_csv(delta_summary_csv, delta_summary)

    total_time = time.time() - global_t0

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

    for r in delta_summary:
        print(
            f"{str(r['experiment']):12s} "
            f"{int(r['n_seeds']):3d} "
            f"{int(r['proposed_wins']):6d} "
            f"{float(r['delta_fitness_mean']):16.6f} "
            f"{float(r['delta_avg_hamming_mean']):14.6f}"
        )


if __name__ == "__main__":
    main()