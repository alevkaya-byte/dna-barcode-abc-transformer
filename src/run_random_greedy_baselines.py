# -*- coding: utf-8 -*-
"""
run_random_greedy_baselines.py

Random constrained and Greedy constrained baseline experiments
for DNA barcode/index library design.

This script does not run Transformer-based candidate generation and does not run ABC.
It evaluates two simple baseline methods:

1. Random constrained:
   Randomly generates valid DNA barcodes satisfying GC and homopolymer constraints.

2. Greedy constrained:
   Builds a larger random valid candidate pool and selects barcodes greedily
   to increase separation and diversity.

Required script:
- src/barcode_abc_reference_free_transformer.py

Example command:
python src/run_random_greedy_baselines.py
"""

import csv
import sys
import time
from pathlib import Path
from statistics import mean, stdev

import numpy as np


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from barcode_abc_reference_free_transformer import (
    BarcodeConfig,
    ReferenceFreeCandidateGenerator,
    evaluate_library,
    metrics_to_dict,
    gc_fraction,
    max_homopolymer_run,
    hamming,
    kmer_counts,
    gc_penalty,
    homopolymer_penalty,
)


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
    },
    {
        "tag": "L12_N128",
        "length": 12,
        "size": 128,
    },
    {
        "tag": "L16_N64",
        "length": 16,
        "size": 64,
    },
]

GREEDY_POOL_MULTIPLIER = 20
GREEDY_POOL_MIN = 1024


# ============================================================
# Helper functions
# ============================================================

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


def kmer_novelty(seq, selected_kmers, k):
    kmers = list(kmer_counts(seq, k).keys())

    if not kmers:
        return 0.0

    novel = sum(1 for km in kmers if km not in selected_kmers)

    return novel / len(kmers)


def generate_random_library(cfg, seed):
    """
    Random constrained baseline:
    generate N unique barcodes satisfying GC and homopolymer constraints.
    """

    generator = ReferenceFreeCandidateGenerator(
        cfg=cfg,
        seed=seed,
        use_transformer=False,
    )

    library = []
    seen = set()

    max_attempts = cfg.size * 1000
    attempts = 0

    while len(library) < cfg.size and attempts < max_attempts:
        attempts += 1
        seq = generator.random_sequence()

        if seq not in seen:
            seen.add(seq)
            library.append(seq)

    if len(library) < cfg.size:
        raise RuntimeError(
            f"Could not generate enough unique random barcodes: "
            f"{len(library)}/{cfg.size}"
        )

    return library


def generate_random_candidate_pool(cfg, seed, pool_size):
    """
    Generate a random valid candidate pool for the greedy baseline.
    """

    generator = ReferenceFreeCandidateGenerator(
        cfg=cfg,
        seed=seed,
        use_transformer=False,
    )

    pool = []
    seen = set()

    max_attempts = pool_size * 200
    attempts = 0

    while len(pool) < pool_size and attempts < max_attempts:
        attempts += 1
        seq = generator.random_sequence()

        if seq not in seen:
            seen.add(seq)
            pool.append(seq)

    if len(pool) < cfg.size:
        raise RuntimeError(
            f"Candidate pool too small: {len(pool)} for target size {cfg.size}"
        )

    return pool


def greedy_select_library(candidate_pool, cfg, seed):
    """
    Greedy constrained baseline:
    select N barcodes from a valid candidate pool.

    This is a simple heuristic baseline, not ABC.
    """

    rng = np.random.default_rng(seed)
    pool = list(dict.fromkeys(candidate_pool))
    rng.shuffle(pool)

    first = max(
        pool[: min(1000, len(pool))],
        key=lambda seq: (
            -abs(gc_fraction(seq) - cfg.gc_target)
            - homopolymer_penalty(seq, cfg)
        ),
    )

    library = [first]
    selected_kmers = set(kmer_counts(first, cfg.kmer_k).keys())
    remaining = [seq for seq in pool if seq != first]

    while len(library) < cfg.size and remaining:
        best = None
        best_score = -1e9

        sample_size = min(len(remaining), 600)
        sample_indices = rng.choice(
            len(remaining),
            size=sample_size,
            replace=False,
        )

        for idx in sample_indices:
            seq = remaining[int(idx)]

            nearest = min(hamming(seq, other) for other in library)
            avgd = float(np.mean([hamming(seq, other) for other in library]))
            novelty = kmer_novelty(seq, selected_kmers, cfg.kmer_k)

            score = (
                4.0 * nearest / cfg.length
                + 1.0 * avgd / cfg.length
                + 0.20 * novelty
                - 1.0 * gc_penalty(seq, cfg)
                - 1.0 * homopolymer_penalty(seq, cfg)
            )

            if score > best_score:
                best = seq
                best_score = score

        if best is None:
            break

        library.append(best)
        selected_kmers.update(kmer_counts(best, cfg.kmer_k).keys())
        remaining.remove(best)

    if len(library) < cfg.size:
        raise RuntimeError(
            f"Greedy library incomplete: {len(library)}/{cfg.size}"
        )

    return library


def run_one_experiment(exp, seed):
    cfg = BarcodeConfig(
        length=exp["length"],
        size=exp["size"],
        gc_min=0.40,
        gc_max=0.60,
        gc_target=0.50,
        homopolymer_max=3,
        error_radius=1,
        kmer_k=3,
    )

    rows = []

    # ---------------------------------------------------------------------
    # Random constrained baseline
    # ---------------------------------------------------------------------
    start_time = time.time()

    random_library = generate_random_library(
        cfg=cfg,
        seed=seed + 1000,
    )

    random_runtime = time.time() - start_time
    random_metrics = evaluate_library(random_library, cfg)

    rows.append(
        {
            "experiment": exp["tag"],
            "length": cfg.length,
            "size": cfg.size,
            "seed": seed,
            "method": "Random constrained",
            **metrics_to_dict(random_metrics),
            "runtime_seconds": random_runtime,
            "candidate_pool_size": "",
        }
    )

    # ---------------------------------------------------------------------
    # Greedy constrained baseline
    # ---------------------------------------------------------------------
    greedy_pool_size = max(GREEDY_POOL_MIN, GREEDY_POOL_MULTIPLIER * cfg.size)

    start_time = time.time()

    pool = generate_random_candidate_pool(
        cfg=cfg,
        seed=seed + 2000,
        pool_size=greedy_pool_size,
    )

    greedy_library = greedy_select_library(
        candidate_pool=pool,
        cfg=cfg,
        seed=seed + 3000,
    )

    greedy_runtime = time.time() - start_time
    greedy_metrics = evaluate_library(greedy_library, cfg)

    rows.append(
        {
            "experiment": exp["tag"],
            "length": cfg.length,
            "size": cfg.size,
            "seed": seed,
            "method": "Greedy constrained",
            **metrics_to_dict(greedy_metrics),
            "runtime_seconds": greedy_runtime,
            "candidate_pool_size": greedy_pool_size,
        }
    )

    return rows


def summarize_group(rows):
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
    ]

    groups = {}

    for row in rows:
        key = (
            row["experiment"],
            row["length"],
            row["size"],
            row["method"],
        )

        groups.setdefault(key, []).append(row)

    summary_rows = []

    for key, group_rows in groups.items():
        experiment, length, size, method = key

        output_row = {
            "experiment": experiment,
            "length": length,
            "size": size,
            "method": method,
            "n_seeds": len(group_rows),
        }

        for field in fields:
            values = [float(row[field]) for row in group_rows]

            output_row[f"{field}_mean"] = mean(values)
            output_row[f"{field}_std"] = safe_stdev(values)
            output_row[f"{field}_min"] = min(values)
            output_row[f"{field}_max"] = max(values)

        summary_rows.append(output_row)

    summary_rows.sort(key=lambda item: (item["length"], item["size"], item["method"]))

    return summary_rows


# ============================================================
# Main
# ============================================================

def main():
    all_rows = []

    print("=" * 80)
    print("RANDOM + GREEDY BASELINE EXPERIMENTS")
    print("=" * 80)
    print(f"Seeds: {SEEDS}")
    print("Experiments:")

    for exp in EXPERIMENTS:
        print(
            f"  {exp['tag']}: length={exp['length']}, size={exp['size']}"
        )

    print("=" * 80)

    for exp in EXPERIMENTS:
        for seed in SEEDS:
            print("\n" + "-" * 80)
            print(
                f"Running baselines | {exp['tag']} | "
                f"length={exp['length']} size={exp['size']} seed={seed}"
            )
            print("-" * 80)

            rows = run_one_experiment(exp, seed)
            all_rows.extend(rows)

            for row in rows:
                print(
                    f"{row['method']:22s} "
                    f"fitness={row['fitness']:.6f} "
                    f"minD={row['min_hamming']} "
                    f"avgD={row['avg_hamming']:.4f} "
                    f"coll={row['collision_pairs_radius1']} "
                    f"dup={row['duplicate_count']} "
                    f"time={row['runtime_seconds']:.2f}s"
                )

            partial_path = RESULTS_DIR / "random_greedy_baselines_all_runs_partial.csv"
            write_csv(partial_path, all_rows)

    all_runs_path = RESULTS_DIR / "random_greedy_baselines_all_runs.csv"
    summary_path = RESULTS_DIR / "random_greedy_baselines_group_summary.csv"

    write_csv(all_runs_path, all_rows)

    summary_rows = summarize_group(all_rows)
    write_csv(summary_path, summary_rows)

    print("\n" + "=" * 80)
    print("BASELINE RUNS COMPLETED")
    print("=" * 80)
    print(f"Saved: {all_runs_path}")
    print(f"Saved: {summary_path}")

    print("\nSummary")
    print("-" * 80)

    for row in summary_rows:
        print(
            f"{row['experiment']:10s} "
            f"{row['method']:22s} "
            f"fitness={row['fitness_mean']:.6f}±{row['fitness_std']:.6f} "
            f"minD={row['min_hamming_mean']:.2f} "
            f"avgD={row['avg_hamming_mean']:.4f} "
            f"coll={row['collision_pairs_radius1_mean']:.2f} "
            f"dup={row['duplicate_count_mean']:.2f}"
        )


if __name__ == "__main__":
    main()
