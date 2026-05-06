# -*- coding: utf-8 -*-
"""
run_statistical_tests.py

Paired statistical comparison for Proposed vs ABC-only results.

This script searches the repository results directory for paired delta CSV files
and applies two-sided Wilcoxon signed-rank tests on seed-level paired differences.

Input files searched:
- critical10_delta_raw_FIXED.csv
- critical10_delta_raw.csv
- scalability_L16_N256_delta_raw.csv

Output:
- results/paired_wilcoxon_summary.csv

Example command:
python src/run_statistical_tests.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except ImportError as exc:
    raise ImportError(
        "scipy is required. Install it with: pip install scipy"
    ) from exc


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "paired_wilcoxon_summary.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Metrics
# ============================================================

METRICS = [
    "fitness",
    "min_hamming",
    "avg_hamming",
    "collision_pairs_radius1",
]


# ============================================================
# Helpers
# ============================================================

def find_file(patterns):
    """Find the first matching file recursively under RESULTS_DIR."""

    for pattern in patterns:
        matches = sorted(RESULTS_DIR.glob(pattern))
        if matches:
            return matches[0]

    return None


def wilcoxon_summary(df: pd.DataFrame, experiment_label: str) -> pd.DataFrame:
    rows = []

    required_cols = {"length", "size"}

    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"{experiment_label}: file must contain columns {required_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    for (length, size), group in df.groupby(["length", "size"]):
        for metric in METRICS:
            delta_col = f"delta_{metric}"

            if delta_col not in group.columns:
                print(f"[WARNING] Missing column: {delta_col} in {experiment_label}")
                continue

            deltas = group[delta_col].dropna().astype(float).to_numpy()

            if len(deltas) == 0:
                continue

            if np.allclose(deltas, 0):
                statistic = np.nan
                p_value = np.nan
            else:
                result = wilcoxon(
                    deltas,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                )
                statistic = result.statistic
                p_value = result.pvalue

            rows.append(
                {
                    "experiment": experiment_label,
                    "length": length,
                    "size": size,
                    "metric": metric,
                    "n_pairs": len(deltas),
                    "mean_delta": np.mean(deltas),
                    "std_delta": np.std(deltas, ddof=1)
                    if len(deltas) > 1 else np.nan,
                    "median_delta": np.median(deltas),
                    "min_delta": np.min(deltas),
                    "max_delta": np.max(deltas),
                    "proposed_wins": int(np.sum(deltas > 0)),
                    "ties": int(np.sum(deltas == 0)),
                    "abc_only_wins": int(np.sum(deltas < 0)),
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p_two_sided": p_value,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main() -> None:
    critical_file = find_file(
        [
            "**/critical10_delta_raw_FIXED.csv",
            "**/critical10_delta_raw_FIXED*.csv",
            "**/critical10_delta_raw.csv",
            "**/critical10_delta_raw*.csv",
        ]
    )

    scalability_file = find_file(
        [
            "**/scalability_L16_N256_delta_raw.csv",
            "**/scalability_L16_N256_delta_raw*.csv",
        ]
    )

    print("\nDetected input files:")
    print("Critical 10-seed file:", critical_file)
    print("Scalability file:", scalability_file)

    summaries = []

    if critical_file is not None:
        critical = pd.read_csv(critical_file)
        summaries.append(wilcoxon_summary(critical, "critical_10seed"))

    if scalability_file is not None:
        scalability = pd.read_csv(scalability_file)
        summaries.append(wilcoxon_summary(scalability, "scalability_L16_N256"))

    if not summaries:
        print("\nNo matching input files were found.")
        print("CSV files currently under results/:")

        for path in RESULTS_DIR.glob("**/*.csv"):
            print(" -", path)

        raise FileNotFoundError(
            "No input CSV files were found. "
            "Please check whether the paired delta CSV files exist under results/."
        )

    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\nPaired Wilcoxon summary")
    print(summary.to_string(index=False))
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
