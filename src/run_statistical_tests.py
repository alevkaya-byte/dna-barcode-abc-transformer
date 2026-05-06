# -*- coding: utf-8 -*-
"""
Created on Wed May  6 10:29:17 2026

@author: kaya-
"""

# -*- coding: utf-8 -*-
"""
run_statistical_tests.py

Paired statistical comparison for Proposed vs ABC-only results.
Searches input CSV files recursively under BASE_DIR.
"""

from pathlib import Path
import pandas as pd
import numpy as np

try:
    from scipy.stats import wilcoxon
except ImportError as exc:
    raise ImportError(
        "scipy is required. Install it with: pip install scipy"
    ) from exc


BASE_DIR = Path(r"C:/Users/kaya-/Desktop/ABC")
OUTPUT_FILE = BASE_DIR / "paired_wilcoxon_summary.csv"

METRICS = [
    "fitness",
    "min_hamming",
    "avg_hamming",
    "collision_pairs_radius1",
]


def find_file(patterns):
    """Find first matching file recursively under BASE_DIR."""
    for pattern in patterns:
        matches = sorted(BASE_DIR.glob(pattern))
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

            rows.append({
                "experiment": experiment_label,
                "length": length,
                "size": size,
                "metric": metric,
                "n_pairs": len(deltas),
                "mean_delta": np.mean(deltas),
                "std_delta": np.std(deltas, ddof=1) if len(deltas) > 1 else np.nan,
                "median_delta": np.median(deltas),
                "min_delta": np.min(deltas),
                "max_delta": np.max(deltas),
                "proposed_wins": int(np.sum(deltas > 0)),
                "ties": int(np.sum(deltas == 0)),
                "abc_only_wins": int(np.sum(deltas < 0)),
                "wilcoxon_statistic": statistic,
                "wilcoxon_p_two_sided": p_value,
            })

    return pd.DataFrame(rows)


def main() -> None:
    critical_file = find_file([
        "**/critical10_delta_raw_FIXED.csv",
        "**/critical10_delta_raw_FIXED*.csv",
    ])

    scalability_file = find_file([
        "**/scalability_L16_N256_delta_raw.csv",
        "**/scalability_L16_N256_delta_raw*.csv",
    ])

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
        print("CSV files currently under BASE_DIR:")
        for p in BASE_DIR.glob("**/*.csv"):
            print(" -", p)
        raise FileNotFoundError(
            "No input CSV files were found. "
            "Please check whether the delta_raw CSV files exist under BASE_DIR."
        )

    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\nPaired Wilcoxon summary")
    print(summary.to_string(index=False))
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()