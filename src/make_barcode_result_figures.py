# -*- coding: utf-8 -*-
"""
make_barcode_result_figures.py

Article-ready figure generator for DNA barcode/index library optimization results.

Required files inside RESULT_DIR:
- run_log.csv
- run_log_abc_only_control.csv
- best_library.csv
- pairwise_distances.csv

Example command:
python src/make_barcode_result_figures.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CHANGE THIS FOLDER NAME IF NEEDED
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RESULT_DIR = BASE_DIR / "results" / "matrix_reference_free_transformer_L12_N64_seed_20260428"
OUTPUT_DIR = BASE_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Global plot settings
# ============================================================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["figure.dpi"] = 300


def save_figure(fig, name):
    png_path = OUTPUT_DIR / f"{name}.png"
    svg_path = OUTPUT_DIR / f"{name}.svg"

    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")


def read_required_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


# ============================================================
# Figure 2: Convergence curve
# ============================================================

def figure_convergence(result_dir):
    proposed_log = read_required_csv(result_dir / "run_log.csv")
    control_log = read_required_csv(result_dir / "run_log_abc_only_control.csv")

    # column names safety
    if "iteration" not in proposed_log.columns or "best_fitness" not in proposed_log.columns:
        raise ValueError("run_log.csv must contain 'iteration' and 'best_fitness' columns.")
    if "iteration" not in control_log.columns or "best_fitness" not in control_log.columns:
        raise ValueError("run_log_abc_only_control.csv must contain 'iteration' and 'best_fitness' columns.")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    ax.plot(
        control_log["iteration"],
        control_log["best_fitness"],
        marker="o",
        linewidth=1.6,
        markersize=4,
        label="ABC-only control",
    )

    ax.plot(
        proposed_log["iteration"],
        proposed_log["best_fitness"],
        marker="s",
        linewidth=1.6,
        markersize=4,
        label="Reference-free Transformer + ABC",
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best fitness")
    ax.set_xticks(np.arange(1, int(max(proposed_log["iteration"].max(), control_log["iteration"].max())) + 1, 1))
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend(frameon=False, loc="lower right")

    save_figure(fig, "Figure2_convergence")


# ============================================================
# Figure 3: Pairwise Hamming histogram
# ============================================================

def figure_hamming_histogram(result_dir):
    matrix_df = read_required_csv(result_dir / "pairwise_distances.csv")

    if matrix_df.shape[1] < 2:
        raise ValueError("pairwise_distances.csv does not have expected matrix format.")

    matrix = matrix_df.iloc[:, 1:].to_numpy(dtype=float)
    upper = matrix[np.triu_indices_from(matrix, k=1)]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    min_val = int(np.floor(np.min(upper)))
    max_val = int(np.ceil(np.max(upper)))
    bins = np.arange(min_val, max_val + 2) - 0.5

    ax.hist(
        upper,
        bins=bins,
        edgecolor="black",
        linewidth=0.7,
    )

    ax.set_xlabel("Pairwise Hamming distance")
    ax.set_ylabel("Number of barcode pairs")
    ax.set_xticks(np.arange(min_val, max_val + 1, 1))
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.4)

    save_figure(fig, "Figure3_hamming_histogram")


# ============================================================
# Figure 4: Pairwise Hamming heatmap
# ============================================================

def figure_hamming_heatmap(result_dir):
    matrix_df = read_required_csv(result_dir / "pairwise_distances.csv")

    if matrix_df.shape[1] < 2:
        raise ValueError("pairwise_distances.csv does not have expected matrix format.")

    matrix = matrix_df.iloc[:, 1:].to_numpy(dtype=float)

    # mask diagonal so that zero self-distances do not dominate the color scale
    masked_matrix = matrix.copy().astype(float)
    np.fill_diagonal(masked_matrix, np.nan)

    finite_vals = masked_matrix[np.isfinite(masked_matrix)]
    vmin = np.min(finite_vals)
    vmax = np.max(finite_vals)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")  # diagonal shown as white

    fig, ax = plt.subplots(figsize=(6.0, 5.2))

    im = ax.imshow(
        masked_matrix,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Hamming distance")

    ax.set_xlabel("Barcode index")
    ax.set_ylabel("Barcode index")

    n = masked_matrix.shape[0]
    if n <= 64:
        ticks = np.arange(0, n, 8)
    else:
        step = max(1, n // 8)
        ticks = np.arange(0, n, step)

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks + 1)
    ax.set_yticklabels(ticks + 1)

    save_figure(fig, "Figure4_hamming_heatmap")


# ============================================================
# Figure 5: GC + Homopolymer combined figure
# ============================================================

def figure_constraints_combined(result_dir):
    lib_df = read_required_csv(result_dir / "best_library.csv")

    required_cols = {"gc_fraction", "max_homopolymer"}
    if not required_cols.issubset(set(lib_df.columns)):
        raise ValueError("best_library.csv must include 'gc_fraction' and 'max_homopolymer' columns.")

    gc_values = lib_df["gc_fraction"].to_numpy(dtype=float)
    hp_values = lib_df["max_homopolymer"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))

    # ---------------- Left panel: GC distribution ----------------
    ax = axes[0]

    unique_gc = np.sort(np.unique(gc_values))
    if len(unique_gc) > 1:
        diffs = np.diff(unique_gc)
        step = np.min(diffs)
    else:
        step = 0.05

    bins = np.concatenate(([unique_gc[0] - step / 2], unique_gc + step / 2))

    ax.hist(
        gc_values,
        bins=bins,
        edgecolor="black",
        linewidth=0.7,
    )

    ax.axvline(0.40, linestyle="--", linewidth=1.2)
    ax.axvline(0.60, linestyle="--", linewidth=1.2)

    ax.set_xlabel("GC fraction")
    ax.set_ylabel("Number of barcodes")
    ax.set_xlim(0.35, 0.65)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.4)
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, ha="left", va="top")

    # ---------------- Right panel: Homopolymer distribution ----------------
    ax = axes[1]

    values, counts = np.unique(hp_values, return_counts=True)

    ax.bar(
        values,
        counts,
        width=0.8,
        edgecolor="black",
        linewidth=0.7,
    )

    ax.axvline(3, linestyle="--", linewidth=1.2)

    ax.set_xlabel("Maximum homopolymer run")
    ax.set_ylabel("Number of barcodes")
    ax.set_xticks(values)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.4)
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, ha="left", va="top")

    save_figure(fig, "Figure5_constraint_distributions")


def main():
    print("=" * 80)
    print("Generating barcode result figures")
    print("=" * 80)
    print(f"Result folder: {RESULT_DIR.resolve()}")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")

    figure_convergence(RESULT_DIR)
    figure_hamming_histogram(RESULT_DIR)
    figure_hamming_heatmap(RESULT_DIR)
    figure_constraints_combined(RESULT_DIR)

    print("\nAll figures generated successfully.")


if __name__ == "__main__":
    main()
