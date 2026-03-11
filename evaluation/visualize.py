"""
Visualization tools for misalignment benchmark results.

Generates:
  - Degradation curve plots (BLEU-4 vs severity, one line per condition type)
  - 3x3 Heatmaps (sign(delta_s) x sign(delta_e) color-coded by metric)
  - Model comparison overlay plots
  - Failure distribution stacked bar charts
"""

import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns


# Consistent styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

# Color palette for condition types
CONDITION_COLORS = {
    "head_trunc": "#e74c3c",
    "tail_trunc": "#3498db",
    "head_contam": "#e67e22",
    "tail_contam": "#2ecc71",
    "head_trunc+tail_trunc": "#9b59b6",
    "head_trunc+tail_contam": "#f39c12",
    "head_contam+tail_trunc": "#1abc9c",
    "head_contam+tail_contam": "#34495e",
}


def plot_degradation_curves(
    results_csv: str,
    output_dir: str,
    metric: str = "bleu4",
    model_name: Optional[str] = None,
) -> str:
    """Plot BLEU-4 (or other metric) vs severity for each condition type.

    Args:
        results_csv: Path to the benchmark results CSV.
        output_dir: Directory to save the plot.
        metric: Metric to plot (default: bleu4).
        model_name: Optional model name for the title.

    Returns:
        Path to the saved plot file.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(results_csv)

    # Get clean baseline score
    clean_row = df[df["condition"] == "clean"]
    clean_score = clean_row[metric].values[0] if len(clean_row) > 0 else None

    # Filter to misaligned conditions
    misaligned = df[df["condition"] != "clean"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each condition type
    for cond_name in misaligned["condition"].unique():
        cond_data = misaligned[misaligned["condition"] == cond_name].sort_values(
            "severity"
        )
        color = CONDITION_COLORS.get(cond_name, "#7f8c8d")
        ax.plot(
            cond_data["severity"] * 100,
            cond_data[metric],
            marker="o",
            linewidth=2,
            markersize=5,
            label=cond_name.replace("_", " ").replace("+", " + "),
            color=color,
        )

    # Plot clean baseline as horizontal line
    if clean_score is not None:
        ax.axhline(
            y=clean_score,
            color="green",
            linestyle="--",
            linewidth=1.5,
            label=f"Clean baseline ({clean_score:.1f})",
            alpha=0.7,
        )

    ax.set_xlabel("Severity (%)")
    ax.set_ylabel(metric.upper())
    title = f"Translation Quality vs Misalignment Severity"
    if model_name:
        title += f" — {model_name}"
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"degradation_curves_{metric}.png")
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    return plot_path


def plot_heatmap(
    results_csv: str,
    output_dir: str,
    severity: float = 0.20,
    metric: str = "bleu4",
    model_name: Optional[str] = None,
) -> str:
    """Plot a 3x3 heatmap of metric scores at a fixed severity.

    Rows: sign(delta_s) in {-1 (contam), 0 (clean), +1 (trunc)}
    Cols: sign(delta_e) in {-1 (trunc), 0 (clean), +1 (contam)}

    Args:
        results_csv: Path to the benchmark results CSV.
        output_dir: Directory to save the plot.
        severity: Severity level to visualize.
        metric: Metric to show in cells.
        model_name: Optional model name for the title.

    Returns:
        Path to the saved plot file.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(results_csv)

    # Build 3x3 grid
    # Map condition names to (row, col) positions
    condition_to_pos = {
        "head_contam+tail_trunc": (0, 0),
        "head_contam": (0, 1),
        "head_contam+tail_contam": (0, 2),
        "tail_trunc": (1, 0),
        "clean": (1, 1),
        "tail_contam": (1, 2),
        "head_trunc+tail_trunc": (2, 0),
        "head_trunc": (2, 1),
        "head_trunc+tail_contam": (2, 2),
    }

    grid = np.full((3, 3), np.nan)

    # Fill clean baseline (center)
    clean_row = df[df["condition"] == "clean"]
    if len(clean_row) > 0:
        grid[1, 1] = clean_row[metric].values[0]

    # Fill misaligned conditions at the specified severity
    target_df = df[
        (df["severity"] == severity) | (df["condition"] == "clean")
    ]

    for _, row in target_df.iterrows():
        cond = row["condition"]
        if cond in condition_to_pos:
            r, c = condition_to_pos[cond]
            grid[r, c] = row[metric]

    fig, ax = plt.subplots(figsize=(8, 6))

    row_labels = ["δs < 0\n(head contam)", "δs = 0\n(clean head)", "δs > 0\n(head trunc)"]
    col_labels = ["δe < 0\n(tail trunc)", "δe = 0\n(clean tail)", "δe > 0\n(tail contam)"]

    sns.heatmap(
        grid,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        xticklabels=col_labels,
        yticklabels=row_labels,
        linewidths=2,
        linecolor="white",
        cbar_kws={"label": metric.upper()},
        ax=ax,
    )

    title = f"Misalignment Impact at {severity:.0%} Severity"
    if model_name:
        title += f" — {model_name}"
    ax.set_title(title, pad=15)

    plt.tight_layout()
    plot_path = os.path.join(
        output_dir, f"heatmap_{metric}_sev{int(severity*100)}.png"
    )
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    return plot_path


def plot_model_comparison(
    results_csvs: Dict[str, str],
    output_dir: str,
    metric: str = "bleu4",
    conditions_to_plot: Optional[List[str]] = None,
) -> str:
    """Overlay degradation curves for multiple models.

    Args:
        results_csvs: Dict mapping model_name to CSV path.
        output_dir: Directory to save the plot.
        metric: Metric to plot.
        conditions_to_plot: Optional subset of conditions to include.

    Returns:
        Path to the saved plot file.
    """
    os.makedirs(output_dir, exist_ok=True)

    if conditions_to_plot is None:
        conditions_to_plot = ["head_trunc", "tail_trunc", "head_contam", "tail_contam"]

    fig, axes = plt.subplots(1, len(conditions_to_plot), figsize=(5 * len(conditions_to_plot), 5))
    if len(conditions_to_plot) == 1:
        axes = [axes]

    model_colors = plt.cm.tab10(np.linspace(0, 1, len(results_csvs)))

    for ax, cond_name in zip(axes, conditions_to_plot):
        for (model_name, csv_path), color in zip(results_csvs.items(), model_colors):
            df = pd.read_csv(csv_path)
            cond_data = df[df["condition"] == cond_name].sort_values("severity")
            if len(cond_data) > 0:
                ax.plot(
                    cond_data["severity"] * 100,
                    cond_data[metric],
                    marker="o",
                    linewidth=2,
                    label=model_name,
                    color=color,
                )

        ax.set_xlabel("Severity (%)")
        ax.set_ylabel(metric.upper())
        ax.set_title(cond_name.replace("_", " ").title())
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Model Comparison: {metric.upper()} vs Severity", fontsize=14, y=1.02)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"model_comparison_{metric}.png")
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    return plot_path


def plot_failure_distribution(
    failure_csv: str,
    output_dir: str,
    model_name: Optional[str] = None,
) -> str:
    """Plot stacked bar chart of failure type distribution per condition.

    Args:
        failure_csv: Path to failure analysis CSV with columns:
                     condition, severity, hallucination, under_generation,
                     incoherent, acceptable (each as counts or fractions).
        output_dir: Directory to save the plot.
        model_name: Optional model name for the title.

    Returns:
        Path to the saved plot file.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(failure_csv)

    failure_types = ["hallucination", "under_generation", "incoherent", "acceptable"]
    colors = ["#e74c3c", "#f39c12", "#9b59b6", "#2ecc71"]

    # Create condition labels
    df["label"] = df.apply(
        lambda r: f"{r['condition']}\n({r['severity']:.0%})" if r["severity"] > 0 else "clean",
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(14, 6))

    bottom = np.zeros(len(df))
    for ftype, color in zip(failure_types, colors):
        if ftype in df.columns:
            values = df[ftype].values
            ax.bar(df["label"], values, bottom=bottom, label=ftype.replace("_", " ").title(), color=color)
            bottom += values

    ax.set_xlabel("Condition")
    ax.set_ylabel("Count")
    title = "Failure Type Distribution"
    if model_name:
        title += f" — {model_name}"
    ax.set_title(title)
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "failure_distribution.png")
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    return plot_path
