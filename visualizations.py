#!/usr/bin/env python3
"""
Visualization scripts for LLM Memory Analysis

Generates:
1. f1_by_condition.png - Grouped bar chart (F1 by condition, grouped by vendor)
2. performance_degradation.png - Line plot showing F1 drop across conditions
3. f1_heatmap.png - Heatmap (vendors x conditions)
4. valid_rates.png - Valid prediction rates
5. confusion_matrices.png - 3x3 grid (vendors x conditions)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Set Arial as default font
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.sans-serif"] = ["Arial"]


def get_vendor_display_name(vendor: str) -> str:
    """Map internal vendor names to display names."""
    mapping = {
        "openai": "GPT-5.2",
        "gemini": "Gemini 3 Flash",
        "anthropic": "Claude Opus 4.5",
    }
    return mapping.get(vendor, vendor.upper())


def get_condition_display_name(condition: str) -> str:
    """Map condition names to display names."""
    mapping = {
        "baseline": "Baseline\n(Title + Abstract)",
        "title_only": "Title Only",
        "doi_only": "DOI Only",
        "counterfactual": "Counterfactual",
        "counterfactual_with_doi": "Counterfactual\n+ DOI",
    }
    return mapping.get(condition, condition)


def plot_f1_by_condition(metrics_df: pd.DataFrame, output_dir: Path):
    """Grouped bar chart of F1 by condition, grouped by vendor."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data
    vendors = ["openai", "gemini", "anthropic"]
    conditions = ["baseline", "title_only", "doi_only", "counterfactual", "counterfactual_with_doi"]

    x = np.arange(len(conditions))
    width = 0.25

    # Paul Tol colorblind-friendly palette
    colors = ["#4477AA", "#EE6677", "#228833"]

    for i, vendor in enumerate(vendors):
        vendor_df = metrics_df[metrics_df["vendor"] == vendor]
        f1_values = []
        for condition in conditions:
            cond_df = vendor_df[vendor_df["condition"] == condition]
            if len(cond_df) > 0:
                f1_values.append(cond_df["f1_macro"].iloc[0])
            else:
                f1_values.append(0)

        bars = ax.bar(x + i * width, f1_values, width, label=get_vendor_display_name(vendor), color=colors[i])

        # Add value labels on bars
        for bar, val in zip(bars, f1_values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    ax.set_xlabel("Input Condition", fontsize=12)
    ax.set_ylabel("F1 Score (Macro)", fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels([get_condition_display_name(c) for c in conditions])
    ax.legend(title="Model")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = output_dir / "f1_by_condition.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_performance_degradation(metrics_df: pd.DataFrame, output_dir: Path):
    """Line plot showing F1 drop across conditions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    vendors = ["openai", "gemini", "anthropic"]
    conditions = ["baseline", "title_only", "doi_only"]
    # Paul Tol colorblind-friendly palette
    colors = ["#4477AA", "#EE6677", "#228833"]
    markers = ["o", "s", "^"]

    for i, vendor in enumerate(vendors):
        vendor_df = metrics_df[metrics_df["vendor"] == vendor]

        f1_values = []
        for condition in conditions:
            cond_df = vendor_df[vendor_df["condition"] == condition]
            if len(cond_df) > 0:
                f1_values.append(cond_df["f1_macro"].iloc[0])
            else:
                f1_values.append(np.nan)

        ax.plot(
            range(len(conditions)),
            f1_values,
            marker=markers[i],
            markersize=10,
            linewidth=2,
            label=get_vendor_display_name(vendor),
            color=colors[i],
        )

    ax.set_xlabel("Input Condition", fontsize=12)
    ax.set_ylabel("F1 Score (Macro)", fontsize=12)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([get_condition_display_name(c) for c in conditions])
    ax.legend(title="Model")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    output_path = output_dir / "performance_degradation.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_f1_heatmap(metrics_df: pd.DataFrame, output_dir: Path):
    """Heatmap of F1 scores (vendors x conditions)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create pivot table
    pivot = metrics_df.pivot_table(values="f1_macro", index="vendor", columns="condition", aggfunc="first")

    # Reorder
    condition_order = ["baseline", "title_only", "doi_only", "counterfactual", "counterfactual_with_doi"]
    vendor_order = ["openai", "gemini", "anthropic"]
    pivot = pivot[[c for c in condition_order if c in pivot.columns]]
    pivot = pivot.reindex([v for v in vendor_order if v in pivot.index])

    # Rename for display
    pivot.index = [get_vendor_display_name(v) for v in pivot.index]
    column_labels = {
        "baseline": "Baseline",
        "title_only": "Title Only",
        "doi_only": "DOI Only",
        "counterfactual": "Counterfactual",
        "counterfactual_with_doi": "Counterfactual + DOI",
    }
    pivot.columns = [column_labels.get(c, c) for c in pivot.columns]

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0.4,
        vmax=1.0,
        ax=ax,
        cbar_kws={"label": "F1 Score"},
        linewidths=0.5,
    )

    ax.set_xlabel("Input Condition", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)

    plt.tight_layout()
    output_path = output_dir / "f1_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_valid_rates(metrics_df: pd.DataFrame, output_dir: Path):
    """Bar chart of valid prediction rates."""
    fig, ax = plt.subplots(figsize=(12, 6))

    vendors = ["openai", "gemini", "anthropic"]
    conditions = ["baseline", "title_only", "doi_only", "counterfactual", "counterfactual_with_doi"]

    x = np.arange(len(conditions))
    width = 0.25

    colors = sns.color_palette("rocket", n_colors=3)

    for i, vendor in enumerate(vendors):
        vendor_df = metrics_df[metrics_df["vendor"] == vendor]
        valid_pct = []
        for condition in conditions:
            cond_df = vendor_df[vendor_df["condition"] == condition]
            if len(cond_df) > 0:
                valid_pct.append(cond_df["pct_valid"].iloc[0])
            else:
                valid_pct.append(0)

        bars = ax.bar(x + i * width, valid_pct, width, label=get_vendor_display_name(vendor), color=colors[i])

        # Add value labels
        for bar, val in zip(bars, valid_pct):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Input Condition", fontsize=12)
    ax.set_ylabel("Valid Predictions (%)", fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels([get_condition_display_name(c) for c in conditions])
    ax.legend(title="Model")
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = output_dir / "valid_rates.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_confusion_matrices_combined(predictions_df: pd.DataFrame, output_dir: Path):
    """5x3 grid of confusion matrices (5 conditions × 3 vendors), no figure title."""
    from sklearn.metrics import f1_score

    vendors = ["openai", "gemini", "anthropic"]
    conditions = ["baseline", "title_only", "doi_only", "counterfactual", "counterfactual_with_doi"]
    condition_labels = {
        "baseline": "Baseline",
        "title_only": "Title Only",
        "doi_only": "DOI Only",
        "counterfactual": "Counterfactual",
        "counterfactual_with_doi": "Counterfactual + DOI",
    }

    fig, axes = plt.subplots(5, 3, figsize=(12, 16))

    for row_idx, condition in enumerate(conditions):
        for col_idx, vendor in enumerate(vendors):
            ax = axes[row_idx, col_idx]

            # Filter predictions
            mask = (predictions_df["vendor"] == vendor) & (predictions_df["condition"] == condition)
            setting_df = predictions_df[mask]

            # Filter valid predictions only
            valid_mask = setting_df["parsed_output"].isin(["POSITIVE", "NEGATIVE"]) & setting_df["ground_truth"].isin(
                ["POSITIVE", "NEGATIVE"]
            )
            valid_df = setting_df[valid_mask]

            if len(valid_df) > 0:
                y_true = valid_df["ground_truth"]
                y_pred = valid_df["parsed_output"]

                # Compute confusion matrix
                cm = confusion_matrix(y_true, y_pred, labels=["POSITIVE", "NEGATIVE"])

                # Calculate F1
                f1 = f1_score(y_true, y_pred, average="macro", labels=["POSITIVE", "NEGATIVE"])

                # Plot
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Pos", "Neg"])
                disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")

                # Column headers (vendor names) on top row only
                if row_idx == 0:
                    ax.set_title(f"{get_vendor_display_name(vendor)}\n(F1={f1:.2f})", fontsize=10, fontweight="bold")
                else:
                    ax.set_title(f"(F1={f1:.2f})", fontsize=10)
            else:
                ax.text(0.5, 0.5, "No valid\npredictions", ha="center", va="center", fontsize=9)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                if row_idx == 0:
                    ax.set_title(get_vendor_display_name(vendor), fontsize=10, fontweight="bold")

            # Row labels (condition names) on left column
            if col_idx == 0:
                ax.set_ylabel(f"{condition_labels[condition]}\n\nTrue Label", fontsize=10, fontweight="bold")
            else:
                ax.set_ylabel("")

            # X-axis label only on bottom row
            if row_idx < 4:
                ax.set_xlabel("")
            else:
                ax.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")

    plt.tight_layout()
    output_path = output_dir / "confusion_matrices_combined.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_counterfactual_comparison(metrics_df: pd.DataFrame, output_dir: Path):
    """Grouped bar chart comparing baseline vs counterfactual vs counterfactual_with_doi F1 by vendor."""
    # Check if counterfactual data exists
    cf_df = metrics_df[metrics_df["condition"] == "counterfactual"]
    if len(cf_df) == 0:
        print("No counterfactual data found, skipping counterfactual_f1_comparison.png")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    vendors = ["openai", "gemini", "anthropic"]
    conditions = ["baseline", "counterfactual", "counterfactual_with_doi"]

    x = np.arange(len(vendors))
    width = 0.25

    # Paul Tol colorblind-friendly palette
    colors = ["#4477AA", "#EE6677", "#AA3377"]  # Blue for baseline, coral for counterfactual, magenta for cf+doi

    for i, condition in enumerate(conditions):
        f1_values = []
        for vendor in vendors:
            cond_df = metrics_df[(metrics_df["vendor"] == vendor) & (metrics_df["condition"] == condition)]
            if len(cond_df) > 0:
                f1_values.append(cond_df["f1_macro"].iloc[0])
            else:
                f1_values.append(0)

        label_map = {"baseline": "Baseline", "counterfactual": "Counterfactual", "counterfactual_with_doi": "Counterfactual + DOI"}
        label = label_map.get(condition, condition)
        bars = ax.bar(x + i * width, f1_values, width, label=label, color=colors[i])

        # Add value labels on bars
        for bar, val in zip(bars, f1_values):
            if not np.isnan(val) and val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("F1 Score (Macro)", fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels([get_vendor_display_name(v) for v in vendors])
    ax.legend(title="Condition")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = output_dir / "counterfactual_f1_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_counterfactual_confusion_matrices(predictions_df: pd.DataFrame, output_dir: Path):
    """2x3 grid of confusion matrices for counterfactual conditions (2 conditions × 3 vendors)."""
    # Check if counterfactual data exists
    cf_df = predictions_df[predictions_df["condition"] == "counterfactual"]
    if len(cf_df) == 0:
        print("No counterfactual data found, skipping counterfactual_confusion_matrices.png")
        return

    vendors = ["openai", "gemini", "anthropic"]
    conditions = ["counterfactual", "counterfactual_with_doi"]
    condition_labels = {"counterfactual": "Counterfactual", "counterfactual_with_doi": "Counterfactual + DOI"}

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    from sklearn.metrics import f1_score

    for row_idx, condition in enumerate(conditions):
        for col_idx, vendor in enumerate(vendors):
            ax = axes[row_idx, col_idx]

            # Filter predictions
            mask = (predictions_df["vendor"] == vendor) & (predictions_df["condition"] == condition)
            setting_df = predictions_df[mask]

            # Filter valid predictions only
            valid_mask = setting_df["parsed_output"].isin(["POSITIVE", "NEGATIVE"]) & setting_df["ground_truth"].isin(
                ["POSITIVE", "NEGATIVE"]
            )
            valid_df = setting_df[valid_mask]

            if len(valid_df) > 0:
                y_true = valid_df["ground_truth"]
                y_pred = valid_df["parsed_output"]

                # Compute confusion matrix
                cm = confusion_matrix(y_true, y_pred, labels=["POSITIVE", "NEGATIVE"])

                # Calculate F1
                f1 = f1_score(y_true, y_pred, average="macro", labels=["POSITIVE", "NEGATIVE"])

                # Plot
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Positive", "Negative"])
                disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")

                # Title: show condition label on top row, model name and F1 for all
                if row_idx == 0:
                    title = f"{condition_labels[condition]}\n{get_vendor_display_name(vendor)}\n(F1={f1:.3f})"
                else:
                    title = f"{condition_labels[condition]}\n{get_vendor_display_name(vendor)}\n(F1={f1:.3f})"
                ax.set_title(title, fontsize=10, fontweight="bold")
            else:
                ax.text(0.5, 0.5, "No valid\npredictions", ha="center", va="center", fontsize=10)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_title(f"{condition_labels[condition]}\n{get_vendor_display_name(vendor)}", fontsize=10, fontweight="bold")

            # Clean up labels
            if row_idx < 1:
                ax.set_xlabel("")
            if col_idx > 0:
                ax.set_ylabel("")

    plt.tight_layout()
    output_path = output_dir / "counterfactual_confusion_matrices.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_counterfactual_performance(metrics_df: pd.DataFrame, output_dir: Path):
    """Bar chart showing counterfactual and counterfactual_with_doi F1 scores by vendor."""
    # Check if counterfactual data exists
    cf_df = metrics_df[metrics_df["condition"] == "counterfactual"]
    if len(cf_df) == 0:
        print("No counterfactual data found, skipping counterfactual_performance.png")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    vendors = ["openai", "gemini", "anthropic"]
    conditions = ["counterfactual", "counterfactual_with_doi"]
    # Paul Tol colorblind-friendly palette
    colors = ["#EE6677", "#AA3377"]  # Coral for counterfactual, magenta for cf+doi

    x = np.arange(len(vendors))
    width = 0.35

    for i, condition in enumerate(conditions):
        f1_values = []
        for vendor in vendors:
            cond_df = metrics_df[(metrics_df["vendor"] == vendor) & (metrics_df["condition"] == condition)]
            if len(cond_df) > 0:
                f1_values.append(cond_df["f1_macro"].iloc[0])
            else:
                f1_values.append(0)

        label = "Counterfactual" if condition == "counterfactual" else "Counterfactual + DOI"
        bars = ax.bar(x + i * width, f1_values, width, label=label, color=colors[i])

        # Add value labels on bars
        for bar, val in zip(bars, f1_values):
            if not np.isnan(val) and val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("F1 Score (Macro)", fontsize=12)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([get_vendor_display_name(v) for v in vendors])
    ax.legend(title="Condition")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = output_dir / "counterfactual_performance.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Generate all visualizations."""
    script_dir = Path(__file__).parent

    # Load data
    predictions_path = script_dir / "results" / "predictions.csv"
    metrics_path = script_dir / "results" / "metrics.csv"

    if not predictions_path.exists():
        print(f"Error: {predictions_path} not found")
        print("Run run_evaluation.py and analyze_results.py first.")
        return 1

    if not metrics_path.exists():
        print(f"Error: {metrics_path} not found")
        print("Run analyze_results.py first.")
        return 1

    predictions_df = pd.read_csv(predictions_path)
    metrics_df = pd.read_csv(metrics_path)

    # Output directory for plots
    plots_dir = script_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print("Generating visualizations...\n")

    # Create all plots
    plot_f1_by_condition(metrics_df, plots_dir)
    plot_performance_degradation(metrics_df, plots_dir)
    plot_f1_heatmap(metrics_df, plots_dir)
    plot_valid_rates(metrics_df, plots_dir)
    plot_confusion_matrices_combined(predictions_df, plots_dir)

    # Create counterfactual-specific plots (if data exists)
    plot_counterfactual_comparison(metrics_df, plots_dir)
    plot_counterfactual_confusion_matrices(predictions_df, plots_dir)
    plot_counterfactual_performance(metrics_df, plots_dir)

    print("\nAll visualizations generated successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
