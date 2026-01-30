#!/usr/bin/env python3
"""
Analyze LLM Memory Evaluation Results

Computes per (vendor, condition):
- F1 score (macro) and accuracy
- Valid prediction percentage
- n_invalid, n_error counts

Output tables:
- metrics.csv - Full metrics per setting
- table_f1_by_condition.csv - Pivot (vendors x conditions)
- table_degradation.csv - Performance drop from baseline
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def analyze_predictions(predictions_path: Path, output_dir: Path):
    """Main analysis function."""

    # Load predictions
    df = pd.read_csv(predictions_path)
    print(f"Loaded {len(df)} predictions")

    # Get unique (vendor, condition) combinations
    settings_cols = ["vendor", "model", "condition"]
    settings = df[settings_cols].drop_duplicates()
    print(f"Found {len(settings)} unique (vendor, condition) combinations")

    # Results storage
    metrics_rows = []

    for _, setting in settings.iterrows():
        vendor = setting["vendor"]
        model = setting["model"]
        condition = setting["condition"]

        # Filter to this setting
        mask = (df["vendor"] == vendor) & (df["condition"] == condition)
        df_setting = df[mask].copy()

        if len(df_setting) == 0:
            continue

        n_total = len(df_setting)

        # Count valid, invalid, error
        n_valid = len(df_setting[df_setting["parsed_output"].isin(["POSITIVE", "NEGATIVE"])])
        n_invalid = len(df_setting[df_setting["parsed_output"] == "INVALID"])
        n_error = len(df_setting[df_setting["parsed_output"] == "ERROR"])
        pct_valid = (n_valid / n_total * 100) if n_total > 0 else 0.0

        # Filter for valid predictions for F1/accuracy
        valid_mask = df_setting["parsed_output"].isin(["POSITIVE", "NEGATIVE"]) & df_setting["ground_truth"].isin(
            ["POSITIVE", "NEGATIVE"]
        )
        valid_df = df_setting[valid_mask]

        if len(valid_df) > 0:
            y_true = valid_df["ground_truth"]
            y_pred = valid_df["parsed_output"]
            accuracy = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average="macro", labels=["POSITIVE", "NEGATIVE"])
        else:
            accuracy = np.nan
            f1_macro = np.nan

        metrics_rows.append(
            {
                "vendor": vendor,
                "model": model,
                "condition": condition,
                "accuracy": accuracy,
                "f1_macro": f1_macro,
                "n_total": n_total,
                "n_valid": n_valid,
                "n_invalid": n_invalid,
                "n_error": n_error,
                "pct_valid": pct_valid,
            }
        )

        print(f"{vendor} {condition}: F1={f1_macro:.3f}, acc={accuracy:.3f}, valid={pct_valid:.1f}%")

    # Create metrics DataFrame
    metrics_df = pd.DataFrame(metrics_rows)

    # Save full metrics
    metrics_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics to {metrics_path}")

    # Generate pivot table: F1 by condition (vendors x conditions)
    generate_f1_pivot_table(metrics_df, output_dir)

    # Generate degradation table
    generate_degradation_table(metrics_df, output_dir)

    # Generate counterfactual analysis (if counterfactual data exists)
    generate_counterfactual_analysis(metrics_df, output_dir)

    # Generate manuscript table
    generate_manuscript_table(predictions_path, output_dir)

    # Print summary
    print("\n=== Summary ===")
    print(f"Total settings analyzed: {len(metrics_df)}")

    print("\nF1 by vendor and condition:")
    pivot = metrics_df.pivot_table(values="f1_macro", index="vendor", columns="condition")
    print(pivot.to_string())

    print("\nBest F1 scores:")
    top_f1 = metrics_df.nlargest(5, "f1_macro")[["vendor", "condition", "f1_macro", "accuracy", "pct_valid"]]
    print(top_f1.to_string(index=False))


def generate_f1_pivot_table(metrics_df: pd.DataFrame, output_dir: Path):
    """Generate pivot table of F1 scores (vendors x conditions)."""

    # Create pivot
    pivot = metrics_df.pivot_table(values="f1_macro", index="vendor", columns="condition", aggfunc="first")

    # Reorder columns
    condition_order = ["baseline", "title_only", "doi_only", "counterfactual", "counterfactual_with_doi"]
    pivot = pivot[[c for c in condition_order if c in pivot.columns]]

    # Reorder index (vendors)
    vendor_order = ["openai", "gemini", "anthropic"]
    pivot = pivot.reindex([v for v in vendor_order if v in pivot.index])

    # Save
    output_path = output_dir / "table_f1_by_condition.csv"
    pivot.to_csv(output_path)
    print(f"Saved F1 pivot table to {output_path}")


def generate_degradation_table(metrics_df: pd.DataFrame, output_dir: Path):
    """Generate table showing performance degradation from baseline."""

    rows = []
    for vendor in metrics_df["vendor"].unique():
        vendor_df = metrics_df[metrics_df["vendor"] == vendor]

        # Get baseline F1
        baseline_row = vendor_df[vendor_df["condition"] == "baseline"]
        if len(baseline_row) == 0:
            continue
        baseline_f1 = baseline_row["f1_macro"].iloc[0]

        for _, row in vendor_df.iterrows():
            condition = row["condition"]
            f1 = row["f1_macro"]

            # Calculate absolute and relative degradation
            abs_degradation = baseline_f1 - f1
            rel_degradation = (abs_degradation / baseline_f1 * 100) if baseline_f1 > 0 else 0

            rows.append(
                {
                    "vendor": vendor,
                    "condition": condition,
                    "f1_macro": f1,
                    "baseline_f1": baseline_f1,
                    "abs_degradation": abs_degradation,
                    "rel_degradation_pct": rel_degradation,
                }
            )

    degradation_df = pd.DataFrame(rows)

    # Save
    output_path = output_dir / "table_degradation.csv"
    degradation_df.to_csv(output_path, index=False)
    print(f"Saved degradation table to {output_path}")


def generate_counterfactual_analysis(metrics_df: pd.DataFrame, output_dir: Path):
    """Generate dedicated counterfactual analysis tables."""
    # Filter to counterfactual only
    cf_df = metrics_df[metrics_df["condition"] == "counterfactual"].copy()

    if len(cf_df) == 0:
        print("No counterfactual data found, skipping counterfactual analysis")
        return

    # Save counterfactual metrics
    cf_path = output_dir / "table_counterfactual_metrics.csv"
    cf_df.to_csv(cf_path, index=False)

    # Baseline vs Counterfactual comparison
    baseline_df = metrics_df[metrics_df["condition"] == "baseline"]
    comparison_rows = []
    for vendor in cf_df["vendor"].unique():
        cf_row = cf_df[cf_df["vendor"] == vendor]
        bl_row = baseline_df[baseline_df["vendor"] == vendor]
        if len(cf_row) > 0 and len(bl_row) > 0:
            comparison_rows.append({
                "vendor": vendor,
                "baseline_f1": bl_row["f1_macro"].iloc[0],
                "counterfactual_f1": cf_row["f1_macro"].iloc[0],
                "f1_difference": bl_row["f1_macro"].iloc[0] - cf_row["f1_macro"].iloc[0],
            })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = output_dir / "table_baseline_vs_counterfactual.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Saved counterfactual analysis to {cf_path} and {comparison_path}")

    # Analyze sections modified
    script_dir = Path(__file__).parent
    trials_path = script_dir / "data" / "trials.csv"
    if trials_path.exists():
        generate_sections_modified_analysis(trials_path, output_dir)


def generate_sections_modified_analysis(trials_path: Path, output_dir: Path):
    """Analyze which sections were modified in counterfactual abstracts."""
    trials_df = pd.read_csv(trials_path)

    if "sections_modified" not in trials_df.columns:
        print("No sections_modified column found, skipping section analysis")
        return

    # Parse the sections_modified column
    trials_df["sections_list"] = trials_df["sections_modified"].apply(json.loads)

    total_trials = len(trials_df)

    # Count how often each section is modified
    section_counts = {}
    for sections in trials_df["sections_list"]:
        for section in sections:
            section_counts[section] = section_counts.get(section, 0) + 1

    # Create per-section table
    section_rows = []
    for section in ["title", "methods", "results", "conclusion"]:
        count = section_counts.get(section, 0)
        pct = count / total_trials * 100
        section_rows.append({
            "section": section,
            "n_modified": count,
            "pct_modified": pct,
        })

    section_df = pd.DataFrame(section_rows)
    section_path = output_dir / "table_counterfactual_sections.csv"
    section_df.to_csv(section_path, index=False)

    # Create modification patterns table
    pattern_counts = trials_df["sections_modified"].value_counts()
    pattern_rows = []
    for pattern, count in pattern_counts.items():
        pct = count / total_trials * 100
        pattern_rows.append({
            "pattern": pattern,
            "n_trials": count,
            "pct_trials": pct,
        })

    pattern_df = pd.DataFrame(pattern_rows)
    pattern_path = output_dir / "table_counterfactual_patterns.csv"
    pattern_df.to_csv(pattern_path, index=False)

    print(f"Saved section analysis to {section_path} and {pattern_path}")


def _normal_approx_ci(p: float, n: int, z: float = 1.96) -> tuple:
    """Calculate 95% CI using normal approximation (Wald interval)."""
    if n == 0 or np.isnan(p):
        return (np.nan, np.nan)
    se = np.sqrt(p * (1 - p) / n)
    lower = max(0, p - z * se)
    upper = min(1, p + z * se)
    return (lower, upper)


def _format_with_ci(value: float, ci_lower: float, ci_upper: float) -> str:
    """Format a value with its 95% CI."""
    if np.isnan(value):
        return "N/A"
    return f"{value:.2f} ({ci_lower:.2f}-{ci_upper:.2f})"


def generate_manuscript_table(predictions_path: Path, output_dir: Path):
    """Generate manuscript-ready table with Condition, Model, Valid %, Accuracy, Sensitivity, Specificity, F1."""
    df = pd.read_csv(predictions_path)

    # Define order
    condition_order = ["baseline", "title_only", "doi_only", "counterfactual", "counterfactual_with_doi"]
    condition_labels = {
        "baseline": "Baseline",
        "title_only": "Title only",
        "doi_only": "DOI only",
        "counterfactual": "Counterfactual",
        "counterfactual_with_doi": "Counterfactual + DOI",
    }
    vendor_order = ["openai", "gemini", "anthropic"]
    vendor_labels = {
        "openai": "GPT-5.2",
        "gemini": "Gemini 3 Flash",
        "anthropic": "Claude Opus 4.5",
    }

    rows = []
    for condition in condition_order:
        for vendor in vendor_order:
            mask = (df["vendor"] == vendor) & (df["condition"] == condition)
            df_setting = df[mask]

            if len(df_setting) == 0:
                continue

            n_total = len(df_setting)
            n_valid = len(df_setting[df_setting["parsed_output"].isin(["POSITIVE", "NEGATIVE"])])
            pct_valid = (n_valid / n_total * 100) if n_total > 0 else 0.0

            # Filter for valid predictions
            valid_mask = df_setting["parsed_output"].isin(["POSITIVE", "NEGATIVE"]) & df_setting["ground_truth"].isin(
                ["POSITIVE", "NEGATIVE"]
            )
            valid_df = df_setting[valid_mask]

            if len(valid_df) > 0:
                y_true = valid_df["ground_truth"]
                y_pred = valid_df["parsed_output"]
                n = len(valid_df)

                accuracy = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average="macro", labels=["POSITIVE", "NEGATIVE"])

                # Compute confusion matrix for sensitivity/specificity
                cm = confusion_matrix(y_true, y_pred, labels=["POSITIVE", "NEGATIVE"])
                tp, fn = cm[0, 0], cm[0, 1]
                fp, tn = cm[1, 0], cm[1, 1]

                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

                # Calculate 95% CIs using normal approximation
                acc_ci = _normal_approx_ci(accuracy, n)
                sens_ci = _normal_approx_ci(sensitivity, tp + fn)
                spec_ci = _normal_approx_ci(specificity, tn + fp)
                f1_ci = _normal_approx_ci(f1, n)
            else:
                accuracy = np.nan
                f1 = np.nan
                sensitivity = np.nan
                specificity = np.nan
                acc_ci = (np.nan, np.nan)
                sens_ci = (np.nan, np.nan)
                spec_ci = (np.nan, np.nan)
                f1_ci = (np.nan, np.nan)

            rows.append({
                "Condition": condition_labels[condition],
                "Model": vendor_labels[vendor],
                "Valid predictions (%)": f"{pct_valid:.1f}",
                "Accuracy (95% CI)": _format_with_ci(accuracy, *acc_ci),
                "Sensitivity (95% CI)": _format_with_ci(sensitivity, *sens_ci),
                "Specificity (95% CI)": _format_with_ci(specificity, *spec_ci),
                "F1 Score (95% CI)": _format_with_ci(f1, *f1_ci),
            })

    manuscript_df = pd.DataFrame(rows)

    # Save as CSV
    output_path = output_dir / "table_manuscript.csv"
    manuscript_df.to_csv(output_path, index=False)
    print(f"Saved manuscript table to {output_path}")

    # Also save as formatted text for easy copy-paste
    txt_path = output_dir / "table_manuscript.txt"
    with open(txt_path, "w") as f:
        f.write(manuscript_df.to_string(index=False))
    print(f"Saved manuscript table (text) to {txt_path}")

    return manuscript_df


def main():
    parser = argparse.ArgumentParser(description="Analyze LLM memory results")
    parser.add_argument("--predictions", type=str, default=None, help="Path to predictions.csv")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")
    args = parser.parse_args()

    # Paths
    script_dir = Path(__file__).parent
    predictions_path = Path(args.predictions) if args.predictions else script_dir / "results" / "predictions.csv"
    output_dir = Path(args.output_dir) if args.output_dir else script_dir / "results"

    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)

    if not predictions_path.exists():
        print(f"Error: Predictions file not found: {predictions_path}")
        print("Run run_evaluation.py first to generate predictions.")
        return 1

    analyze_predictions(predictions_path, output_dir)
    return 0


if __name__ == "__main__":
    exit(main())
