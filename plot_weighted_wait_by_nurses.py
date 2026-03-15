#!/usr/bin/env python3
"""
Plot weighted wait time by nurse count (2-6) with grouped bars:
- Neural
- Hybrid Neural
- ESI
- MTS

Data source: logs/scraped_analysis/comprehensive_analysis_table.csv
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_or_nan(values):
    valid = [v for v in values if v is not None]
    if not valid:
        return np.nan
    return float(sum(valid) / len(valid))


def load_weighted_wait_data(csv_path, pattern, min_nurses=2, max_nurses=6):
    """Load weighted wait values by nurse count.

    If pattern='all', returns averages across all available patterns
    for each nurse count.
    """
    pattern_nurse_data = {}
    available_patterns = set()

    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            row_pattern = row.get("pattern")
            if not row_pattern:
                continue
            if pattern != "all" and row_pattern != pattern:
                continue

            available_patterns.add(row_pattern)

            nurses = int(row["nurses"])
            if nurses < min_nurses or nurses > max_nurses:
                continue

            optimizer_type = row.get("optimizer_type", "").strip().lower()

            entry = pattern_nurse_data.setdefault(
                (row_pattern, nurses),
                {
                    "neural": None,
                    "hybrid": None,
                    "esi_values": [],
                    "mts_values": [],
                },
            )

            neural_weighted = _to_float(row.get("neural_weighted_wait"))
            esi_weighted = _to_float(row.get("esi_baseline_weighted"))
            mts_weighted = _to_float(row.get("mts_baseline_weighted"))

            if optimizer_type == "neural":
                entry["neural"] = neural_weighted
            elif optimizer_type == "hybrid":
                entry["hybrid"] = neural_weighted

            if esi_weighted is not None:
                entry["esi_values"].append(esi_weighted)
            if mts_weighted is not None:
                entry["mts_values"].append(mts_weighted)

    nurses_sorted = sorted(
        {
            nurse_count
            for _, nurse_count in pattern_nurse_data.keys()
            if min_nurses <= nurse_count <= max_nurses
        }
    )
    if not nurses_sorted:
        raise ValueError(f"No rows found for pattern '{pattern}' in nurse range {min_nurses}-{max_nurses}.")

    selected_patterns = sorted(available_patterns)
    if pattern != "all":
        selected_patterns = [pattern]

    neural = []
    hybrid = []
    esi = []
    mts = []

    for nurse_count in nurses_sorted:
        nurse_neural = []
        nurse_hybrid = []
        nurse_esi = []
        nurse_mts = []

        for current_pattern in selected_patterns:
            entry = pattern_nurse_data.get((current_pattern, nurse_count))
            if not entry:
                continue

            nurse_neural.append(entry["neural"])
            nurse_hybrid.append(entry["hybrid"])
            nurse_esi.append(_mean_or_nan(entry["esi_values"]))
            nurse_mts.append(_mean_or_nan(entry["mts_values"]))

        neural.append(_mean_or_nan(nurse_neural))
        hybrid.append(_mean_or_nan(nurse_hybrid))
        esi.append(_mean_or_nan(nurse_esi))
        mts.append(_mean_or_nan(nurse_mts))

    return nurses_sorted, neural, hybrid, esi, mts, selected_patterns


def plot_grouped_bars(nurses, neural, hybrid, esi, mts, pattern_label, output_path):
    """Create grouped bar chart for weighted wait time by nurses."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    x = np.arange(len(nurses))
    width = 0.2

    fig, ax = plt.subplots(figsize=(11, 6.5))

    ax.bar(x - 1.5 * width, neural, width, label="Neural")
    ax.bar(x - 0.5 * width, hybrid, width, label="Hybrid Neural")
    ax.bar(x + 0.5 * width, esi, width, label="ESI")
    ax.bar(x + 1.5 * width, mts, width, label="MTS")

    ax.set_title(f"Weighted Wait Time by Nurse Count ({pattern_label})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Nurses", fontsize=11)
    ax.set_ylabel("Weighted Wait Time (hours)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in nurses])
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Grouped bar chart: weighted wait vs nurses")
    parser.add_argument(
        "--input",
        default="logs/scraped_analysis/comprehensive_analysis_table.csv",
        help="Path to comprehensive analysis CSV",
    )
    parser.add_argument(
        "--pattern",
        default="disaster",
        help="Arrival pattern to plot, or 'all' for average across patterns (default: disaster)",
    )
    parser.add_argument("--min-nurses", type=int, default=2, help="Minimum nurse count (default: 2)")
    parser.add_argument("--max-nurses", type=int, default=6, help="Maximum nurse count (default: 6)")
    parser.add_argument(
        "--output",
        default="",
        help="Output image path",
    )

    args = parser.parse_args()

    nurses, neural, hybrid, esi, mts, selected_patterns = load_weighted_wait_data(
        csv_path=args.input,
        pattern=args.pattern,
        min_nurses=args.min_nurses,
        max_nurses=args.max_nurses,
    )

    if args.pattern == "all":
        pattern_label = "All Patterns Average"
        default_output = "report_visualizations/evaluation_results/weighted_wait_by_nurses_all_patterns.png"
    else:
        pattern_label = f"{args.pattern.title()} Pattern"
        default_output = f"report_visualizations/evaluation_results/weighted_wait_by_nurses_{args.pattern}.png"

    output_path = args.output or default_output

    plot_grouped_bars(
        nurses=nurses,
        neural=neural,
        hybrid=hybrid,
        esi=esi,
        mts=mts,
        pattern_label=pattern_label,
        output_path=output_path,
    )

    print(f"Pattern: {args.pattern}")
    if args.pattern == "all":
        print(f"Averaged across patterns: {selected_patterns}")
    print(f"Nurses: {nurses}")
    print(f"Saved chart: {output_path}")


if __name__ == "__main__":
    main()
