#!/usr/bin/env python3
"""
Create a pie chart for hybrid optimizer decision split:
- Neural decisions
- ESI fallbacks

Data source: logs/scraped_analysis/comprehensive_analysis_table.csv
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt


def to_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def load_hybrid_decision_totals(csv_path, pattern="all"):
    neural_total = 0
    fallback_total = 0
    row_count = 0

    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("optimizer_type", "").strip().lower() != "hybrid":
                continue
            if pattern != "all" and row.get("pattern") != pattern:
                continue

            neural_total += to_int(row.get("total_neural_decisions"))
            fallback_total += to_int(row.get("total_esi_fallbacks"))
            row_count += 1

    if row_count == 0:
        raise ValueError("No matching hybrid rows found for the selected filter.")

    return neural_total, fallback_total, row_count


def create_pie_chart(neural_total, fallback_total, output_path, title_suffix):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    values = [neural_total, fallback_total]
    labels = ["Neural Decisions", "ESI Fallbacks"]

    def autopct_with_count(pct):
        total = sum(values)
        count = int(round(pct * total / 100.0))
        return f"{pct:.1f}%\n(n={count:,})"

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        values,
        labels=labels,
        autopct=autopct_with_count,
        startangle=90,
        counterclock=False,
    )
    ax.set_title(f"Hybrid Decision Split{title_suffix}", fontsize=13, fontweight="bold")
    ax.axis("equal")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Pie chart of hybrid neural decisions vs ESI fallbacks")
    parser.add_argument(
        "--input",
        default="logs/scraped_analysis/comprehensive_analysis_table.csv",
        help="Path to comprehensive analysis table CSV",
    )
    parser.add_argument(
        "--pattern",
        default="all",
        help="Pattern filter (default: all)",
    )
    parser.add_argument(
        "--output",
        default="report_visualizations/evaluation_results/hybrid_neural_vs_esi_fallbacks_pie.png",
        help="Output image path",
    )

    args = parser.parse_args()

    neural_total, fallback_total, row_count = load_hybrid_decision_totals(
        csv_path=args.input,
        pattern=args.pattern,
    )

    title_suffix = " (All Patterns)" if args.pattern == "all" else f" ({args.pattern.title()} Pattern)"
    create_pie_chart(neural_total, fallback_total, args.output, title_suffix)

    print(f"Hybrid rows included: {row_count}")
    print(f"Neural decisions: {neural_total:,}")
    print(f"ESI fallbacks:   {fallback_total:,}")
    print(f"Saved chart: {args.output}")


if __name__ == "__main__":
    main()
