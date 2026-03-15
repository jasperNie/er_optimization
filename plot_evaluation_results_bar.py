#!/usr/bin/env python3
"""
Create a bar chart from logs/evaluation_results.txt FINAL PERFORMANCE RANKING table.
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np


def parse_ranking_table(results_path):
    """Parse ranking rows from evaluation_results.txt."""
    with open(results_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    in_table = False
    rows = []

    row_pattern = re.compile(
        r"^\s*(\d+)\s+(.+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$"
    )

    for raw_line in lines:
        line = raw_line.rstrip("\n")

        if line.strip().startswith("Rank Algorithm"):
            in_table = True
            continue

        if not in_table:
            continue

        stripped = line.strip()
        if not stripped:
            break
        if stripped.startswith("IMPROVEMENT ANALYSIS") or stripped.startswith("==="):
            break
        if set(stripped) == {"-"}:
            continue

        match = row_pattern.match(line)
        if match:
            rows.append(
                {
                    "rank": int(match.group(1)),
                    "algorithm": match.group(2).strip(),
                    "weighted_wait": float(match.group(3)),
                    "completed": float(match.group(4)),
                    "unattended": float(match.group(5)),
                    "combined_score": float(match.group(6)),
                }
            )

    if not rows:
        raise ValueError(
            "No ranking rows found. Check that the file contains the FINAL PERFORMANCE RANKING table."
        )

    rows.sort(key=lambda item: item["rank"])
    return rows


def make_bar_chart(rows, output_path):
    """Generate and save grouped bar chart for weighted and combined scores."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    labels = [item["algorithm"] for item in rows]
    weighted = [item["weighted_wait"] for item in rows]
    combined = [item["combined_score"] for item in rows]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 7))

    bars_weighted = ax.bar(x - width / 2, weighted, width, label="Weighted Wait", alpha=0.9)
    bars_combined = ax.bar(x + width / 2, combined, width, label="Combined Score", alpha=0.9)

    ax.set_title("Final Performance Ranking by Algorithm", fontsize=14, fontweight="bold")
    ax.set_xlabel("Algorithm", fontsize=11)
    ax.set_ylabel("Score (Lower is Better)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")

    for bar in bars_weighted:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for bar in bars_combined:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot bar chart from evaluation_results ranking table")
    parser.add_argument(
        "--input",
        default="logs/evaluation_results.txt",
        help="Path to evaluation results text file",
    )
    parser.add_argument(
        "--output",
        default="report_visualizations/evaluation_results/final_ranking_bar_chart.png",
        help="Path to save the output bar chart",
    )

    args = parser.parse_args()

    rows = parse_ranking_table(args.input)
    make_bar_chart(rows, args.output)

    print(f"Parsed {len(rows)} algorithms from: {args.input}")
    print(f"Bar chart saved to: {args.output}")


if __name__ == "__main__":
    main()
