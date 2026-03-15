#!/usr/bin/env python3
"""
Plot estimated per-timestep patient arrival probability for each registered arrival pattern
on a single line chart.
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from arrival_patterns import ARRIVAL_PATTERNS, get_pattern_description


def estimate_pattern_probability(pattern_func, total_time, arrival_prob, num_seeds, start_seed):
    """Estimate arrival probability at each timestep by averaging arrivals over many seeds."""
    arrival_counts = np.zeros(total_time, dtype=float)

    for offset in range(num_seeds):
        seed = start_seed + offset
        arrivals = pattern_func(total_time, arrival_prob, seed)
        arrival_counts += np.array([1.0 if patient is not None else 0.0 for patient in arrivals])

    return arrival_counts / max(1, num_seeds)


def smooth_series(values, window):
    """Apply centered moving-average smoothing with edge padding."""
    if window <= 1:
        return values.copy()

    if window % 2 == 0:
        window += 1

    kernel = np.ones(window, dtype=float) / window
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed


def save_probability_csv(probability_by_pattern, total_time, output_csv):
    """Save per-timestep probabilities for each pattern to CSV."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    pattern_names = list(probability_by_pattern.keys())

    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["timestep", "hour", *pattern_names])

        for timestep in range(total_time):
            hour = timestep / 4.0
            row = [timestep + 1, f"{hour:.2f}"]
            row.extend(f"{probability_by_pattern[name][timestep]:.6f}" for name in pattern_names)
            writer.writerow(row)


def plot_probabilities(
    probability_by_pattern,
    total_time,
    arrival_prob,
    num_seeds,
    output_path,
    smoothing_window,
):
    """Create and save a line plot with all patterns on one graph."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    x_hours = np.arange(total_time) / 4.0

    plt.figure(figsize=(12, 7))

    for pattern_name, probabilities in probability_by_pattern.items():
        label = pattern_name.replace("_", " ").title()
        smoothed = smooth_series(probabilities, smoothing_window)
        plt.plot(x_hours, smoothed, linewidth=2.8, label=label, alpha=0.95)

    plt.title(
        f"Estimated Arrival Probability by Pattern\n"
        f"(base arrival_prob={arrival_prob}, seeds per pattern={num_seeds}, smoothing window={smoothing_window} timesteps)",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Simulation Time (hours)", fontsize=11)
    plt.ylabel("Estimated Arrival Probability", fontsize=11)
    plt.ylim(0, 1)
    plt.xlim(0, max(0, total_time / 4.0 - 0.25))
    plt.xticks(np.arange(0, total_time / 4.0 + 0.001, 2.0))
    plt.grid(True, alpha=0.2, linestyle="--")
    plt.legend(loc="upper right", frameon=True, title="Arrival Pattern")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def parse_pattern_list(patterns_arg):
    """Return an ordered list of patterns to include."""
    if not patterns_arg:
        return list(ARRIVAL_PATTERNS.keys())

    requested = [name.strip() for name in patterns_arg.split(",") if name.strip()]
    invalid = [name for name in requested if name not in ARRIVAL_PATTERNS]

    if invalid:
        valid = ", ".join(ARRIVAL_PATTERNS.keys())
        raise ValueError(f"Unknown pattern(s): {invalid}. Valid patterns: {valid}")

    return requested


def main():
    parser = argparse.ArgumentParser(
        description="Plot estimated patient-arrival probability curves for ER arrival patterns."
    )
    parser.add_argument("--total-time", type=int, default=96, help="Total timesteps to simulate (default: 96)")
    parser.add_argument("--arrival-prob", type=float, default=0.3, help="Base arrival probability (default: 0.3)")
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=500,
        help="Number of seeds used to estimate probability per pattern (default: 500)",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=10000,
        help="Starting seed for Monte Carlo estimation (default: 10000)",
    )
    parser.add_argument(
        "--patterns",
        type=str,
        default="",
        help="Comma-separated subset of patterns to plot (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="report_visualizations/arrival_pattern_probabilities/arrival_probability_comparison.png",
        help="Output plot path",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default="report_visualizations/arrival_pattern_probabilities/arrival_probability_comparison.csv",
        help="Output CSV path with per-timestep probabilities",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=9,
        help="Centered moving-average window in timesteps (default: 9; set 1 for no smoothing)",
    )

    args = parser.parse_args()

    if args.total_time <= 0:
        raise ValueError("--total-time must be > 0")
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be > 0")
    if not (0 <= args.arrival_prob <= 1):
        raise ValueError("--arrival-prob must be between 0 and 1")
    if args.smoothing_window <= 0:
        raise ValueError("--smoothing-window must be > 0")

    selected_patterns = parse_pattern_list(args.patterns)

    probability_by_pattern = {}
    print("Estimating arrival probabilities...")

    for pattern_name in selected_patterns:
        pattern_func = ARRIVAL_PATTERNS[pattern_name]
        probabilities = estimate_pattern_probability(
            pattern_func,
            total_time=args.total_time,
            arrival_prob=args.arrival_prob,
            num_seeds=args.num_seeds,
            start_seed=args.start_seed,
        )
        probability_by_pattern[pattern_name] = probabilities

        print(
            f"  {pattern_name}: {get_pattern_description(pattern_name)} | "
            f"mean p={float(np.mean(probabilities)):.3f}"
        )

    plot_probabilities(
        probability_by_pattern=probability_by_pattern,
        total_time=args.total_time,
        arrival_prob=args.arrival_prob,
        num_seeds=args.num_seeds,
        output_path=args.output,
        smoothing_window=args.smoothing_window,
    )
    save_probability_csv(
        probability_by_pattern=probability_by_pattern,
        total_time=args.total_time,
        output_csv=args.csv_output,
    )

    print(f"\nPlot saved to: {args.output}")
    print(f"CSV saved to:  {args.csv_output}")


if __name__ == "__main__":
    main()
