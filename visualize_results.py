#!/usr/bin/env python
"""Generate simple comparison plots from baseline CSV results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

MetricRow = Dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise baseline results")
    parser.add_argument("--csv", type=Path, default=Path("results/baselines.csv"), help="Input CSV file")
    parser.add_argument("--out-dir", type=Path, default=Path("results/figures"), help="Directory to save plots")
    parser.add_argument("--difficulty", type=str, default="hard", help="Difficulty to filter (e.g., hard)")
    return parser.parse_args()


def load_rows(csv_path: Path, difficulty: str) -> List[MetricRow]:
    rows: List[MetricRow] = []
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for raw in reader:
            if raw.get("difficulty") != difficulty:
                continue
            row: MetricRow = {}
            for key, value in raw.items():
                if key in {"algorithm", "difficulty"}:
                    row[key] = value
                elif value:
                    try:
                        row[key] = float(value)
                    except ValueError:
                        row[key] = value
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {csv_path} for difficulty '{difficulty}'")
    return rows


def plot_bar(ax, algorithms, values, title, ylabel, ylim=None):
    ax.bar(algorithms, values, color=["#4c72b0", "#dd8452", "#55a868", "#c44e52"][: len(values)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv, args.difficulty)

    by_algo: Dict[str, MetricRow] = {}
    for row in rows:
        algo = row.get("algorithm", "unknown")
        by_algo[algo] = row

    algorithms = list(by_algo.keys())
    success = [by_algo[a].get("success_rate", 0.0) * 100 for a in algorithms]
    path_lengths = [by_algo[a].get("mean_path_length", 0.0) for a in algorithms]
    steps = [by_algo[a].get("mean_steps", 0.0) for a in algorithms]
    rewards = [by_algo[a].get("mean_reward", 0.0) for a in algorithms]
    expansions = [by_algo[a].get("mean_nodes_expanded", 0.0) for a in algorithms]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    plot_bar(axes[0][0], algorithms, success, "Success Rate", "%", ylim=(0, 105))
    plot_bar(axes[0][1], algorithms, path_lengths, "Mean Path Length", "pixels")

    if any(steps):
        plot_bar(axes[1][0], algorithms, steps, "Mean Steps", "steps")
    else:
        axes[1][0].axis("off")
        axes[1][0].set_title("Mean Steps (not available)")

    if any(rewards):
        plot_bar(axes[1][1], algorithms, rewards, "Mean Reward", "reward units")
    else:
        plot_bar(axes[1][1], algorithms, expansions, "Mean Nodes Expanded", "nodes")

    fig.suptitle(f"Baseline Comparison - Difficulty: {args.difficulty}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = args.out_dir / f"comparison_{args.difficulty}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved comparison plot to {out_path}")


if __name__ == "__main__":
    main()

