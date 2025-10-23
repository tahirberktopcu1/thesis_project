#!/usr/bin/env python
"""Plot PPO training curves from TensorBoard scalar logs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training curve from TensorBoard logs")
    parser.add_argument("--logdir", type=Path, default=Path("logs"), help="TensorBoard log directory")
    parser.add_argument("--tag", type=str, default="rollout/ep_rew_mean", help="Scalar tag to plot")
    parser.add_argument("--out", type=Path, default=Path("results/figures/training_curve.png"), help="Output PNG path")
    parser.add_argument("--smooth", type=int, default=10, help="SMA window for smoothing")
    return parser.parse_args()


def load_scalars(logdir: Path, tag: str) -> tuple[list[float], list[float]]:
    if not logdir.exists():
        raise FileNotFoundError(f"Log directory not found: {logdir}")

    event_files = list(logdir.rglob("events.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {logdir}")

    steps: list[float] = []
    values: list[float] = []

    for event_file in sorted(event_files):
        acc = EventAccumulator(str(event_file))
        acc.Reload()
        if tag not in acc.Tags().get("scalars", []):
            continue
        scalar_events = acc.Scalars(tag)
        steps.extend([event.step for event in scalar_events])
        values.extend([event.value for event in scalar_events])

    if not steps:
        raise ValueError(f"Tag '{tag}' not found in TensorBoard logs under {logdir}")
    return steps, values


def smooth(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    smoothed = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        slice_ = values[start : idx + 1]
        smoothed.append(sum(slice_) / len(slice_))
    return smoothed


def main() -> None:
    args = parse_args()
    steps, values = load_scalars(args.logdir, args.tag)
    values_smoothed = smooth(values, args.smooth)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, values, label="Raw", alpha=0.4)
    plt.plot(steps, values_smoothed, label=f"SMA (window={args.smooth})", linewidth=2)
    plt.xlabel("Timesteps")
    plt.ylabel(args.tag)
    plt.title(f"Training Curve - {args.tag}")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved training curve to {args.out}")


if __name__ == "__main__":
    main()
