#!/usr/bin/env python
"""Run a PPO policy (or random agent) and plot the resulting trajectory."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Iterable
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from navigator import LinearNavigatorEnv, NavigatorConfig, RectangleObstacle, RandomObstacleSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot trajectory in the Navigator environment")
    parser.add_argument("--model", type=Path, default=Path("models/ppo_linear_navigator.zip"), help="Path to PPO model (omit for random policy)")
    parser.add_argument("--seed", type=int, default=42, help="Environment seed for reproducibility")
    parser.add_argument("--difficulty", choices=["warmup", "medium", "hard"], default="hard", help="Config profile")
    parser.add_argument("--max-steps", type=int, default=400, help="Maximum rollout length")
    parser.add_argument("--out", type=Path, default=Path("results/figures/trajectory.png"), help="Output plot path")
    parser.add_argument("--random", action="store_true", help="Ignore model and use random actions")
    return parser.parse_args()


def config_for_difficulty(name: str) -> NavigatorConfig:
    base = NavigatorConfig(randomize_obstacles=True)
    if name == "warmup":
        return base
    if name == "medium":
        return replace(
            base,
            random_obstacle_spec=RandomObstacleSpec(
                count=6,
                min_size=(42.0, 72.0),
                max_size=(108.0, 192.0),
                min_margin=48.0,
                max_attempts=220,
            ),
            safety_threshold=0.35,
            safety_penalty_gain=0.18,
            forward_block_threshold=0.28,
            corner_margin=10.0,
        )
    if name == "hard":
        return replace(
            base,
            random_obstacle_spec=RandomObstacleSpec(
                count=8,
                min_size=(48.0, 84.0),
                max_size=(132.0, 220.0),
                min_margin=56.0,
                max_attempts=240,
            ),
            safety_threshold=0.36,
            safety_penalty_gain=0.2,
            forward_block_threshold=0.26,
            corner_margin=12.0,
            turn_bonus_gain=0.04,
            idle_penalty=0.1,
        )
    raise ValueError(f"Unknown difficulty '{name}'")


def plot_obstacles(ax, obstacles: Iterable[RectangleObstacle]) -> None:
    for obs in obstacles:
        x_min, y_min, x_max, y_max = obs.as_bounds()
        ax.add_patch(
            plt.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                facecolor="#c44e52",
                alpha=0.4,
                edgecolor="#8c1d3a",
            )
        )


def main() -> None:
    args = parse_args()
    config = config_for_difficulty(args.difficulty)
    env = LinearNavigatorEnv(config=config, render_mode=None)

    model = None
    if not args.random and args.model.exists():
        model = PPO.load(args.model)

    obs, _ = env.reset(seed=args.seed)
    positions = [env.agent_pos.copy()]
    done = False
    truncated = False
    steps = 0

    while not (done or truncated) and steps < args.max_steps:
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        positions.append(env.agent_pos.copy())
        steps += 1

    positions = np.array(positions)
    start = positions[0]
    goal = env.goal_pos

    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_obstacles(ax, env._obstacles)
    ax.plot(positions[:, 0], positions[:, 1], color="#4c72b0", linewidth=2, label="Trajectory")
    ax.scatter([start[0]], [start[1]], color="#55a868", s=80, marker="o", label="Start")
    ax.scatter([goal[0]], [goal[1]], color="#dd8452", s=80, marker="*", label="Goal")
    ax.set_xlim(0, env.config.map_width)
    ax.set_ylim(0, env.config.map_height)
    ax.set_aspect("equal")
    ax.set_title(f"Trajectory ({'random' if model is None else 'PPO'}) - seed={args.seed}")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved trajectory plot to {args.out}")


if __name__ == "__main__":
    main()


