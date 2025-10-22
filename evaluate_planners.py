"""Evaluate classical planners (A*) against random Navigator scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import replace
from statistics import mean
from typing import List

import numpy as np

from navigator import LinearNavigatorEnv, NavigatorConfig, RandomObstacleSpec
from planners.astar import AStarResult, plan_with_astar, path_length


def evaluate_astar(
    episodes: int,
    resolution: float,
    seed: int,
    config: NavigatorConfig,
) -> List[AStarResult]:
    env = LinearNavigatorEnv(config=config, render_mode=None)
    results: List[AStarResult] = []

    for episode in range(episodes):
        env_seed = seed + episode
        env.reset(seed=env_seed)
        start = tuple(float(x) for x in env.agent_pos)
        goal = tuple(float(x) for x in env.goal_pos)

        scenario_config = replace(
            env.config,
            randomize_obstacles=False,
            obstacles=tuple(env._obstacles),
            start_pos=start,
            goal_pos=goal,
        )

        result = plan_with_astar(scenario_config, start, goal, resolution=resolution)
        results.append(result)

    env.close()
    return results


def summarize(results: List[AStarResult]) -> dict:
    successes = [res for res in results if res.success]
    success_rate = len(successes) / len(results) if results else 0.0
    lengths = [path_length(res.path) for res in successes]
    expansions = [res.expanded for res in results]

    print(f"Episodes           : {len(results)}")
    print(f"Success rate       : {success_rate*100:.1f}% ({len(successes)}/{len(results)})")
    if successes:
        print(f"Mean path length   : {mean(lengths):.2f} px")
        print(f"Median path length : {np.median(lengths):.2f} px")
    mean_length = mean(lengths) if lengths else float("nan")
    median_length = float(np.median(lengths)) if lengths else float("nan")
    mean_expanded = mean(expansions) if expansions else float("nan")

    print(f"Mean nodes expanded: {mean_expanded:.1f}")

    return {
        "episodes": len(results),
        "success_rate": success_rate,
        "mean_path_length": mean_length,
        "median_path_length": median_length,
        "mean_nodes_expanded": mean_expanded,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate classical planners on Navigator maps")
    parser.add_argument("--episodes", type=int, default=50, help="Number of random scenarios to test")
    parser.add_argument("--resolution", type=float, default=12.0, help="Grid resolution for A* (pixels)")
    parser.add_argument("--seed", type=int, default=123, help="Random seed base for scenario generation")
    parser.add_argument(
        "--difficulty",
        choices=["warmup", "medium", "hard"],
        default="hard",
        help="Match curriculum phase parameters",)
    parser.add_argument("--csv", type=Path, help="Optional CSV file to append results")
    parser.add_argument("--tag", type=str, default="astar", help="Algorithm tag for CSV output")
    return parser.parse_args()


def config_for_difficulty(name: str) -> NavigatorConfig:
    base = NavigatorConfig(
        randomize_obstacles=True,
    )
    if name == "warmup":
        return replace(
            base,
            random_obstacle_spec=RandomObstacleSpec(
                count=4,
                min_size=(36.0, 60.0),
                max_size=(84.0, 144.0),
                min_margin=44.0,
                max_attempts=200,
            ),
            safety_threshold=0.32,
            safety_penalty_gain=0.12,
            forward_block_threshold=0.32,
            corner_margin=8.0,
        )
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


def main() -> None:
    args = parse_args()
    config = config_for_difficulty(args.difficulty)
    results = evaluate_astar(args.episodes, args.resolution, args.seed, config)
    summary = summarize(results)

    if args.csv:
        write_csv(args.csv, args.tag, args.difficulty, summary)


def write_csv(csv_path: Path, algorithm: str, difficulty: str, summary: dict) -> None:
    import csv

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "algorithm",
        "difficulty",
        "episodes",
        "success_rate",
        "mean_path_length",
        "median_path_length",
        "mean_steps",
        "mean_reward",
        "mean_nodes_expanded",
    ]
    row = {
        "algorithm": algorithm,
        "difficulty": difficulty,
        "episodes": summary.get("episodes"),
        "success_rate": summary.get("success_rate"),
        "mean_path_length": summary.get("mean_path_length"),
        "median_path_length": summary.get("median_path_length"),
        "mean_steps": "",
        "mean_reward": "",
        "mean_nodes_expanded": summary.get("mean_nodes_expanded"),
    }
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()


