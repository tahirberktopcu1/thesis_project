"""Evaluate trained PPO policy on random Navigator scenarios."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import List

import numpy as np
from stable_baselines3 import PPO

from navigator import LinearNavigatorEnv, NavigatorConfig, RandomObstacleSpec


class EpisodeStats:
    __slots__ = ("success", "steps", "path_length", "reward")

    def __init__(self, success: bool, steps: int, path_length: float, reward: float) -> None:
        self.success = success
        self.steps = steps
        self.path_length = path_length
        self.reward = reward


def config_for_difficulty(name: str) -> NavigatorConfig:
    base = NavigatorConfig(randomize_obstacles=True)
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


def evaluate_policy(model_path: Path, episodes: int, seed: int, difficulty: str) -> List[EpisodeStats]:
    config = config_for_difficulty(difficulty)
    env = LinearNavigatorEnv(config=config, render_mode=None)
    model = PPO.load(model_path)

    stats: List[EpisodeStats] = []

    for episode in range(episodes):
        env_seed = seed + episode
        obs, _ = env.reset(seed=env_seed)
        done = False
        truncated = False
        cumulative_reward = 0.0
        path = 0.0
        prev_pos = env.agent_pos.copy()
        step_count = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            cumulative_reward += reward
            path += np.linalg.norm(env.agent_pos - prev_pos)
            prev_pos = env.agent_pos.copy()
            step_count += 1

        success = done and info.get("terminated_reason") == "goal"
        stats.append(EpisodeStats(success, step_count, float(path), float(cumulative_reward)))

    env.close()
    return stats


def summarize(stats: List[EpisodeStats]) -> dict:
    successes = [s for s in stats if s.success]
    success_rate = len(successes) / len(stats) if stats else 0.0
    mean_steps = np.mean([s.steps for s in stats]) if stats else 0.0
    mean_path = np.mean([s.path_length for s in stats]) if stats else 0.0
    mean_reward = np.mean([s.reward for s in stats]) if stats else 0.0

    print(f"Episodes         : {len(stats)}")
    print(f"Success rate     : {success_rate*100:.1f}% ({len(successes)}/{len(stats)})")
    print(f"Mean steps       : {mean_steps:.1f}")
    print(f"Mean path length : {mean_path:.2f} px")
    print(f"Mean reward      : {mean_reward:.3f}")

    return {
        "episodes": len(stats),
        "success_rate": success_rate,
        "mean_steps": mean_steps,
        "mean_path_length": mean_path,
        "mean_reward": mean_reward,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO policy on Navigator maps")
    parser.add_argument("--model", type=Path, default=Path("models/ppo_linear_navigator.zip"), help="Path to trained PPO model")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=123, help="Random seed base")
    parser.add_argument(
        "--difficulty",
        choices=["warmup", "medium", "hard"],
        default="hard",
        help="Match A* evaluation configs",
    )
    parser.add_argument("--csv", type=Path, help="Optional CSV file to append results")
    parser.add_argument("--tag", type=str, default="ppo", help="Algorithm tag for CSV output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = evaluate_policy(args.model, args.episodes, args.seed, args.difficulty)
    summary = summarize(stats)

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
        "median_path_length": "",
        "mean_steps": summary.get("mean_steps"),
        "mean_reward": summary.get("mean_reward"),
        "mean_nodes_expanded": "",
    }
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()

