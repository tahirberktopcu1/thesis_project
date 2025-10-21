"""Visual playback utility for the trained navigator agent."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO

from navigator import NavigatorConfig, LinearNavigatorEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch the trained navigator agent")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/ppo_linear_navigator.zip"),
        help="Path to the PPO model to load (EvalCallback saves best model as .zip)",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to watch")
    parser.add_argument("--sleep", type=float, default=0.02, help="Delay between steps in seconds")
    parser.add_argument(
        "--random-map",
        action="store_true",
        help="Randomize obstacle placement on reset (must match training configuration).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible playback.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {args.model_path}. Run `python train.py` first."
        )

    model = PPO.load(args.model_path)
    config = NavigatorConfig(randomize_obstacles=args.random_map)
    env = LinearNavigatorEnv(config=config, render_mode="human")

    try:
        seed = args.seed
        for _ in range(args.episodes):
            if seed is not None:
                obs, _info = env.reset(seed=seed)
                seed = None
            else:
                obs, _info = env.reset()
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action, _state = model.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, _info = env.step(int(action))
                time.sleep(args.sleep)
    finally:
        env.close()


if __name__ == "__main__":
    main()
