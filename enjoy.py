"""Visual playback utility for the trained navigator agent."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO

from navigator.env import LinearNavigatorEnv


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {args.model_path}. Run `python train.py` first."
        )

    model = PPO.load(args.model_path)
    env = LinearNavigatorEnv(render_mode="human")

    try:
        for _ in range(args.episodes):
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
