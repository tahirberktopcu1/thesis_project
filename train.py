"""PPO training script for the linear navigator agent."""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from navigator import NavigatorConfig, LinearNavigatorEnv


def make_env(
    config: NavigatorConfig,
    render_mode: str | None = None,
    seed: int | None = None,
):
    def _init():
        env_config = replace(config)
        env = LinearNavigatorEnv(config=env_config, render_mode=render_mode)
        wrapped = Monitor(env)
        wrapped.reset(seed=seed)
        return wrapped

    return _init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear Navigator PPO training")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Total training steps")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"), help="TensorBoard and monitor directory")
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="Directory where models are saved")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--render-training",
        action="store_true",
        help="Open a Pygame window and render the agent live during training (slows training).",
    )
    parser.add_argument(
        "--random-map",
        action="store_true",
        help="Randomize obstacle placement on every episode reset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    base_config = NavigatorConfig(randomize_obstacles=args.random_map)

    train_render_mode = "human" if args.render_training else None
    env = DummyVecEnv([make_env(base_config, render_mode=train_render_mode, seed=args.seed)])
    eval_env = DummyVecEnv([make_env(base_config, render_mode=None, seed=args.seed + 1)])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=str(args.log_dir),
        seed=args.seed,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.model_dir),
        log_path=str(args.log_dir),
        eval_freq=5_000,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    model_path = args.model_dir / "ppo_linear_navigator"
    model.save(model_path)

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
