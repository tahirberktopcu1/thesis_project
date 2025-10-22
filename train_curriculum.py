"""Curriculum training script for the Linear Navigator agent."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from navigator import LinearNavigatorEnv, NavigatorConfig, RandomObstacleSpec


@dataclass(frozen=True)
class CurriculumPhase:
    """Defines a curriculum phase with environment overrides and step budget."""

    name: str
    timesteps: int
    overrides: Mapping[str, object]

    def with_timesteps(self, timesteps: int) -> "CurriculumPhase":
        return CurriculumPhase(self.name, timesteps, self.overrides)


DEFAULT_PHASES: Sequence[CurriculumPhase] = (
    CurriculumPhase(
        name="warmup",
        timesteps=200_000,
        overrides={
            "randomize_obstacles": True,
            "randomize_start_goal": False,
            "random_obstacle_spec": RandomObstacleSpec(
                count=4,
                min_size=(36.0, 60.0),
                max_size=(84.0, 144.0),
                min_margin=44.0,
                max_attempts=200,
            ),
            "safety_threshold": 0.32,
            "safety_penalty_gain": 0.12,
            "forward_block_threshold": 0.32,
            "corner_margin": 8.0,
        },
    ),
    CurriculumPhase(
        name="medium",
        timesteps=200_000,
        overrides={
            "randomize_obstacles": True,
            "randomize_start_goal": True,
            "random_obstacle_spec": RandomObstacleSpec(
                count=6,
                min_size=(42.0, 72.0),
                max_size=(108.0, 192.0),
                min_margin=48.0,
                max_attempts=220,
            ),
            "safety_threshold": 0.35,
            "safety_penalty_gain": 0.18,
            "forward_block_threshold": 0.28,
            "corner_margin": 10.0,
            "start_x_range": (0.05, 0.22),
            "goal_x_range": (0.78, 0.95),
        },
    ),
    CurriculumPhase(
        name="hard",
        timesteps=200_000,
        overrides={
            "randomize_obstacles": True,
            "randomize_start_goal": True,
            "random_obstacle_spec": RandomObstacleSpec(
                count=8,
                min_size=(48.0, 84.0),
                max_size=(132.0, 220.0),
                min_margin=56.0,
                max_attempts=240,
            ),
            "safety_threshold": 0.36,
            "safety_penalty_gain": 0.2,
            "forward_block_threshold": 0.26,
            "corner_margin": 12.0,
            "turn_bonus_gain": 0.04,
            "idle_penalty": 0.1,
            "start_x_range": (0.03, 0.22),
            "goal_x_range": (0.78, 0.97),
            "start_goal_margin": 48.0,
        },
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curriculum PPO training for the navigator agent")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"), help="TensorBoard log root directory")
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="Directory to store checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--render-training",
        action="store_true",
        help="Render training environment during each phase (slows training).",
    )
    parser.add_argument(
        "--phases",
        type=str,
        help="Comma-separated subset of phases to run (default: all phases).",
    )
    parser.add_argument(
        "--phase-steps",
        type=str,
        help="Comma-separated timesteps per selected phase (overrides defaults).",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10_000,
        help="Number of environment steps between evaluations.",
    )
    return parser.parse_args()


def select_phases(args: argparse.Namespace) -> Sequence[CurriculumPhase]:
    phases = list(DEFAULT_PHASES)
    if args.phases:
        requested = [name.strip() for name in args.phases.split(",") if name.strip()]
        phase_map = {phase.name: phase for phase in phases}
        phases = [phase_map[name] for name in requested if name in phase_map]
    if args.phase_steps:
        step_values = [int(value.strip()) for value in args.phase_steps.split(",") if value.strip()]
        if len(step_values) != len(phases):
            raise ValueError("--phase-steps length must match the number of selected phases")
        phases = [phase.with_timesteps(steps) for phase, steps in zip(phases, step_values)]
    return phases


def make_env(config: NavigatorConfig, render_mode: str | None, seed: int | None):
    phase_config = replace(config)

    def _init():
        env = LinearNavigatorEnv(config=phase_config, render_mode=render_mode)
        wrapped = Monitor(env)
        wrapped.reset(seed=seed)
        return wrapped

    return _init


def apply_overrides(base: NavigatorConfig, overrides: Mapping[str, object]) -> NavigatorConfig:
    return replace(base, **dict(overrides))


def build_model(env: DummyVecEnv, log_dir: Path, seed: int) -> PPO:
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    return PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=str(log_dir),
        seed=seed,
        n_steps=4096,
        batch_size=1024,
        learning_rate=3e-4,
        gae_lambda=0.95,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
    )


def train_curriculum(args: argparse.Namespace) -> None:
    phases = select_phases(args)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    base_config = NavigatorConfig(randomize_obstacles=True)
    model: PPO | None = None
    total_steps = 0

    for idx, phase in enumerate(phases):
        phase_config = apply_overrides(base_config, phase.overrides)
        phase_log_dir = args.log_dir / phase.name
        phase_model_dir = args.model_dir / phase.name
        os.makedirs(phase_log_dir, exist_ok=True)
        os.makedirs(phase_model_dir, exist_ok=True)

        train_seed = args.seed + idx * 1000
        eval_seed = args.seed + idx * 1000 + 500

        render_mode = "human" if args.render_training else None
        train_env = DummyVecEnv([make_env(phase_config, render_mode, train_seed)])
        eval_env = DummyVecEnv([make_env(phase_config, None, eval_seed)])

        if model is None:
            model = build_model(train_env, args.log_dir, args.seed)
        else:
            model.set_env(train_env)

        callback = EvalCallback(
            eval_env,
            best_model_save_path=str(phase_model_dir),
            log_path=str(phase_log_dir),
            eval_freq=args.eval_freq,
            deterministic=True,
            render=False,
        )

        model.learn(
            total_timesteps=phase.timesteps,
            callback=callback,
            reset_num_timesteps=False,
            tb_log_name=f"phase_{phase.name}",
        )

        total_steps += phase.timesteps
        model.save(args.model_dir / f"ppo_curriculum_{phase.name}")

        train_env.close()
        eval_env.close()

    if model is not None:
        model.save(args.model_dir / "ppo_curriculum_final")
    print(f"Curriculum training complete after {total_steps} steps across {len(phases)} phases.")


def main() -> None:
    args = parse_args()
    train_curriculum(args)


if __name__ == "__main__":
    main()
