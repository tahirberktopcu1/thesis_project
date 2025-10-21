# Linear Navigator PPO Project

This project trains a forward-only agent that can rotate in place using PPO inside a custom Gymnasium (Gym) environment. A Pygame visualizer shows the agent, its sensors, and the goal in real time.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Training

```bash
python train.py --timesteps 200000
```

Add `--render-training` if you want a live Pygame window while the agent trains (training slows down because rendering runs every step).

TensorBoard logs are written to `logs/` and model checkpoints to `models/`.

## Watching the Trained Agent

```bash
python enjoy.py --model-path models/ppo_linear_navigator.zip
```

`train.py` saves the final agent as `ppo_linear_navigator.zip` and the evaluation callback also stores `best_model.zip`. `enjoy.py` opens a Pygame window with sensor rays and orientation arrows.

## Project Layout

- `navigator/config.py`: Tunable parameters for the environment.
- `navigator/env.py`: Gym environment, observations, and reward shaping.
- `navigator/renderer.py`: Pygame renderer with arrow and sensor visuals.
- `train.py`: PPO learner with evaluation callback setup.
- `enjoy.py`: Playback script to watch the trained agent in human render mode.
