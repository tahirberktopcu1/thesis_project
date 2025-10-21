# Linear Navigator PPO Project

This project trains a forward-only agent that can rotate in place using PPO inside a custom Gymnasium (Gym) environment. The arena now contains static rectangular obstacles and boundary walls; the agent must weave through them using 360° distance sensors. Touching any obstacle or wall immediately ends the episode with a configurable collision penalty. A Pygame visualizer shows the agent, its sensors (color coded by proximity), and the goal in real time.

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

## Reward Shaping

- Progress toward the goal is rewarded each step; moving closer yields positive reward, while drifting away incurs a loss.
- Turning actions carry only a light time penalty, encouraging exploration instead of the agent driving straight.
- The agent receives a small bonus for pointing toward the goal and a quadratic penalty when any sensor detects close obstacles, pushing it to steer away from walls.
- Colliding with the outer frame or any obstacle ends the episode immediately with an additional penalty; reaching the goal grants a success bonus.

## Project Layout

- `navigator/config.py`: Tunable parameters for the environment.
- `navigator/env.py`: Gym environment, observations, and reward shaping with obstacle collisions.
- `navigator/renderer.py`: Pygame renderer (obstacles, arrow indicator, and sensor rays with distance-based colors).
- `train.py`: PPO learner with evaluation callback setup.
- `enjoy.py`: Playback script to watch the trained agent in human render mode.
