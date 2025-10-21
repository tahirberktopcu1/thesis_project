# Linear Navigator PPO Project

This project trains a forward-only agent that can rotate in place using PPO inside a custom Gymnasium (Gym) environment. The arena now contains static rectangular obstacles and boundary walls; the agent must weave through them using 24 ray sensors that cover the full 360 degrees. Touching any obstacle or wall immediately ends the episode with a configurable collision penalty. A Pygame visualizer shows the agent, its sensors (color coded by proximity), and the goal in real time.

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

Add `--render-training` if you want a live Pygame window while the agent trains (training slows down because rendering runs every step). Pass `--random-map` to shuffle obstacle layouts every episode instead of using the static defaults.

> Tip: For the best results on highly cluttered maps use a longer run, e.g. `python train.py --timesteps 500000 --random-map`. The default script trains PPO with a `[128, 128]` MLP, `n_steps=4096`, `batch_size=1024`, and a slight entropy bonus to keep exploration high.

TensorBoard logs are written to `logs/` and model checkpoints to `models/`.

## Watching the Trained Agent

```bash
python enjoy.py --model-path models/ppo_linear_navigator.zip
```

`train.py` saves the final agent as `ppo_linear_navigator.zip` and the evaluation callback also stores `best_model.zip`. `enjoy.py` opens a Pygame window with sensor rays and orientation arrows. Use `--random-map` (and optionally `--seed`) to replay the policy in the same randomized world distribution as training.

## Dynamic Maps

- Toggle random obstacle generation via CLI (`--random-map`) or directly through `NavigatorConfig`.
- `NavigatorConfig.random_obstacle_spec` lets you adjust the count, size range, margin, and sampling attempts for random rectangles.
- When random maps are enabled, every obstacle is sampled procedurally (static defaults are ignored unless `keep_static_when_random=True`). All rectangles respect safety margins around the borders, start, and goal so the agent always has a viable route.

## Reward Shaping

- Progress toward the goal is rewarded each step (scaled by 0.25); moving away produces an equal-magnitude penalty.
- Every action carries a small time cost (forward > turn) plus a minor orientation bonus for facing the goal.
- Safety is enforced via the closest distance sensor: if the reading drops below 35 % of the range, a proportional penalty is applied and forward motion is discouraged in favor of turning toward free space.
- Only genuine loops are penalized—if the agent alternates left/right without translating for several steps, the idle penalty grows; purposeful turns that involve movement remain unpenalized. Repeated negative progress is also discouraged.
- Colliding with the outer frame or any obstacle ends the episode immediately with an additional penalty; reaching the goal grants a success bonus.

## Project Layout

- `navigator/config.py`: Tunable parameters for the environment.
- `navigator/env.py`: Gym environment, observations, and reward shaping with obstacle collisions.
- `navigator/renderer.py`: Pygame renderer (obstacles, arrow indicator, and sensor rays with distance-based colors).
- `train.py`: PPO learner with evaluation callback setup (now supports randomized maps).
- `enjoy.py`: Playback script to watch the trained agent in human render mode.


