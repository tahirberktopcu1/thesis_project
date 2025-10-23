# Step 2 - Environment Design

This step documents the simulated environment created for the thesis, aligned with the requirements "Create a simulated environment using PyBullet or OpenAI Gym. Define start, goal, and obstacle distributions."

## 2.1 Platform and Implementation
- **Framework:** Custom OpenAI Gymnasium environment (`navigator/env.py`) with `gym.Env` interface.
- **Rendering:** Pygame-based renderer (`navigator/renderer.py`) supports both human display and `rgb_array` outputs for debugging and qualitative inspection.
- **Coordinate system:** 2D top-down plane measured in pixels. Agent moves kinodynamically without inertia (simplified kinematics).

## 2.2 Workspace Geometry
- **Map size:** `NavigatorConfig.map_width = 768`, `map_height = 576`.
- **Borders:** The renderer draws a rectangular frame; motion logic clamps agent positions so that the agent's circular footprint (radius 12 px) stays inside the arena.
- **Clearances:** `corner_margin = 10` px used when checking forward motion, preventing glancing collisions.

## 2.3 Obstacles
- **Primitive:** Axis-aligned rectangles (`RectangleObstacle` dataclass).
- **Procedural generation:** `_generate_random_obstacles` samples rectangles using rejection sampling with:
  - `count` (e.g., 6 for default training, 8 for "hard" difficulty).
  - `min_size` / `max_size` for width/height bounds.
  - `min_margin` to enforce minimum spacing between obstacles and from borders (inflated by agent radius).
  - `max_attempts` to avoid infinite loops when maps become crowded.
- **Static defaults:** Three predefined obstacles retained for deterministic testing when `randomize_obstacles=False`.

## 2.4 Start and Goal Configurations
- **Default positions:** Start near the left edge (`(radius + 10, map_height / 2)`), goal near the right edge (`(map_width - goal_radius - 10, map_height / 2)`).
- **Randomization:** Currently disabled to simplify analysis, but the environment supports reintroducing random start-goal bands if needed (see history of `NavigatorConfig`).

## 2.5 Observation Model
Observation vector (float32) length = 31:
1. **Agent position** normalized to `[-1, 1]`: `(2x / width - 1, 2y / height - 1)`.
2. **Heading unit vector:** `(cos theta, sin theta)`.
3. **Normalized distance to goal:** Euclidean distance divided by map diagonal.
4. **Goal direction unit vector:** Unit vector from agent to goal.
5. **Range sensors:** 24 ray-cast distances equally spaced over 360 degrees, normalized by `sensor_range = 240`. Rays account for obstacles and borders.

## 2.6 Action Space and Dynamics
- **Action space:** `Discrete(3)` -> `0: forward`, `1: turn left`, `2: turn right`.
- **Speeds:** `forward_speed = 6 px/step`, `turn_speed = pi/14 rad/step`.
- **Forward safety check:** `_move_forward` computes available clearance along heading; if remaining distance is below threshold after considering borders/obstacles, the move is shortened or cancelled.
- **Episode limit:** `max_episode_steps = 400`; collision or goal also terminate the episode.

## 2.7 Summary
The custom Gymnasium environment satisfies Step 2 by providing:
- A reproducible 2D workspace with configurable obstacles.
- Normalized observations covering position, orientation, goal information, and 360 degrees proximity sensors.
- A discrete action space mapping to simple robot motion primitives.
- Safety-aware forward dynamics acknowledging the agent's footprint.

This foundation supports subsequent steps (agent design, training, evaluation) without additional assumptions.***
