# Step 3 – Agent and Reward Design

This step satisfies the requirement “Define the robot’s state (position, distance to goal, obstacle sensors) and action space (forward, turn left/right). Construct a reward function balancing path efficiency and safety.”

## 3.1 State Representation
As described in Step 2, each observation vector contains:
1. Normalized agent position (x, y).
2. Heading unit vector (cos θ, sin θ).
3. Normalized distance to goal.
4. Goal direction unit vector.
5. 24 normalized ray sensors (360° proximity).

These features provide the PPO policy with:
- Spatial awareness (agent vs. goal).
- Orientation information to align heading.
- Obstacle proximity from all directions.

## 3.2 Action Space
- Discrete actions: forward, turn left, turn right.
- Commands directly affect position/heading; no continuous throttle to keep training manageable.
- Action space matches the specification and parallels many navigation benchmarks where mobile robots alternate between moving forward and rotating in place.

## 3.3 Reward Function
Designed to balance efficiency and safety:
- **Progress reward:** `0.25 × (prev_distance - new_distance)` encourages goal-directed movement.
- **Time penalty:** `0.01` for forward, `0.005` for turns to reduce dithering.
- **Alignment bonus:** `0.02 × dot(heading, goal_dir)` keeps the agent oriented toward the goal.
- **Safety penalty:** If the minimum sensor value < 35% of range, apply `0.2 × (threshold - min_sensor_norm)`; discourages hugging obstacles.
- **Directional heuristic:** If forward ray is blocked and a side ray is clearer, turning towards the free side grants a small bonus (`turn_bonus_gain = 0.03`).
- **Oscillation penalty:** Detects alternating left/right actions with negligible displacement; penalty grows with duration (`idle_penalty = 0.08`).
- **Negative progress penalty:** Consecutive increases in goal distance yield incremental penalties to discourage backtracking.
- **Terminal rewards:** `+1.0` for reaching the goal, `-1.0` (collision_penalty) on impact. Episodes also truncate after `max_episode_steps`.

## 3.4 Design Rationale
- **Efficiency vs. Safety:** Progress incentive combined with safety penalty ensures the agent seeks shortest routes while maintaining distance from obstacles.
- **Loop avoidance:** Idle/oscillation penalties break local minima (agent spinning in place).
- **Directional heuristic:** Encourages proactive turning instead of blindly moving forward when the path is blocked.
- **Reward shaping stability:** Terms are scaled modestly so that terminal rewards still dominate; PPO converges without instability.

## 3.5 Implementation References
- All reward logic resides in `navigator/env.py` within the `step` method.
- Sensor calculations (`_sensor_distances`, `_directional_sensor_fraction`) support heuristics without exposing implementation complexity to the policy.
- Default parameters are configurable via `NavigatorConfig` for ablation studies (e.g., adjusting penalty gains).

This completes Step 3 by providing a well-defined state, discrete action space, and reward formulation that aligns with the project objectives.***
