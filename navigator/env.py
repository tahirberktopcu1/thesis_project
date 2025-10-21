"""Custom OpenAI Gym environment for a simple forward-moving navigator agent."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import NavigatorConfig, RectangleObstacle
from .renderer import NavigatorRenderer, RenderFrame


class LinearNavigatorEnv(gym.Env):
    """A minimal navigation task where the agent must reach the opposite side."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self, config: Optional[NavigatorConfig] = None, render_mode: Optional[str] = None
    ) -> None:
        super().__init__()
        self.config = config or NavigatorConfig()
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(3)  # forward, turn left, turn right
        self._obstacles: Tuple[RectangleObstacle, ...] = self.config.resolved_obstacles()
        self._sensor_count = len(self.config.sensor_angles)

        base_low = np.array([-1.0, -1.0, -1.0, -1.0, 0.0], dtype=np.float32)
        base_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        sensor_low = np.zeros(self._sensor_count, dtype=np.float32)
        sensor_high = np.ones(self._sensor_count, dtype=np.float32)
        obs_low = np.concatenate([base_low, sensor_low])
        obs_high = np.concatenate([base_high, sensor_high])
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self.agent_pos = np.zeros(2, dtype=np.float32)
        self.agent_angle = 0.0
        self.goal_pos = np.zeros(2, dtype=np.float32)
        self._step_count = 0
        self._renderer: Optional[NavigatorRenderer] = None
        self._max_goal_distance = math.hypot(self.config.map_width, self.config.map_height)
        self._last_sensor_distances: Tuple[float, ...] = tuple(0.0 for _ in range(self._sensor_count))

        self.reset(seed=None)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0
        self.agent_pos = np.array(self.config.resolved_start(), dtype=np.float32)
        self.agent_angle = 0.0
        self.goal_pos = np.array(self.config.resolved_goal(), dtype=np.float32)
        self._last_sensor_distances = self._sensor_distances()

        if self.render_mode == "human":
            self._ensure_renderer()
            if self._renderer:
                self._renderer.reset()
                self.render()

        observation = self._get_observation()
        info = {"goal_position": self.goal_pos.copy()}
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action), f"{action} gecersiz eylem"
        prev_distance = self._distance_to_goal()

        collision = False
        if action == 0:  # move forward
            collision = self._move_forward()
        elif action == 1:  # turn left
            self.agent_angle += self.config.turn_speed
        elif action == 2:  # turn right
            self.agent_angle -= self.config.turn_speed

        self.agent_angle = self._wrap_angle(self.agent_angle)
        self._step_count += 1

        observation = self._get_observation()
        distance = self._distance_to_goal()
        distance_reward = (prev_distance - distance) * 0.1
        reward = distance_reward - 0.01

        if collision:
            reward -= 0.1

        reached_goal = distance <= self.config.goal_radius
        if reached_goal:
            reward += 1.0

        terminated = reached_goal
        truncated = self._step_count >= self.config.max_episode_steps

        if self.render_mode == "human":
            self.render()

        info: Dict[str, Any] = {
            "distance_to_goal": distance,
            "step_count": self._step_count,
            "collision": collision,
        }
        return observation, float(reward), terminated, truncated, info

    def render(self) -> Optional[RenderFrame]:
        if self.render_mode is None:
            raise RuntimeError("render_mode ayarlanmadi, render cagrisi yapilamaz.")

        self._ensure_renderer()
        if self._renderer is None:
            return None

        frame = self._renderer.draw(
            agent_pos=tuple(self.agent_pos),
            agent_angle=self.agent_angle,
            goal_pos=tuple(self.goal_pos),
            sensor_rays=self._sensor_rays(),
            obstacles=tuple(obstacle.as_bounds() for obstacle in self._obstacles),
        )
        if self.render_mode == "human":
            return None
        return frame

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        super().close()

    # --- Internal helpers -------------------------------------------------

    def _ensure_renderer(self) -> None:
        if self.render_mode is None:
            return
        if self._renderer is None:
            if self.render_mode not in self.metadata["render_modes"]:
                raise ValueError(f"Bilinmeyen render_mode: {self.render_mode}")
            self._renderer = NavigatorRenderer(
                width=self.config.map_width,
                height=self.config.map_height,
                agent_radius=self.config.agent_radius,
                goal_radius=self.config.goal_radius,
                render_mode=self.render_mode,
            )

    def _move_forward(self) -> bool:
        dx = math.cos(self.agent_angle) * self.config.forward_speed
        dy = math.sin(self.agent_angle) * self.config.forward_speed
        candidate = self.agent_pos + np.array([dx, dy], dtype=np.float32)
        collision = False

        clamped_x = np.clip(
            candidate[0],
            self.config.agent_radius,
            self.config.map_width - self.config.agent_radius,
        )
        clamped_y = np.clip(
            candidate[1],
            self.config.agent_radius,
            self.config.map_height - self.config.agent_radius,
        )

        if not (math.isclose(candidate[0], clamped_x) and math.isclose(candidate[1], clamped_y)):
            collision = True

        new_pos = np.array([clamped_x, clamped_y], dtype=np.float32)
        if self._collides_with_obstacle(new_pos):
            collision = True
            new_pos = self.agent_pos  # stay in place on obstacle collision

        self.agent_pos = new_pos
        return collision

    def _get_observation(self) -> np.ndarray:
        self._last_sensor_distances = self._sensor_distances()
        sensor_readings = np.array(
            [np.clip(dist / self.config.sensor_range, 0.0, 1.0) for dist in self._last_sensor_distances],
            dtype=np.float32,
        )
        normalized_pos = np.array(
            [
                (self.agent_pos[0] / self.config.map_width) * 2.0 - 1.0,
                (self.agent_pos[1] / self.config.map_height) * 2.0 - 1.0,
            ],
            dtype=np.float32,
        )
        orientation = np.array(
            [math.cos(self.agent_angle), math.sin(self.agent_angle)], dtype=np.float32
        )
        distance_norm = np.array(
            [np.clip(self._distance_to_goal() / self._max_goal_distance, 0.0, 1.0)],
            dtype=np.float32,
        )
        observation = np.concatenate([normalized_pos, orientation, distance_norm, sensor_readings])
        return observation.astype(np.float32)

    def _sensor_distances(self) -> Tuple[float, ...]:
        distances = []
        for rel_angle in self.config.sensor_angles:
            angle = self.agent_angle + rel_angle
            boundary_dist = self._distance_to_boundary(angle)
            obstacle_dist = self._distance_to_obstacles(angle)
            distances.append(min(boundary_dist, obstacle_dist, self.config.sensor_range))
        return tuple(distances)

    def _sensor_rays(
        self,
    ) -> Tuple[Tuple[Tuple[float, float], Tuple[float, float], float], ...]:
        if not self._last_sensor_distances:
            self._last_sensor_distances = self._sensor_distances()
        rays = []
        for rel_angle, distance in zip(self.config.sensor_angles, self._last_sensor_distances):
            angle = self.agent_angle + rel_angle
            length = min(distance, self.config.sensor_range)
            end_x = self.agent_pos[0] + math.cos(angle) * length
            end_y = self.agent_pos[1] + math.sin(angle) * length
            normalized = np.clip(distance / self.config.sensor_range, 0.0, 1.0)
            rays.append(
                (
                    (self.agent_pos[0], self.agent_pos[1]),
                    (end_x, end_y),
                    float(normalized),
                )
            )
        return tuple(rays)

    def _distance_to_boundary(self, angle: float) -> float:
        dx = math.cos(angle)
        dy = math.sin(angle)
        distances = []

        if abs(dx) > 1e-6:
            if dx > 0:
                dist_x = (self.config.map_width - self.config.agent_radius - self.agent_pos[0]) / dx
            else:
                dist_x = (self.agent_pos[0] - self.config.agent_radius) / -dx
            if dist_x >= 0:
                distances.append(dist_x)

        if abs(dy) > 1e-6:
            if dy > 0:
                dist_y = (self.config.map_height - self.config.agent_radius - self.agent_pos[1]) / dy
            else:
                dist_y = (self.agent_pos[1] - self.config.agent_radius) / -dy
            if dist_y >= 0:
                distances.append(dist_y)

        if not distances:
            return self.config.sensor_range
        return float(min(min(distances), self.config.sensor_range))

    def _distance_to_obstacles(self, angle: float) -> float:
        dx = math.cos(angle)
        dy = math.sin(angle)
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return self.config.sensor_range

        min_distance = self.config.sensor_range
        for obstacle in self._obstacles:
            distance = self._ray_rectangle_intersection(self.agent_pos, (dx, dy), obstacle)
            if distance is not None and 0.0 <= distance < min_distance:
                min_distance = distance
        return min_distance

    def _distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.goal_pos - self.agent_pos))

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle <= -math.pi:
            angle += 2 * math.pi
        return angle

    def _collides_with_obstacle(self, position: np.ndarray) -> bool:
        radius = self.config.agent_radius
        px, py = float(position[0]), float(position[1])
        for obstacle in self._obstacles:
            x_min, y_min, x_max, y_max = obstacle.as_bounds()
            nearest_x = min(max(px, x_min), x_max)
            nearest_y = min(max(py, y_min), y_max)
            if (px - nearest_x) ** 2 + (py - nearest_y) ** 2 <= radius**2:
                return True
        return False

    def _ray_rectangle_intersection(
        self,
        origin: Sequence[float],
        direction: Sequence[float],
        obstacle: RectangleObstacle,
    ) -> Optional[float]:
        x_min, y_min, x_max, y_max = obstacle.as_bounds()
        ox, oy = origin
        dx, dy = direction
        t_min = 0.0
        t_max = self.config.sensor_range

        for origin_component, dir_component, min_bound, max_bound in (
            (ox, dx, x_min, x_max),
            (oy, dy, y_min, y_max),
        ):
            if abs(dir_component) < 1e-6:
                if origin_component < min_bound or origin_component > max_bound:
                    return None
                continue

            inv_dir = 1.0 / dir_component
            t1 = (min_bound - origin_component) * inv_dir
            t2 = (max_bound - origin_component) * inv_dir
            t_near = min(t1, t2)
            t_far = max(t1, t2)

            if t_far < 0:
                return None

            t_min = max(t_min, t_near)
            t_max = min(t_max, t_far)

            if t_min > t_max:
                return None

        if t_min < 0.0:
            return t_max if t_max >= 0.0 else None
        return t_min
