"""Configuration primitives for the navigator RL environment."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class RectangleObstacle:
    """Axis-aligned rectangular obstacle description."""

    x: float
    y: float
    width: float
    height: float

    def as_bounds(self) -> Tuple[float, float, float, float]:
        """Return (x_min, y_min, x_max, y_max)."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass(frozen=True)
class RandomObstacleSpec:
    """Parameters that control randomly generated rectangular obstacles."""

    count: int = 6
    min_size: Tuple[float, float] = (42.0, 72.0)
    max_size: Tuple[float, float] = (96.0, 192.0)
    min_margin: float = 48.0
    max_attempts: int = 200


@dataclass
class NavigatorConfig:
    """Holds tunable parameters for the navigation task."""

    map_width: int = 768  # 20% daha geniş
    map_height: int = 576  # 20% daha yuksek
    agent_radius: int = 12
    goal_radius: int = 18
    forward_speed: float = 6.0
    turn_speed: float = math.pi / 14.0
    max_episode_steps: int = 400
    collision_penalty: float = 1.0
    idle_penalty: float = 0.08
    border_thickness: int = 6
    safety_threshold: float = 0.35
    safety_penalty_gain: float = 0.2
    forward_block_threshold: float = 0.25
    turn_bonus_gain: float = 0.03
    corner_margin: float = 10.0
    sensor_angles: Sequence[float] = field(
        default_factory=lambda: tuple(
            i * (2.0 * math.pi / 24.0) for i in range(24)
        )
    )
    sensor_range: float = 240.0
    start_pos: Optional[Tuple[float, float]] = None
    goal_pos: Optional[Tuple[float, float]] = None
    obstacles: Sequence[RectangleObstacle | Tuple[float, float, float, float]] = field(
        default_factory=lambda: (
            RectangleObstacle(264.0, 144.0, 48.0, 240.0),
            RectangleObstacle(432.0, 72.0, 60.0, 168.0),
            RectangleObstacle(516.0, 312.0, 84.0, 192.0),
        )
    )
    randomize_obstacles: bool = False
    keep_static_when_random: bool = False
    random_obstacle_spec: RandomObstacleSpec = field(default_factory=RandomObstacleSpec)

    def resolved_start(self) -> Tuple[float, float]:
        """Return the starting position, defaulting to the left center."""
        if self.start_pos is not None:
            return self.start_pos
        return (self.agent_radius + 10.0, self.map_height / 2.0)

    def resolved_goal(self) -> Tuple[float, float]:
        """Return the goal position, defaulting to the right center."""
        if self.goal_pos is not None:
            return self.goal_pos
        return (self.map_width - self.goal_radius - 10.0, self.map_height / 2.0)

    def resolved_obstacles(self) -> Tuple[RectangleObstacle, ...]:
        """Return normalized obstacle descriptions."""
        normalized = []
        for obstacle in self.obstacles:
            if isinstance(obstacle, RectangleObstacle):
                normalized.append(obstacle)
            else:
                x, y, w, h = obstacle
                normalized.append(RectangleObstacle(float(x), float(y), float(w), float(h)))
        return tuple(normalized)


