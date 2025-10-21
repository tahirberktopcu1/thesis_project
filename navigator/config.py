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


@dataclass
class NavigatorConfig:
    """Holds tunable parameters for the navigation task."""

    map_width: int = 640
    map_height: int = 480
    agent_radius: int = 12
    goal_radius: int = 18
    forward_speed: float = 6.0
    turn_speed: float = math.pi / 14.0
    max_episode_steps: int = 400
    collision_penalty: float = 1.0
    border_thickness: int = 6
    sensor_angles: Sequence[float] = field(
        default_factory=lambda: tuple(
            i * (2.0 * math.pi / 8.0) for i in range(8)
        )
    )
    sensor_range: float = 200.0
    start_pos: Optional[Tuple[float, float]] = None
    goal_pos: Optional[Tuple[float, float]] = None
    obstacles: Sequence[RectangleObstacle | Tuple[float, float, float, float]] = field(
        default_factory=lambda: (
            RectangleObstacle(220.0, 120.0, 40.0, 200.0),
            RectangleObstacle(360.0, 60.0, 50.0, 140.0),
            RectangleObstacle(430.0, 260.0, 70.0, 160.0),
        )
    )

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
