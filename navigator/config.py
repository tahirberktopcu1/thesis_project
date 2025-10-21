"""Configuration primitives for the navigator RL environment."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple


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
    sensor_angles: Sequence[float] = field(
        default_factory=lambda: (-math.pi / 4, 0.0, math.pi / 4)
    )
    sensor_range: float = 200.0
    start_pos: Optional[Tuple[float, float]] = None
    goal_pos: Optional[Tuple[float, float]] = None

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
