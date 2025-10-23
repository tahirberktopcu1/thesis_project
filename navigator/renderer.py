"""Pygame based renderer for the navigator environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

try:
    import pygame
except ImportError as exc:
    raise RuntimeError(
        "Pygame is required. Install it with `pip install pygame`."
    ) from exc


RenderFrame = np.ndarray


@dataclass
class NavigatorRenderer:
    """Handles Pygame drawing for the navigation environment."""

    width: int
    height: int
    agent_radius: int
    goal_radius: int
    render_mode: str
    border_thickness: int = 4

    def __post_init__(self) -> None:
        pygame.init()
        self._surface: Optional["pygame.Surface"] = None
        self._screen: Optional["pygame.Surface"] = None

        if self.render_mode == "human":
            pygame.display.set_caption("Linear Navigator")
            self._screen = pygame.display.set_mode((self.width, self.height))
            self._surface = pygame.Surface((self.width, self.height))
        elif self.render_mode == "rgb_array":
            self._surface = pygame.Surface((self.width, self.height))
        else:
            raise ValueError(f"Unknown render_mode: {self.render_mode}")

        self.reset()

    def reset(self) -> None:
        if self._surface is None:
            return
        self._surface.fill((24, 24, 32))
        if self.render_mode == "human" and self._screen is not None:
            self._screen.blit(self._surface, (0, 0))
            pygame.display.flip()

    def draw(
        self,
        agent_pos: Tuple[float, float],
        agent_angle: float,
        goal_pos: Tuple[float, float],
        sensor_rays: Sequence[Tuple[Tuple[float, float], Tuple[float, float], float]],
        obstacles: Sequence[Tuple[float, float, float, float]],
    ) -> Optional[RenderFrame]:
        self._handle_events()
        surface = self._surface
        if surface is None:
            return None

        surface.fill((24, 24, 32))
        self._draw_border(surface)
        self._draw_obstacles(surface, obstacles)
        self._draw_goal(surface, goal_pos)
        self._draw_sensors(surface, sensor_rays)
        self._draw_agent(surface, agent_pos, agent_angle)

        if self.render_mode == "human" and self._screen is not None:
            self._screen.blit(surface, (0, 0))
            pygame.display.flip()
            return None

        frame = pygame.surfarray.array3d(surface)
        return np.transpose(frame, (1, 0, 2))

    def close(self) -> None:
        if self._screen is not None:
            pygame.display.quit()
        pygame.quit()
        self._screen = None
        self._surface = None

    def _draw_goal(self, surface: "pygame.Surface", goal_pos: Tuple[float, float]) -> None:
        pygame.draw.circle(
            surface,
            (60, 200, 90),
            (int(goal_pos[0]), int(goal_pos[1])),
            self.goal_radius,
        )

    def _draw_sensors(
        self,
        surface: "pygame.Surface",
        rays: Sequence[Tuple[Tuple[float, float], Tuple[float, float], float]],
    ) -> None:
        for start, end, normalized in rays:
            pygame.draw.line(
                surface,
                self._sensor_color(normalized),
                (int(start[0]), int(start[1])),
                (int(end[0]), int(end[1])),
                2,
            )

    def _draw_agent(
        self,
        surface: "pygame.Surface",
        agent_pos: Tuple[float, float],
        agent_angle: float,
    ) -> None:
        cx, cy = agent_pos
        pygame.draw.circle(surface, (90, 140, 255), (int(cx), int(cy)), self.agent_radius)

        tip_length = self.agent_radius * 1.8
        left_angle = agent_angle + 2.5
        right_angle = agent_angle - 2.5

        tip = (cx + tip_length * np.cos(agent_angle), cy + tip_length * np.sin(agent_angle))
        left = (cx + self.agent_radius * np.cos(left_angle), cy + self.agent_radius * np.sin(left_angle))
        right = (cx + self.agent_radius * np.cos(right_angle), cy + self.agent_radius * np.sin(right_angle))

        pygame.draw.polygon(
            surface,
            (255, 255, 255),
            [(int(tip[0]), int(tip[1])), (int(left[0]), int(left[1])), (int(right[0]), int(right[1]))],
        )

    def _draw_obstacles(
        self,
        surface: "pygame.Surface",
        obstacles: Sequence[Tuple[float, float, float, float]],
    ) -> None:
        for x_min, y_min, x_max, y_max in obstacles:
            pygame.draw.rect(
                surface,
                (120, 80, 70),
                pygame.Rect(int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)),
            )

    def _draw_border(self, surface: "pygame.Surface") -> None:
        thickness = max(1, int(self.border_thickness))
        pygame.draw.rect(
            surface,
            (200, 200, 210),
            pygame.Rect(0, 0, self.width, self.height),
            width=thickness,
        )

    @staticmethod
    def _sensor_color(normalized: float) -> Tuple[int, int, int]:
        normalized = max(0.0, min(1.0, normalized))
        red = int(255 * (1.0 - normalized))
        green = int(255 * normalized)
        blue = 80
        return red, green, blue

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
