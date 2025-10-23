"""Grid-based A* planner compatible with Navigator environments."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from navigator import NavigatorConfig, RectangleObstacle

GridIndex = Tuple[int, int]
Point = Tuple[float, float]


@dataclass(frozen=True)
class AStarResult:
    path: List[Point]
    cost: float
    expanded: int
    success: bool


@dataclass
class GridMap:
    occupancy: np.ndarray
    resolution: float
    origin: Tuple[float, float]

    def world_to_index(self, point: Point) -> GridIndex:
        x, y = point
        col = int((x - self.origin[0]) / self.resolution)
        row = int((y - self.origin[1]) / self.resolution)
        return row, col

    def index_to_world(self, index: GridIndex) -> Point:
        row, col = index
        x = self.origin[0] + (col + 0.5) * self.resolution
        y = self.origin[1] + (row + 0.5) * self.resolution
        return x, y

    @property
    def shape(self) -> Tuple[int, int]:
        return self.occupancy.shape

    def in_bounds(self, index: GridIndex) -> bool:
        row, col = index
        return 0 <= row < self.occupancy.shape[0] and 0 <= col < self.occupancy.shape[1]

    def is_free(self, index: GridIndex) -> bool:
        row, col = index
        return not self.occupancy[row, col]


def build_grid(config: NavigatorConfig, resolution: float = 12.0, inflate_radius: Optional[float] = None) -> GridMap:
    """Rasterize rectangular obstacles into an occupancy grid."""
    inflate = inflate_radius if inflate_radius is not None else config.agent_radius
    height = config.map_height
    width = config.map_width

    rows = math.ceil(height / resolution)
    cols = math.ceil(width / resolution)
    occupancy = np.zeros((rows, cols), dtype=bool)

    obstacles: Sequence[RectangleObstacle] = config.resolved_obstacles()

    inflated = []
    for obs in obstacles:
        x_min, y_min, x_max, y_max = obs.as_bounds()
        x_min -= inflate
        y_min -= inflate
        x_max += inflate
        y_max += inflate
        inflated.append((x_min, y_min, x_max, y_max))

    for row in range(rows):
        for col in range(cols):
            cx = (col + 0.5) * resolution
            cy = (row + 0.5) * resolution
            if cx < 0 or cy < 0 or cx > width or cy > height:
                occupancy[row, col] = True
                continue
            for x_min, y_min, x_max, y_max in inflated:
                if x_min <= cx <= x_max and y_min <= cy <= y_max:
                    occupancy[row, col] = True
                    break

    origin = (0.0, 0.0)
    return GridMap(occupancy=occupancy, resolution=resolution, origin=origin)


def heuristic(a: GridIndex, b: GridIndex, scale: float) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1]) * scale


def neighbors(index: GridIndex) -> Iterable[GridIndex]:
    row, col = index
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield row + dr, col + dc


def astar(grid: GridMap, start_world: Point, goal_world: Point) -> AStarResult:
    start_idx = grid.world_to_index(start_world)
    goal_idx = grid.world_to_index(goal_world)

    if not grid.in_bounds(start_idx) or not grid.in_bounds(goal_idx):
        return AStarResult([], float("inf"), 0, False)
    if not grid.is_free(start_idx) or not grid.is_free(goal_idx):
        return AStarResult([], float("inf"), 0, False)

    open_set: List[Tuple[float, GridIndex]] = []
    heapq.heappush(open_set, (0.0, start_idx))

    came_from: Dict[GridIndex, Optional[GridIndex]] = {start_idx: None}
    g_score: Dict[GridIndex, float] = {start_idx: 0.0}
    expanded = 0

    while open_set:
        current_f, current = heapq.heappop(open_set)
        expanded += 1

        if current == goal_idx:
            path_idx: List[GridIndex] = []
            node = current
            while node is not None:
                path_idx.append(node)
                node = came_from[node]
            path_idx.reverse()
            path = [grid.index_to_world(idx) for idx in path_idx]
            return AStarResult(path=path, cost=g_score[current], expanded=expanded, success=True)

        for neighbor in neighbors(current):
            if not grid.in_bounds(neighbor) or not grid.is_free(neighbor):
                continue
            step_cost = math.hypot(neighbor[0] - current[0], neighbor[1] - current[1]) * grid.resolution
            tentative_g = g_score[current] + step_cost
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal_idx, grid.resolution)
                heapq.heappush(open_set, (f_score, neighbor))

    return AStarResult(path=[], cost=float("inf"), expanded=expanded, success=False)


def path_length(path: Sequence[Point]) -> float:
    if len(path) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(path, path[1:]):
        total += math.dist(a, b)
    return total


def plan_with_astar(config: NavigatorConfig, start: Point, goal: Point, resolution: float = 12.0) -> AStarResult:
    grid = build_grid(config, resolution=resolution)
    return astar(grid, start, goal)
