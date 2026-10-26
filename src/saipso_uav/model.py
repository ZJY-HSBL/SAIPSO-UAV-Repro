"""Civilian route environment and transparent, dimensionless fitness model.

Original paper specifies a five-constraint, multi-threat objective but omits
complete scenario geometry and penalty definitions. This is an independently
specified civilian terrain + restricted-airspace benchmark; NOT exact figure
or numeric reproduction and NOT air-defense/weapon-evasion functionality.
"""
from dataclasses import dataclass, field
import numpy as np

START = np.array([10.0, 10.0, 10.0])
GOAL = np.array([190.0, 190.0, 10.0])


@dataclass(frozen=True)
class Scenario:
    name: str = "civilian-mountains"
    # (x, y, summit_height, horizontal_sigma)
    hills: tuple[tuple[float, float, float, float], ...] = (
        (76.0, 92.0, 170.0, 23.0),
        (137.0, 135.0, 380.0, 17.0),
    )
    # Civilian restricted-airspace zones (x, y, radius), not defense systems.
    zones: tuple[tuple[float, float, float], ...] = (
        (62.0, 145.0, 16.0),
        (155.0, 52.0, 18.0),
    )
    world_xy: tuple[float, float] = (0.0, 200.0)
    max_altitude: float = 450.0
    min_clearance: float = 10.0
    comfort_clearance: float = 32.0
    min_segment: float = 3.0
    max_turn_degrees: float = 85.0
    max_climb_degrees: float = 60.0
    max_length_factor: float = 3.0
    waypoint_count: int = 10
    samples_per_segment: int = 7
    start: tuple[float, float, float] = (10.0, 10.0, 10.0)
    goal: tuple[float, float, float] = (190.0, 190.0, 10.0)


@dataclass(frozen=True)
class Weights:
    length: float
    exposure: float
    constraints: float

    def __post_init__(self):
        if min(self.length, self.exposure, self.constraints) < 0:
            raise ValueError("Weights cannot be negative")
        if not np.isclose(sum((self.length, self.exposure, self.constraints)), 1.0, atol=2e-4):
            raise ValueError("Weights must sum to one")


# Original paper, section 4.1/4.2: outer AHP weights (rounded values).
FAST = Weights(0.6370, 0.2583, 0.1047)
SAFE = Weights(0.0879, 0.6694, 0.2426)
WEIGHTS = {"fast": FAST, "safe": SAFE}


def terrain_height(x: np.ndarray, y: np.ndarray, scenario: Scenario) -> np.ndarray:
    result = np.zeros(np.broadcast_shapes(np.shape(x), np.shape(y)), dtype=float)
    for cx, cy, h, sigma in scenario.hills:
        result += h * np.exp(-((x-cx)**2 + (y-cy)**2) / (2.0*sigma*sigma))
    return result


def vector_to_route(vector: np.ndarray, scenario: Scenario) -> np.ndarray:
    """Optimize (y,z) at uniformly fixed interior x positions, 2K DOF.

    This dimension reduction is a disclosed adaptation: the article encodes
    full 3-D waypoints but doesn't provide an unambiguous encoding recipe.
    """
    vector = np.asarray(vector, dtype=float)
    k = scenario.waypoint_count
    if vector.shape != (2*k,):
        raise ValueError(f"Expected {(2*k,)} vector, got {vector.shape}")
    start = np.asarray(scenario.start)
    goal = np.asarray(scenario.goal)
    x = np.linspace(start[0], goal[0], k + 2)[1:-1]
    interior = np.column_stack((x, vector[:k], vector[k:]))
    return np.vstack((start, interior, goal))


def fitness_details(vector: np.ndarray, scenario: Scenario, weights: Weights) -> dict[str, float]:
    route = vector_to_route(vector, scenario)
    edges = np.diff(route, axis=0)
    segments = np.linalg.norm(edges, axis=1)
    length = float(segments.sum())
    direct = float(np.linalg.norm(route[-1] - route[0]))
    path_length = length/direct
    t = np.linspace(0.0, 1.0, scenario.samples_per_segment, endpoint=False)
    sample = (route[:-1, None, :] + t[None, :, None] * edges[:, None, :]).reshape(-1, 3)
    sample = np.vstack((sample, route[-1]))
    terrain = terrain_height(sample[:, 0], sample[:, 1], scenario)
    clearance = sample[:, 2] - terrain
    # Exposure to civilian terrain near-misses; collision gets a stronger penalty.
    terrain_risk = np.clip((scenario.comfort_clearance-clearance)/scenario.comfort_clearance, 0, None)
    terrain_risk = np.minimum(terrain_risk, 20.0)
    restricted_risk = np.zeros(len(sample), dtype=float)
    for cx, cy, radius in scenario.zones:
        distance = np.hypot(sample[:, 0]-cx, sample[:, 1]-cy)
        restricted_risk = np.maximum(restricted_risk, np.clip((radius-distance)/radius, 0, 1))
    exposure = float(np.mean(terrain_risk**2) + np.mean(restricted_risk**2))
    short_segments = float(np.mean(np.clip((scenario.min_segment-segments)/scenario.min_segment,0,None)**2))
    directions = edges / np.maximum(segments[:, None], 1e-12)
    cosines = np.clip(np.sum(directions[:-1]*directions[1:], axis=1), -1, 1)
    turns = np.degrees(np.arccos(cosines))
    over_turn = float(np.mean((np.clip(turns-scenario.max_turn_degrees,0,None)/90)**2))
    hnorm = np.linalg.norm(edges[:, :2], axis=1)
    climb_angle = np.degrees(np.arctan2(np.abs(edges[:, 2]), np.maximum(hnorm, 1e-12)))
    over_climb = float(np.mean((np.clip(climb_angle-scenario.max_climb_degrees,0,None)/90)**2))
    below = float(np.mean((np.clip(scenario.min_clearance-clearance, 0, None)/50)**2))
    altitude = float(np.mean((np.clip(sample[:, 2]-scenario.max_altitude,0,None)/50)**2))
    over_length = float((max(0.0, path_length-scenario.max_length_factor))**2)
    constraints = short_segments+over_turn+over_climb+below+altitude+over_length
    result = weights.length*path_length + weights.exposure*exposure + weights.constraints*constraints
    return {
        "fitness": float(result), "length": length, "length_ratio": path_length,
        "exposure": exposure, "constraints": constraints,
        "min_clearance": float(clearance.min()),
        "restricted_fraction": float(np.mean(restricted_risk > 0)),
        "max_turn_deg": float(np.max(turns)),
        "max_climb_deg": float(np.max(climb_angle)),
    }


def fitness(vector: np.ndarray, scenario: Scenario, weights: Weights) -> float:
    return fitness_details(vector, scenario, weights)["fitness"]


def initial_candidate(rng: np.random.Generator, scenario: Scenario) -> np.ndarray:
    k = scenario.waypoint_count
    baseline_y = np.linspace(scenario.start[1], scenario.goal[1], k+2)[1:-1]
    y = np.clip(baseline_y + rng.normal(0, 23.0, k), *scenario.world_xy)
    x = np.linspace(scenario.start[0], scenario.goal[0], k+2)[1:-1]
    ground = terrain_height(x, y, scenario)
    # Initial flying height is only a heuristic: constraints are always checked.
    z = np.clip(np.maximum(35.0, ground + 18.0) + rng.normal(0, 18.0, k), 5.0, scenario.max_altitude)
    return np.concatenate((y, z))


def bounds(scenario: Scenario) -> tuple[np.ndarray, np.ndarray]:
    k = scenario.waypoint_count
    lo = np.r_[np.full(k, scenario.world_xy[0]), np.full(k, 5.0)]
    hi = np.r_[np.full(k, scenario.world_xy[1]), np.full(k, scenario.max_altitude)]
    return lo, hi
