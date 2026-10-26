"""Dense independent geometry validation; not a flight-safety certificate."""
import numpy as np
from .model import Scenario, terrain_height, vector_to_route


def dense_validation(vector: np.ndarray, scenario: Scenario, samples: int = 101) -> dict:
    """Evaluate sampled path feasibility more densely than training objective."""
    if samples < 3:
        raise ValueError("samples must be at least 3")
    route = vector_to_route(vector, scenario)
    diffs = np.diff(route, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    t = np.linspace(0, 1, samples, endpoint=False)
    pts = (route[:-1, None, :] + t[None, :, None]*diffs[:, None, :]).reshape(-1, 3)
    pts = np.vstack((pts, route[-1]))
    clearance = pts[:, 2] - terrain_height(pts[:, 0], pts[:, 1], scenario)
    restricted = np.zeros(len(pts), dtype=bool)
    for x, y, radius in scenario.zones:
        restricted |= np.hypot(pts[:, 0]-x, pts[:, 1]-y) < radius
    turns = np.degrees(np.arccos(np.clip(np.sum(diffs[:-1]*diffs[1:], axis=1) /
                       np.maximum(distances[:-1]*distances[1:], 1e-12), -1, 1)))
    climb = np.degrees(np.arctan2(np.abs(diffs[:, 2]),
                                  np.linalg.norm(diffs[:, :2], axis=1)))
    direct = np.linalg.norm(route[-1]-route[0])
    checks = dict(
        clearance=bool(np.min(clearance) >= scenario.min_clearance-1e-8),
        restricted_airspace=bool(not restricted.any()),
        turn=bool(np.max(turns) <= scenario.max_turn_degrees+1e-8),
        climb=bool(np.max(climb) <= scenario.max_climb_degrees+1e-8),
        minimum_segment=bool(np.min(distances) >= scenario.min_segment-1e-8),
        maximum_length=bool(sum(distances) <= scenario.max_length_factor*direct+1e-8),
        maximum_altitude=bool(np.max(pts[:, 2]) <= scenario.max_altitude+1e-8),
    )
    return {"sampled_feasible": all(checks.values()), "checks": checks,
            "min_dense_clearance": float(np.min(clearance)),
            "restricted_sample_fraction": float(np.mean(restricted)),
            "max_turn": float(np.max(turns)), "max_climb": float(np.max(climb))}
