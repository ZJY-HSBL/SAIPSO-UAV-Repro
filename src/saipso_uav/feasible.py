"""Auditable civilian constrained optimization study (v0.3.0).

This independent extension is NOT the paper's algorithm or a flight controller.
All methods are assessed with the same budget of fitness+constraint evaluations.
Restricted civilian airspace is enforced using exact horizontal segment-circle
geometry; terrain clearance remains densely sampled, not continuously proved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple
import numpy as np
from .model import Scenario, Weights, bounds, fitness_details, initial_candidate, terrain_height, vector_to_route
from .optimizer import OptimizerConfig
from .validation import dense_validation

# Controlled ablations. The only difference between variants is the listed
# feature switches; all use the same evaluation and geometry checks.
ARMS = (
    "soft-saipso",
    "cd-saipso",
    "cd-seed-saipso",
    "cd-seed-repair-pso",
    "cd-seed-repair-ipso",
    "cd-seed-repair-saipso",
)


class EvalRecord(NamedTuple):
    fitness: float
    violation: float
    feasible: bool


def segment_circle_clearance(route: np.ndarray, cx: float, cy: float, radius: float) -> float:
    """Exact signed horizontal clearance from every straight segment to disk."""
    a = route[:-1, :2]
    v = route[1:, :2] - a
    center = np.array((cx, cy))
    t = np.clip(np.sum((center-a)*v, axis=1)/np.maximum(np.sum(v*v, axis=1), 1e-12), 0.0, 1.0)
    closest = a + t[:, None]*v
    return float(np.min(np.linalg.norm(closest-center, axis=1))-radius)


def feasibility_metrics(vector: np.ndarray, scenario: Scenario, samples: int = 61) -> dict:
    """Nonnegative normalized violation components; exactly 0 = sampled feasible.

    This is called *inside* each counted objective evaluation, so no hidden
    extra constraint-oracle calls are used by constrained variants.
    """
    if samples < 3:
        raise ValueError("samples must be >= 3")
    route = vector_to_route(vector, scenario)
    edges = np.diff(route, axis=0)
    distance = np.linalg.norm(edges, axis=1)
    horiz = np.linalg.norm(edges[:, :2], axis=1)
    unit = edges / np.maximum(distance[:, None], 1e-12)
    turns = np.degrees(np.arccos(np.clip(np.sum(unit[:-1]*unit[1:], axis=1), -1.0, 1.0)))
    climbs = np.degrees(np.arctan2(abs(edges[:, 2]), np.maximum(horiz, 1e-12)))
    ts = np.linspace(0, 1, samples, endpoint=False)
    pts = (route[:-1, None, :] + ts[None, :, None]*edges[:, None, :]).reshape(-1, 3)
    pts = np.vstack((pts, route[-1]))
    ground = terrain_height(pts[:, 0], pts[:, 1], scenario)
    clearance_min = float(np.min(pts[:, 2]-ground))
    zone_clearance = min((segment_circle_clearance(route, *zone) for zone in scenario.zones), default=float('inf'))
    length = float(distance.sum())
    max_len = scenario.max_length_factor * float(np.linalg.norm(route[-1]-route[0]))
    values = {
        "clearance": max(0.0, scenario.min_clearance-clearance_min)/50.0,
        "airspace": max(0.0, -zone_clearance)/25.0,
        "turn": max(0.0, float(np.max(turns))-scenario.max_turn_degrees)/90.0,
        "climb": max(0.0, float(np.max(climbs))-scenario.max_climb_degrees)/90.0,
        "min_segment": max(0.0, scenario.min_segment-float(np.min(distance)))/max(scenario.min_segment, 1.0),
        "max_length": max(0.0, length-max_len)/max_len,
        "max_altitude": max(0.0, float(np.max(pts[:, 2]))-scenario.max_altitude)/max(scenario.max_altitude, 1.0),
    }
    violation = float(sum(values.values()))
    return {"violation": violation, "feasible": violation <= 1e-10,
            "components": values, "min_clearance": clearance_min,
            "zone_clearance": zone_clearance}


def seed_corridor(scenario: Scenario, tau: float = 17.0, bump: float = 70.0) -> np.ndarray:
    """Deterministic high-latitude civilian detour warm-start.

    Heuristic uses only scenario endpoints and fixed constants; no objective
    calls or hidden accept/reject loop. It is *not* a general feasibility repair.
    With this corridor the route is deliberately longer than a straight line.
    """
    k = scenario.waypoint_count
    start, goal = np.asarray(scenario.start), np.asarray(scenario.goal)
    x = np.linspace(start[0], goal[0], k+2)[1:-1]
    scale = (goal[0]-start[0])/180.0
    y = goal[1] - (goal[1]-start[1])*np.exp(-(x-start[0])/(tau*scale))
    z = 25.0 + bump*np.exp(-((x-start[0]-17*scale)/(32*scale))**2)
    lo, hi = bounds(scenario)
    return np.clip(np.r_[y, z], lo, hi)


CORRIDOR_GRID = ((12,40), (17,40), (22,40), (12,70), (17,70),
                 (22,70), (17,100), (22,100))


def deterministic_repair(vector: np.ndarray, scenario: Scenario, attractor: np.ndarray,
                         strength: float = .50) -> np.ndarray:
    """Geometric convex blend toward a predeclared detour, without search or FE.

    The same repair geometry is available to pso/ipso/saipso feature arms.
    No guarantee of feasibility: every proposed candidate is evaluated normally.
    """
    if not 0 <= strength <= 1:
        raise ValueError("strength must be in [0,1]")
    lo, hi = bounds(scenario)
    return np.clip((1-strength)*np.asarray(vector) + strength*np.asarray(attractor), lo, hi)


def dominates(a: EvalRecord, b: EvalRecord, constrained: bool) -> bool:
    """Deb's feasibility-first order, with normalized violation as tiebreaker."""
    if constrained:
        if a.feasible != b.feasible:
            return a.feasible
        if not a.feasible:
            return (a.violation, a.fitness) < (b.violation, b.fitness)
    return a.fitness < b.fitness


def quality_key(v: EvalRecord, constrained: bool) -> tuple[float, ...]:
    return ((0.0, v.fitness) if v.feasible else (1.0, v.violation, v.fitness)) if constrained else (v.fitness,)


@dataclass
class FeasibleResult:
    vector: np.ndarray
    score: float
    violation: float
    feasible: bool
    best_feasible_score: float | None
    first_feasible_fe: int | None
    evaluations: int
    sa_calls: int
    history_fitness: np.ndarray
    history_violation: np.ndarray
    history_feasible: np.ndarray


def optimize_feasible(scenario: Scenario, weights: Weights, budget: int,
                      seed: int, arm: str, config: OptimizerConfig | None = None) -> FeasibleResult:
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}")
    cfg = config or OptimizerConfig()
    if budget < cfg.population:
        raise ValueError("budget below initial population")
    if scenario.waypoint_count < 3:
        raise ValueError("at least three interior waypoints required")
    seeded = "seed" in arm
    constrained = arm != "soft-saipso"
    do_repair = "repair" in arm
    immune = arm.endswith("ipso") or arm.endswith("saipso")
    anneal = arm.endswith("saipso")
    rng = np.random.default_rng(seed)
    lo, hi = bounds(scenario)
    span = hi-lo
    pos = np.array([initial_candidate(rng, scenario) for _ in range(cfg.population)])
    vel = rng.uniform(-1.0, 1.0, pos.shape)
    corridors = [seed_corridor(scenario, tau, bump) for tau, bump in CORRIDOR_GRID]
    if seeded:
        for i, corridor in enumerate(corridors[:cfg.population]):
            pos[i] = corridor
    # Keep a fixed attractor (no objective-informed map tuning).
    attractor = seed_corridor(scenario, 17., 70.)
    calls = 0
    first_feasible: int | None = None
    best_feasible_score = float('inf')
    traces_f: list[float] = []
    traces_v: list[float] = []
    traces_feas: list[float] = []
    best_pos: np.ndarray | None = None
    best_rec: EvalRecord | None = None
    archive: list[tuple[EvalRecord, np.ndarray]] = []
    sa_calls = 0

    def evaluate(x: np.ndarray) -> EvalRecord:
        nonlocal calls, first_feasible, best_feasible_score, best_pos, best_rec
        if calls >= budget:
            raise RuntimeError("FE budget exceeded")
        f = float(fitness_details(x, scenario, weights)["fitness"])
        m = feasibility_metrics(x, scenario)
        rec = EvalRecord(f, m["violation"], bool(m["feasible"]))
        calls += 1
        if rec.feasible:
            if first_feasible is None:
                first_feasible = calls
            best_feasible_score = min(best_feasible_score, f)
        if best_rec is None or dominates(rec, best_rec, constrained):
            best_rec, best_pos = rec, x.copy()
        traces_f.append(float(best_rec.fitness))
        traces_v.append(float(best_rec.violation))
        traces_feas.append(float(best_feasible_score) if np.isfinite(best_feasible_score) else float('nan'))
        return rec

    def memorize(x: np.ndarray, rec: EvalRecord) -> None:
        if not immune or cfg.memory <= 0:
            return
        if not any(np.allclose(x, old, atol=1e-7) for _, old in archive):
            archive.append((rec, x.copy()))
            archive.sort(key=lambda item: quality_key(item[0], constrained))
            del archive[cfg.memory:]

    current = [evaluate(p) for p in pos]
    pbest = pos.copy()
    pr = list(current)
    for j in range(len(pos)):
        memorize(pos[j], current[j])
    score_spread = max(float(np.std([rec.fitness for rec in current])), 1e-4)
    stagnation = 0
    iterations = 0
    while calls < budget:
        iterations += 1
        old_quality = quality_key(best_rec, constrained)
        progress = calls / budget
        w = cfg.inertia_max + (cfg.inertia_min-cfg.inertia_max) * progress
        r1 = rng.random(pos.shape)
        r2 = rng.random(pos.shape)
        vnew = np.clip(w*vel + cfg.c1*r1*(pbest-pos) + cfg.c2*r2*(best_pos-pos),
                       -cfg.velocity_max, cfg.velocity_max)
        pnew = np.clip(pos+vnew, lo, hi)
        for i in range(cfg.population):
            if calls >= budget:
                break
            proposal = pnew[i]
            if do_repair and rng.random() < .35:
                proposal = deterministic_repair(proposal, scenario, attractor, .60)
            vel[i] = proposal-pos[i]  # bounded effective displacement
            pos[i] = proposal.copy()
            current[i] = evaluate(pos[i])
            if dominates(current[i], pr[i], constrained):
                pbest[i] = pos[i].copy()
                pr[i] = current[i]
        if immune and cfg.memory > 0:
            for i in sorted(range(cfg.population), key=lambda j: quality_key(current[j], constrained))[:3]:
                memorize(pos[i], current[i])
            if archive:
                # Fixed-count immune injection, all reevaluations charged to FE.
                for i in sorted(range(cfg.population), key=lambda j: quality_key(current[j], constrained), reverse=True)[:cfg.memory_injections]:
                    if calls >= budget:
                        break
                    # Rank-biased immune memory retains near-best candidates.
                    rank = int(rng.choice(len(archive), p=(np.arange(len(archive),0,-1)/sum(range(1,len(archive)+1)))))
                    src = archive[rank][1]
                    proposal = np.clip(src+rng.normal(0, .015*span), lo, hi)
                    if do_repair and rng.random() < .35:
                        proposal = deterministic_repair(proposal, scenario, attractor, .60)
                    pos[i] = proposal
                    current[i] = evaluate(pos[i])
                    if dominates(current[i], pr[i], constrained):
                        pbest[i], pr[i] = pos[i].copy(), current[i]
            memorize(best_pos, best_rec)
        new_quality = quality_key(best_rec, constrained)
        if new_quality < old_quality:
            stagnation = 0
        else:
            stagnation += 1
        if anneal and stagnation >= cfg.stagnation_iterations and calls < budget:
            sa_calls += 1
            stagnation = 0
            trial = best_pos.copy()
            trec = best_rec
            temp = cfg.sa_temperature
            level = 0
            while temp > cfg.sa_terminal and level < cfg.sa_max_levels and calls < budget:
                t_scale = score_spread * temp/cfg.sa_temperature
                for _ in range(cfg.sa_trials_per_level):
                    if calls >= budget:
                        break
                    prop = np.clip(trial + rng.normal(0, cfg.sa_perturbation*span)*np.sqrt(temp/cfg.sa_temperature), lo, hi)
                    if do_repair and rng.random() < .35:
                        prop = deterministic_repair(prop, scenario, attractor, .60)
                    rec = evaluate(prop)
                    if dominates(rec, trec, constrained):
                        trial, trec = prop.copy(), rec
                    else:
                        # Metropolis is restricted to the same feasibility class:
                        # a feasible incumbent is never made infeasible by SA.
                        if rec.feasible == trec.feasible:
                            delta = (rec.fitness-trec.fitness) if rec.feasible or not constrained else (rec.violation-trec.violation)
                            if rng.random() < np.exp(-min(max(delta,0)/max(t_scale,1e-12), 700)):
                                trial, trec = prop.copy(), rec
                temp *= cfg.sa_cooling
                level += 1
            memorize(best_pos, best_rec)
            worst = max(range(cfg.population), key=lambda j: quality_key(current[j], constrained))
            if dominates(best_rec, current[worst], constrained):
                pos[worst] = best_pos.copy()
                pbest[worst] = best_pos.copy()
                current[worst] = pr[worst] = best_rec
    assert calls == budget
    # The optimization search uses 61 sample points per segment; the independent
    # dense validator uses 101, so disagreements are explicitly reported.
    return FeasibleResult(best_pos, best_rec.fitness, best_rec.violation, best_rec.feasible,
                          (best_feasible_score if np.isfinite(best_feasible_score) else None),
                          first_feasible, calls, sa_calls,
                          np.asarray(traces_f), np.asarray(traces_v), np.asarray(traces_feas))
