"""Equal-function-evaluation civilian path benchmark.

Independent extension to the paper-informed implementation, *not* the
original author's code. The calibrated-temperature SA variant is a new
experimental assumption and cannot be attributed to the article.
"""
from dataclasses import dataclass
import numpy as np

from .model import Scenario, Weights, bounds, fitness, initial_candidate
from .optimizer import immune_probabilities, OptimizerConfig

ALGORITHMS = ("pso", "ipso", "saipso-raw", "saipso-cal")


@dataclass
class BudgetResult:
    vector: np.ndarray
    fitness: float
    trace: np.ndarray
    evaluations: int
    sa_calls: int
    iterations: int
    temperature_scale: float


def optimize_budget(scenario: Scenario, weights: Weights, budget: int,
                    seed: int, algorithm: str,
                    config: OptimizerConfig | None = None) -> BudgetResult:
    """Exactly ``budget`` real fitness calls per algorithm, seed, and task.

    Common initial population is identical for a given seed and scenario;
    all stochastic choices after initialization belong to each arm's stream.
    Every re-evaluation from memory injection and SA is charged to the same
    objective-evaluation budget. All trajectories are best-ever by FE.
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}")
    cfg = config if config is not None else OptimizerConfig()
    if budget < cfg.population:
        raise ValueError("budget must be at least population size")
    rng = np.random.default_rng(seed)
    lo, hi = bounds(scenario)
    span = hi - lo
    pos = np.array([initial_candidate(rng, scenario) for _ in range(cfg.population)])
    vel = rng.uniform(-1.0, 1.0, size=pos.shape)
    best_score = float("inf")
    best_vector = pos[0].copy()
    trace: list[float] = []
    calls = 0
    sa_calls = 0
    iterations = 0
    archive: list[tuple[float, np.ndarray]] = []

    def evaluate(vector: np.ndarray) -> float:
        nonlocal calls, best_score, best_vector
        if calls >= budget:
            raise RuntimeError("objective evaluation budget exceeded")
        score = float(fitness(vector, scenario, weights))
        calls += 1
        if score < best_score:
            best_score = score
            best_vector = vector.copy()
        trace.append(best_score)
        return score

    def remember(vector: np.ndarray, score: float) -> None:
        if algorithm == "pso" or cfg.memory == 0:
            return
        if not any(np.allclose(vector, stored, atol=1e-7) for _, stored in archive):
            archive.append((float(score), vector.copy()))
            archive.sort(key=lambda item: item[0])
            del archive[cfg.memory:]

    scores = np.array([evaluate(p) for p in pos])
    pbest = pos.copy()
    pscores = scores.copy()
    # Adapt scale to normalized civilian objective's spread; this is an
    # explicitly new calibrated variant, not the temperature in the article.
    temperature_scale = max(float(np.std(scores)), 1e-4)
    remember(best_vector, best_score)
    stagnant = 0
    while calls < budget:
        old_best = best_score
        iterations += 1
        progress = min(1.0, calls / budget)
        w = cfg.inertia_max + (cfg.inertia_min - cfg.inertia_max) * progress
        # Prepare a synchronous PSO step, avoiding in-place influence changes.
        r1 = rng.random(size=pos.shape)
        r2 = rng.random(size=pos.shape)
        new_vel = np.clip(w * vel + cfg.c1 * r1 * (pbest-pos)
                          + cfg.c2 * r2 * (best_vector-pos),
                          -cfg.velocity_max, cfg.velocity_max)
        new_pos = np.clip(pos + new_vel, lo, hi)
        for i in range(cfg.population):
            if calls >= budget:
                break
            vel[i] = new_vel[i]
            pos[i] = new_pos[i]
            scores[i] = evaluate(pos[i])
            difference = scores[i] - pscores[i]
            if algorithm == "saipso-cal":
                t = temperature_scale * (cfg.sa_terminal/cfg.sa_temperature) ** progress
            else:
                t = max(cfg.sa_terminal, cfg.sa_temperature * cfg.sa_cooling ** (iterations-1))
            admit = (algorithm.startswith("saipso") and difference > 0 and
                     rng.random() < np.exp(-min(difference / max(t, 1e-12), 700)))
            if difference < 0 or admit:
                pbest[i] = pos[i].copy()
                pscores[i] = scores[i]
        if algorithm != "pso" and cfg.memory:
            for i in np.argsort(scores)[:3]:
                remember(pos[i], float(scores[i]))
            if archive and calls < budget:
                pool = np.vstack((pos, [vector for _, vector in archive]))
                pool_scores = np.r_[scores, [score for score, _ in archive]]
                probabilities = immune_probabilities(pool_scores, cfg.immune_beta)
                for i in np.argsort(scores)[-cfg.memory_injections:]:
                    if calls >= budget:
                        break
                    chosen = int(rng.choice(len(pool), p=probabilities))
                    pos[i] = np.clip(pool[chosen] + rng.normal(0, 0.015*span), lo, hi)
                    scores[i] = evaluate(pos[i])
                    if scores[i] < pscores[i]:
                        pbest[i], pscores[i] = pos[i].copy(), scores[i]
        remember(best_vector, best_score)
        if old_best - best_score <= cfg.stagnation_epsilon:
            stagnant += 1
        else:
            stagnant = 0
        if algorithm.startswith("saipso") and stagnant >= cfg.stagnation_iterations and calls < budget:
            stagnant = 0
            sa_calls += 1
            trial = best_vector.copy()
            trial_score = best_score
            temp = cfg.sa_temperature
            level = 0
            while (temp > cfg.sa_terminal and level < cfg.sa_max_levels and calls < budget):
                calibrated_t = (temperature_scale * temp / cfg.sa_temperature
                                if algorithm == "saipso-cal" else temp)
                for _ in range(cfg.sa_trials_per_level):
                    if calls >= budget:
                        break
                    proposal = np.clip(trial + rng.normal(0, cfg.sa_perturbation * span)
                                       * np.sqrt(temp/cfg.sa_temperature), lo, hi)
                    proposed = evaluate(proposal)
                    delta = proposed-trial_score
                    if delta <= 0 or rng.random() < np.exp(-min(delta/max(calibrated_t, 1e-12), 700)):
                        trial, trial_score = proposal, proposed
                temp *= cfg.sa_cooling
                level += 1
            remember(best_vector, best_score)
            if best_score < float(np.min(scores)):
                worst = int(np.argmax(scores))
                pos[worst] = pbest[worst] = best_vector.copy()
                scores[worst] = pscores[worst] = best_score
    assert calls == budget and len(trace) == budget
    return BudgetResult(best_vector, best_score, np.asarray(trace), calls,
                        sa_calls, iterations, temperature_scale)
