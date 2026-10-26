"""Documented SA-IPSO reconstruction for reproducible experiments.

Specified by the article: linear inertia, SA pbest admission, immune memory,
affinity/concentration selection, stagnation-triggered SA refinement.
Implementation choices not specified by the article are explicitly exposed
in OptimizerConfig, with strict best-so-far reporting.
"""
from dataclasses import dataclass
import numpy as np

from .model import Scenario, Weights, bounds, fitness, initial_candidate


@dataclass(frozen=True)
class OptimizerConfig:
    population: int = 20
    memory: int = 10
    iterations: int = 160  # 1000 in paper; smaller CLI default for quick demos
    c1: float = 2.05
    c2: float = 2.05
    inertia_max: float = 0.9
    inertia_min: float = 0.4
    velocity_max: float = 20.0
    stagnation_iterations: int = 2
    stagnation_epsilon: float = 1e-4
    sa_temperature: float = 2000.0
    sa_terminal: float = 10.0
    sa_cooling: float = 0.8
    sa_trials_per_level: int = 2  # assumption; not specified in article
    sa_max_levels: int = 24
    sa_perturbation: float = 0.08  # relative to decision-variable range
    memory_injections: int = 2   # assumption; not specified in article
    immune_beta: float = 0.5

    def __post_init__(self):
        if self.population < 2 or self.memory < 0 or self.iterations < 1:
            raise ValueError("invalid population/memory/iterations")
        if not (0 < self.sa_cooling < 1) or self.sa_temperature <= self.sa_terminal:
            raise ValueError("invalid annealing temperatures")
        if self.sa_trials_per_level < 1 or self.sa_max_levels < 1:
            raise ValueError("invalid SA trial budget")


@dataclass
class OptimizationResult:
    best_vector: np.ndarray
    best_fitness: float
    history: np.ndarray
    evaluations: int
    sa_calls: int
    algorithm: str
    seed: int


def immune_probabilities(scores: np.ndarray, beta: float = 0.5) -> np.ndarray:
    """P = beta * normalized(1/F) + (1-beta) * normalized(1/D).

    Paper defines D_i=1/sum_j |F_i-F_j|; inverse density 1/D is
    consequently the sum of fitness distances, protecting dissimilarity.
    If denominators vanish we use uniform probabilities (documented guard).
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("scores must be a nonempty finite vector")
    if not 0 <= beta <= 1:
        raise ValueError("beta must be in [0,1]")
    shift = max(0.0, -float(np.min(scores))) + 1e-9
    affinity = 1.0 / (scores + shift)
    a = affinity / affinity.sum()
    diversity = np.abs(scores[:, None] - scores[None, :]).sum(axis=1)
    b = diversity / diversity.sum() if diversity.sum() > 1e-12 else np.full(len(scores), 1.0/len(scores))
    result = beta*a+(1-beta)*b
    return result/result.sum()


def optimize(scenario: Scenario, weights: Weights, config: OptimizerConfig,
             seed: int = 0, algorithm: str = "saipso") -> OptimizationResult:
    if algorithm not in ("pso", "ipso", "saipso"):
        raise ValueError("algorithm must be pso, ipso, or saipso")
    rng = np.random.default_rng(seed)
    low, high = bounds(scenario)
    dim = len(low)
    span = high - low
    pos = np.array([initial_candidate(rng, scenario) for _ in range(config.population)])
    vel = rng.uniform(-1.0, 1.0, size=(config.population, dim))
    scores = np.array([fitness(p, scenario, weights) for p in pos])
    evals = config.population
    pbest = pos.copy()
    pscores = scores.copy()
    best_idx = int(np.argmin(scores))
    gbest = pos[best_idx].copy()
    gscore = float(scores[best_idx])
    archive: list[tuple[float, np.ndarray]] = []
    sa_calls = 0
    history = [gscore]
    idle = 0

    def update_archive(vector: np.ndarray, score: float) -> None:
        if algorithm == "pso" or config.memory == 0:
            return
        if not any(np.allclose(vector, v, atol=1e-7) for _, v in archive):
            archive.append((float(score), vector.copy()))
            archive.sort(key=lambda item: item[0])
            del archive[config.memory:]

    update_archive(gbest, gscore)

    for iteration in range(config.iterations):
        prev_best = gscore
        w = config.inertia_max - (config.inertia_max-config.inertia_min) * iteration/max(1, config.iterations-1)
        r1 = rng.random(size=pos.shape)
        r2 = rng.random(size=pos.shape)
        vel = (w*vel + config.c1*r1*(pbest-pos) + config.c2*r2*(gbest-pos))
        vel = np.clip(vel, -config.velocity_max, config.velocity_max)
        pos = np.clip(pos + vel, low, high)
        scores = np.array([fitness(p, scenario, weights) for p in pos])
        evals += len(pos)

        for i in range(len(pos)):
            delta = float(scores[i]-pscores[i])
            # SA admission may replace a personal best with a worse point.
            # Best-ever gbest is maintained independently and never degrades.
            accept_worse = (algorithm == "saipso" and delta > 0 and
                rng.random() < np.exp(-min(delta/max(config.sa_temperature*(config.sa_cooling**iteration),
                                                       config.sa_terminal), 700)))
            if delta < 0 or accept_worse:
                pbest[i] = pos[i].copy()
                pscores[i] = scores[i]

        if algorithm != "pso" and config.memory:
            # Archive both current leaders and independent, nonidentical elite routes.
            for i in np.argsort(scores)[:min(3, len(scores))]:
                update_archive(pos[i], float(scores[i]))
            pool = np.vstack((pos, np.array([v for _, v in archive])))
            pool_scores = np.r_[scores, np.array([f for f, _ in archive])]
            selection = immune_probabilities(pool_scores, config.immune_beta)
            for i in np.argsort(scores)[-min(config.memory_injections, len(scores)):]:
                j = rng.choice(len(pool), p=selection)
                pos[i] = np.clip(pool[j] + rng.normal(0, 0.015*span), low, high)
                scores[i] = fitness(pos[i], scenario, weights)
                evals += 1
                if scores[i] < pscores[i]:
                    pbest[i] = pos[i].copy()
                    pscores[i] = scores[i]

        i = int(np.argmin(scores))
        if scores[i] < gscore:
            gscore = float(scores[i])
            gbest = pos[i].copy()
            update_archive(gbest, gscore)

        if prev_best-gscore <= config.stagnation_epsilon:
            idle += 1
        else:
            idle = 0

        if algorithm == "saipso" and idle >= config.stagnation_iterations:
            sa_calls += 1
            idle = 0
            trial = gbest.copy()
            trial_score = gscore
            temp = config.sa_temperature
            levels = 0
            while temp > config.sa_terminal and levels < config.sa_max_levels:
                for _ in range(config.sa_trials_per_level):
                    proposal = np.clip(trial + rng.normal(0, config.sa_perturbation*span)
                                       * (temp/config.sa_temperature)**0.5, low, high)
                    new_score = fitness(proposal, scenario, weights)
                    evals += 1
                    difference = new_score-trial_score
                    if difference < 0 or rng.random() < np.exp(-min(max(difference, 0)/temp, 700)):
                        trial = proposal
                        trial_score = new_score
                    if trial_score < gscore:
                        gscore, gbest = float(trial_score), trial.copy()
                        update_archive(gbest, gscore)
                temp *= config.sa_cooling
                levels += 1
            # Make SA-discovered coordinates accessible to swarm evolution.
            if gscore < float(np.min(scores)):
                worst = int(np.argmax(scores))
                pos[worst], scores[worst] = gbest.copy(), gscore
                pbest[worst], pscores[worst] = gbest.copy(), gscore

        history.append(gscore)

    return OptimizationResult(gbest, gscore, np.array(history), evals, sa_calls, algorithm, seed)
