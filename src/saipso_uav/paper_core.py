"""Isolated reconstruction of the paper's optimizer on generic bounded functions.

Paper-specified: 0.9->0.4 inertia, c1=c2=2.05, artificial-immune
P=0.5*P_affinity+0.5*P_inverse_concentration, SA acceptance of worsening
personal best, stagnation after 2 iterations, 2000->10 cooling by 0.8.

Not specified by paper: random initialization, elite archive replacement,
local-SA perturbation, random stream, boundary handling and Metropolis
schedule for personal-best updates. These are independently chosen, never
claimed to be the author's unpublished MATLAB code. Pure numerical
optimization only; no military route/threat modeling.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass(frozen=True)
class CoreConfig:
    population: int = 20
    memory: int = 10
    dimension: int = 10
    iterations: int = 1000
    c1: float = 2.05
    c2: float = 2.05
    w_max: float = 0.9
    w_min: float = 0.4
    vmax: float = 20.0
    stagnation: int = 2
    epsilon: float = 1e-4
    t0: float = 2000.0
    t_end: float = 10.0
    alpha: float = 0.8
    beta: float = 0.5
    sa_trials_per_level: int = 2  # independent reproduction assumption
    sa_step_fraction: float = 0.08  # independent reproduction assumption
    memory_injections: int = 2  # independent reproduction assumption

    def __post_init__(self):
        if not (self.dimension > 0 and self.population >= 2 and self.memory >= 0
                and self.iterations > 0 and self.stagnation >= 1):
            raise ValueError('invalid population, dimension, memory or iteration count')
        if not (self.t0 > self.t_end > 0 and 0 < self.alpha < 1
                and 0 <= self.beta <= 1 and self.sa_trials_per_level > 0
                and 0 <= self.memory_injections <= self.population
                and 0 < self.sa_step_fraction <= 1 and self.vmax > 0):
            raise ValueError('invalid optimizer parameters')


@dataclass
class CoreResult:
    x: np.ndarray
    value: float
    fe: int
    iterations: int
    sa_triggers: int
    pbest_worse_accepted: int
    history: np.ndarray
    fe_history: np.ndarray
    algorithm: str
    seed: int


ARMS = ('pso', 'ipso', 'saipso', 'saipso-no-immune',
        'saipso-no-pbest-sa', 'saipso-no-local-sa')


def probability(scores: np.ndarray, beta: float = 0.5) -> np.ndarray:
    """Paper section 2.4, with stable zero/negative-fitness guards.

    D_i=1/sum_j |F_i-F_j| => 1/D_i=sum_j |F_i-F_j|.
    At zero spread the concentration term is uniform by convention.
    """
    f = np.asarray(scores, dtype=float)
    if f.ndim != 1 or not f.size or not np.all(np.isfinite(f)):
        raise ValueError('scores must be a nonempty finite vector')
    if not 0 <= beta <= 1:
        raise ValueError('beta out of range')
    shifted = f - min(0., float(np.min(f))) + 1e-12
    inv = 1.0 / shifted
    affinity = inv / np.sum(inv)
    diversity = np.sum(np.abs(f[:, None] - f[None, :]), axis=1)
    spread = np.sum(diversity)
    concentration = diversity/spread if spread > 1e-12 else np.full(len(f), 1/len(f))
    p = beta*affinity + (1-beta)*concentration
    return p/np.sum(p)


def inertia(t: int, tmax: int, wmax: float = .9, wmin: float = .4) -> float:
    """Exactly omega(t)=omega_max-(omega_max-omega_min)*t/tmax."""
    if tmax <= 0 or not 0 <= t <= tmax:
        raise ValueError('time index out of range')
    return wmax - (wmax-wmin)*t/tmax


def metropolis(delta: float, temperature: float, rng: np.random.Generator) -> bool:
    """Metropolis transition for minimizing a scalar function."""
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    return delta <= 0 or bool(rng.random() < np.exp(-min(delta/temperature, 700.)))


def optimize_function(fn: Callable[[np.ndarray], float], lower: float, upper: float,
                      algorithm: str = 'saipso', seed: int = 0,
                      config: CoreConfig | None = None,
                      fe_budget: int | None = None) -> CoreResult:
    """Optimize a numerical test function with per-evaluation accounting.

    A fixed-iteration experiment uses `iterations` (original paper: 1000).
    For an FE-controlled experiment, stops precisely at `fe_budget`.
    The best-ever record is separate from personal bests accepted by SA.
    All arms share the same initial population for an identical seed.
    """
    cfg = config or CoreConfig()
    if algorithm not in ARMS:
        raise ValueError(f'unknown algorithm {algorithm}')
    if not np.isfinite(lower) or not np.isfinite(upper) or not lower < upper:
        raise ValueError('invalid bounds')
    if fe_budget is not None and fe_budget < cfg.population:
        raise ValueError('FE budget smaller than initial population')
    immune = algorithm in ('ipso','saipso','saipso-no-pbest-sa','saipso-no-local-sa')
    pbest_sa = algorithm in ('saipso','saipso-no-immune','saipso-no-local-sa')
    local_sa = algorithm in ('saipso','saipso-no-immune','saipso-no-pbest-sa')
    rng = np.random.default_rng(seed)
    span = upper-lower
    pos = rng.uniform(lower, upper, size=(cfg.population, cfg.dimension))
    vel = rng.uniform(-1., 1., size=pos.shape)
    evaluations = 0
    best_value = np.inf
    best_x = pos[0].copy()
    trace: list[float] = []
    traces_at: list[int] = []

    def evaluate(x: np.ndarray) -> float:
        nonlocal evaluations, best_x, best_value
        if fe_budget is not None and evaluations >= fe_budget:
            raise RuntimeError('FE budget exceeded')
        v = float(fn(x))
        if not np.isfinite(v):
            raise ValueError('objective returned non-finite result')
        evaluations += 1
        if v < best_value:
            best_value, best_x = v, x.copy()
        trace.append(best_value)
        traces_at.append(evaluations)
        return v

    scores = np.asarray([evaluate(x) for x in pos])
    pbest = pos.copy()
    pscores = scores.copy()
    archive: list[tuple[float, np.ndarray]] = []
    triggers = 0
    worse_accepted = 0
    stagnant = 0

    def remember(x: np.ndarray, f: float) -> None:
        if not immune or cfg.memory == 0:
            return
        if not any(np.array_equal(x, y) for _, y in archive):
            archive.append((float(f),x.copy()))
            archive.sort(key=lambda item: item[0])
            del archive[cfg.memory:]

    remember(best_x, best_value)
    iteration = 0
    while iteration < cfg.iterations:
        if fe_budget is not None and evaluations >= fe_budget:
            break
        old_best = best_value
        w = inertia(iteration, cfg.iterations, cfg.w_max, cfg.w_min)
        r1 = rng.random(pos.shape)
        r2 = rng.random(pos.shape)
        vel = np.clip(w*vel + cfg.c1*r1*(pbest-pos) + cfg.c2*r2*(best_x-pos),-cfg.vmax,cfg.vmax)
        # Evaluate each individual independently to avoid FE overshoot.
        for i in range(cfg.population):
            if fe_budget is not None and evaluations >= fe_budget:
                break
            pos[i] = np.clip(pos[i]+vel[i],lower,upper)
            scores[i] = evaluate(pos[i])
            diff = scores[i]-pscores[i]
            # Assumed pbest-SA cooling schedule; the paper does not specify it.
            temp = max(cfg.t_end, cfg.t0*cfg.alpha**iteration)
            worse = diff > 0 and pbest_sa and metropolis(diff,temp,rng)
            if diff < 0 or worse:
                pbest[i], pscores[i] = pos[i].copy(), scores[i]
                worse_accepted += int(worse)
        if immune and cfg.memory and cfg.memory_injections:
            for i in np.argsort(scores)[:min(3,cfg.population)]:
                remember(pos[i],scores[i])
            if archive:
                pool = np.vstack((pos, [x for _,x in archive]))
                fpool = np.r_[scores, [f for f,_ in archive]]
                choice_p = probability(fpool,cfg.beta)
                worst = np.argsort(scores)[-cfg.memory_injections:]
                for i in worst:
                    if fe_budget is not None and evaluations >= fe_budget:
                        break
                    idx = int(rng.choice(len(pool),p=choice_p))
                    pos[i] = np.clip(pool[idx]+rng.normal(0,.015*span,cfg.dimension),lower,upper)
                    scores[i] = evaluate(pos[i])
                    if scores[i] < pscores[i]:
                        pbest[i],pscores[i] = pos[i].copy(),scores[i]
            remember(best_x,best_value)
        stagnant = stagnant+1 if old_best-best_value <= cfg.epsilon else 0
        if local_sa and stagnant >= cfg.stagnation and (fe_budget is None or evaluations < fe_budget):
            triggers += 1
            stagnant = 0
            trial = best_x.copy()
            trial_f = best_value
            temp = cfg.t0
            # The article only specifies terminal T=10, cooling 0.8.
            # We assume two local proposals at each cooling level.
            while temp > cfg.t_end:
                if fe_budget is not None and evaluations >= fe_budget:
                    break
                for _ in range(cfg.sa_trials_per_level):
                    if fe_budget is not None and evaluations >= fe_budget:
                        break
                    proposed = np.clip(trial+rng.normal(0,cfg.sa_step_fraction*span,cfg.dimension)*
                                       np.sqrt(temp/cfg.t0),lower,upper)
                    candidate_f = evaluate(proposed)
                    if metropolis(candidate_f-trial_f,temp,rng):
                        trial,trial_f = proposed,candidate_f
                temp *= cfg.alpha
            if best_value < float(np.min(scores)):
                i = int(np.argmax(scores))
                pos[i] = pbest[i] = best_x.copy()
                scores[i] = pscores[i] = best_value
            remember(best_x,best_value)
        iteration += 1
    if fe_budget is not None:
        assert evaluations == fe_budget, (evaluations,fe_budget)
    return CoreResult(best_x,best_value,evaluations,iteration,triggers,
                      worse_accepted,np.asarray(trace),np.asarray(traces_at),algorithm,seed)


def benchmark(name: str, dimension: int):
    """Three canonical mathematical functions, not air-defense scenarios."""
    if name=='sphere':
        return lambda x: float(np.dot(x,x)), -5.12, 5.12
    if name=='rastrigin':
        return lambda x: float(10*dimension+np.sum(x*x-10*np.cos(2*np.pi*x))),-5.12,5.12
    if name=='rosenbrock':
        return lambda x: float(np.sum(100*(x[1:]-x[:-1]**2)**2+(1-x[:-1])**2)),-2.048,2.048
    raise ValueError(name)
