import numpy as np
import pytest
from saipso_uav.model import Scenario, FAST, initial_candidate, fitness
from saipso_uav.optimizer import OptimizerConfig, optimize, immune_probabilities


def test_selection_normalization_and_equal_fitness_guard():
    p = immune_probabilities(np.ones(6))
    np.testing.assert_allclose(p, np.full(6, 1/6))
    p = immune_probabilities(np.array([1.0,2.0,5.0]))
    assert np.isclose(p.sum(), 1.0) and np.all(p >= 0)


def test_algorithms_are_reproducible_and_monotone():
    s = Scenario(waypoint_count=3, samples_per_segment=3)
    conf = OptimizerConfig(population=6, memory=3, iterations=5, sa_max_levels=3)
    for name in ("pso", "ipso", "saipso"):
        a = optimize(s, FAST, conf, seed=42, algorithm=name)
        b = optimize(s, FAST, conf, seed=42, algorithm=name)
        np.testing.assert_allclose(a.history, b.history)
        assert len(a.history) == conf.iterations+1
        assert np.all(np.diff(a.history) <= 1e-12)
        assert abs(fitness(a.best_vector, s, FAST)-a.best_fitness) < 1e-9


def test_invalid_algorithm():
    with pytest.raises(ValueError):
        optimize(Scenario(), FAST, OptimizerConfig(iterations=1), algorithm="unknown")
