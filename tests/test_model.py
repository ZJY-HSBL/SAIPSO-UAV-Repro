import numpy as np
import pytest
from saipso_uav.model import (Scenario, FAST, SAFE, WEIGHTS, bounds, terrain_height,
                              vector_to_route, fitness_details, initial_candidate)


def test_paper_ahp_weight_sets():
    assert FAST.length == 0.6370
    assert SAFE.exposure == 0.6694
    assert abs(sum(vars(SAFE).values())-1) < 2e-4


def test_route_endpoints_and_dimensionality():
    s = Scenario()
    low, high = bounds(s)
    assert low.shape == high.shape == (20,)
    route = vector_to_route(initial_candidate(np.random.default_rng(5), s), s)
    assert route.shape == (12, 3)
    np.testing.assert_allclose(route[0], s.start)
    np.testing.assert_allclose(route[-1], s.goal)


def test_terrain_is_nonnegative_and_has_peaks():
    s = Scenario()
    at_peak = terrain_height(np.array([137.0]), np.array([135.0]), s)[0]
    far = terrain_height(np.array([0.0]), np.array([0.0]), s)[0]
    assert at_peak > 370
    assert 0 <= far < at_peak


def test_fitness_finite_and_deterministic():
    s = Scenario()
    v = initial_candidate(np.random.default_rng(9), s)
    for weight in WEIGHTS.values():
        a = fitness_details(v, s, weight)
        b = fitness_details(v, s, weight)
        assert a == b
        assert np.isfinite(a["fitness"]) and a["fitness"] >= 0


def test_invalid_vector_fails():
    with pytest.raises(ValueError):
        vector_to_route(np.zeros(7), Scenario())
