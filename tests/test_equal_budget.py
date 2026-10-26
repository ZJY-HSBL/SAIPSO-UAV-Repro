import numpy as np
import pytest
from saipso_uav.equal_budget import ALGORITHMS, optimize_budget
from saipso_uav.optimizer import OptimizerConfig
from saipso_uav.model import Scenario, FAST, fitness
from saipso_uav.validation import dense_validation
from saipso_uav.study import civilian_scenarios, paired_bootstrap


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("budget", [6, 7, 39, 65])
def test_exact_budget_and_reproducible(algorithm, budget):
    scenario = Scenario(waypoint_count=3, samples_per_segment=3)
    cfg = OptimizerConfig(population=6, memory=3, sa_max_levels=3)
    a = optimize_budget(scenario, FAST, budget=budget, seed=45, algorithm=algorithm, config=cfg)
    b = optimize_budget(scenario, FAST, budget=budget, seed=45, algorithm=algorithm, config=cfg)
    assert a.evaluations == b.evaluations == budget
    assert len(a.trace) == budget
    np.testing.assert_allclose(a.trace, b.trace)
    assert np.all(np.diff(a.trace) <= 1e-12)
    assert a.fitness == a.trace[-1]
    assert abs(fitness(a.vector, scenario, FAST) - a.fitness) < 1e-10


def test_same_initialization_across_algorithms():
    cfg = OptimizerConfig(population=6)
    values = [optimize_budget(Scenario(waypoint_count=3), FAST, 6, 32, a, cfg).trace for a in ALGORITHMS]
    for x in values[1:]:
        np.testing.assert_allclose(x, values[0])


def test_invalid_budget():
    with pytest.raises(ValueError):
        optimize_budget(Scenario(), FAST, 2, 0, "pso")
    with pytest.raises(ValueError):
        optimize_budget(Scenario(), FAST, 100, 0, "other")


def test_dense_validations_and_holdout_are_distinct():
    scenarios = civilian_scenarios()
    assert scenarios["reference"].hills != scenarios["holdout"].hills
    cfg = OptimizerConfig(population=6)
    result = optimize_budget(Scenario(waypoint_count=3), FAST, 18, 5, "pso", cfg)
    v = dense_validation(result.vector, Scenario(waypoint_count=3), 15)
    assert isinstance(v["sampled_feasible"], bool)
    assert all(isinstance(item, bool) for item in v["checks"].values())


def test_bootstrap_constant_paired_difference():
    lo, hi = paired_bootstrap(np.full(10, 0.25), reps=100)
    assert lo == pytest.approx(0.25)
    assert hi == pytest.approx(0.25)


def test_reference_and_holdout_fixed_endpoints_clearance_possible():
    from saipso_uav.model import terrain_height
    for scene in civilian_scenarios().values():
        for x, y, z in (scene.start, scene.goal):
            assert z - float(terrain_height(x, y, scene)) >= scene.min_clearance
