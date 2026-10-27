import numpy as np
import pytest
from saipso_uav.feasible import (ARMS, EvalRecord, dominates, optimize_feasible,
                                segment_circle_clearance, seed_corridor,
                                deterministic_repair, feasibility_metrics)
from saipso_uav.study_v03 import independent_audit, scenes, run
from saipso_uav.model import Scenario, FAST


def test_exact_circle_segment_intersection():
    p=np.array([[0.,0.,0.],[10.,0.,0.]])
    assert segment_circle_clearance(p,5,0,2) == pytest.approx(-2)
    assert segment_circle_clearance(p,5,3,2) == pytest.approx(1)


def test_deb_feasibility_domination():
    safe=EvalRecord(2.0, 0.0, True)
    collision=EvalRecord(.1, .01, False)
    assert dominates(safe, collision, True)
    assert not dominates(safe, collision, False)
    assert dominates(EvalRecord(9,0.1,False),EvalRecord(1,0.2,False),True)


@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.parametrize('budget',[20,43,90])
def test_equal_budget_determinism(arm,budget):
    sc=Scenario(waypoint_count=5,min_clearance=5)
    a=optimize_feasible(sc,FAST,budget,seed=53,arm=arm)
    b=optimize_feasible(sc,FAST,budget,seed=53,arm=arm)
    assert a.evaluations == b.evaluations == budget
    np.testing.assert_allclose(a.history_fitness,b.history_fitness)
    np.testing.assert_allclose(a.history_violation,b.history_violation)
    assert np.all(np.diff(a.history_violation) <= 1e-10) if arm!='soft-saipso' else True
    if a.feasible:
        assert a.violation == pytest.approx(0)
        assert a.first_feasible_fe is not None
    assert not np.any(np.isinf(a.history_fitness))


def test_repair_is_deterministic_bounded_and_not_implicit_evaluation():
    sc=Scenario()
    c=seed_corridor(sc,17,70)
    a=deterministic_repair(np.ones(len(c))*300,sc,c,1)
    np.testing.assert_allclose(a,c)
    assert np.isfinite(feasibility_metrics(c,sc)['violation'])
    with pytest.raises(ValueError):
        deterministic_repair(c,sc,c,1.2)


def test_reference_corridor_is_sampled_feasible():
    sc=scenes()['reference']
    assert independent_audit(seed_corridor(sc,17,40),sc)['sampled_feasible']


def test_no_hidden_v03_holdout_claim():
    assert 'development' in scenes()
    assert 'frozen_novel' in scenes()


def test_small_study_export(tmp_path):
    doc=run(tmp_path,budget=20,seeds=[991],plot=False)
    assert doc['total_runs']==len(ARMS)*len(scenes())*2
    assert (tmp_path/'metrics.csv').is_file()
    assert (tmp_path/'summary.json').is_file()
