"""Mechanism-level invariants and method-specific ablations."""
import numpy as np
import pytest
from saipso_uav.paper_core import (CoreConfig, ARMS, inertia, probability,
                                   metropolis, optimize_function, benchmark)


def test_linear_inertia_paper_formula():
    assert inertia(0,1000)==pytest.approx(.9)
    assert inertia(500,1000)==pytest.approx(.65)
    assert inertia(1000,1000)==pytest.approx(.4)


def test_immune_probability_paper_formula():
    x=np.array([1.,3.,4.])
    a=(1/x)/(1/x).sum()
    d=np.abs(x[:,None]-x[None,:]).sum(axis=1)
    b=d/d.sum()
    np.testing.assert_allclose(probability(x), .5*a+.5*b)
    np.testing.assert_allclose(probability(np.ones(4)),np.full(4,.25))
    assert np.isclose(probability(np.array([-3.,0.,3.])).sum(),1)
    with pytest.raises(ValueError): probability(np.array([]))


def test_metropolis_accepts_improvement():
    assert metropolis(-5,1,np.random.default_rng(1))
    assert not metropolis(1e8,1,np.random.default_rng(1))


@pytest.mark.parametrize('arm',ARMS)
def test_exact_budget_determinism_and_best_ever(arm):
    f,lo,hi=benchmark('rastrigin',6)
    cfg=CoreConfig(population=6,memory=3,dimension=6,iterations=200)
    a=optimize_function(f,lo,hi,arm,seed=41,config=cfg,fe_budget=111)
    b=optimize_function(f,lo,hi,arm,seed=41,config=cfg,fe_budget=111)
    assert a.fe==b.fe==len(a.history)==111
    assert a.value==pytest.approx(f(a.x))
    np.testing.assert_allclose(a.history,b.history)
    assert np.all(np.diff(a.history)<=1e-12)


def test_full_paper_iteration_count_and_parameters():
    cfg=CoreConfig()
    assert (cfg.population,cfg.memory,cfg.dimension,cfg.iterations)==(20,10,10,1000)
    assert (cfg.c1,cfg.c2,cfg.vmax,cfg.stagnation)==(2.05,2.05,20,2)
    assert (cfg.t0,cfg.t_end,cfg.alpha,cfg.epsilon)==(2000,10,.8,1e-4)
    f,lo,hi=benchmark('sphere',10)
    a=optimize_function(f,lo,hi,'pso',seed=3,config=cfg)
    assert a.iterations==1000 and a.fe==20*(1000+1)


def test_invalid_bounds_budget_and_objective():
    with pytest.raises(ValueError): optimize_function(lambda x:0,1,1)
    with pytest.raises(ValueError): optimize_function(lambda x:0,-1,1,fe_budget=2)
    with pytest.raises(ValueError): optimize_function(lambda x:float('nan'),-1,1)
