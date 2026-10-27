import numpy as np
import pytest
from saipso_uav.model import Scenario, FAST
from saipso_uav.route3d import (route_from_controls,route_audit,route_objective,
                                ENCODINGS,INITIALIZATIONS)


def test_waypoints_match_paper_count_and_endpoints():
    s=Scenario(min_clearance=5.)
    p=route_from_controls(np.ones(30)*100,s)
    assert p.shape==(12,3)
    np.testing.assert_array_equal(p[0],s.start)
    np.testing.assert_array_equal(p[-1],s.goal)


def test_x_is_truly_free_and_fixed_variant_ignores_same_control():
    s=Scenario(min_clearance=5.)
    u=np.full(30,100.)
    before=route_from_controls(u,s,'full_xyz')
    u[:10]=200
    after=route_from_controls(u,s,'full_xyz')
    assert (after[1:-1,0]>before[1:-1,0]).all()
    fixed_a=route_from_controls(u,s,'fixed_x')
    u[:10]=0
    fixed_b=route_from_controls(u,s,'fixed_x')
    np.testing.assert_array_equal(fixed_a,fixed_b)


def test_paper_constraints_are_separately_checked():
    s=Scenario(waypoint_count=3,hills=(),zones=(),min_clearance=0.,
               min_segment=3.,max_turn_degrees=85.,max_climb_degrees=60.)
    pts=np.array([[10.,10.,10.],[20.,20.,10.],[30.,30.,10.],[40.,40.,10.],[50.,50.,10.]])
    assert route_audit(pts,s)['feasible']
    q=pts.copy();q[1]=q[0]
    assert not route_audit(q,s)['checks']['minimum_segment']
    q=pts.copy();q[1,2]=150.
    assert not route_audit(q,s)['checks']['maximum_climb']
    q=pts.copy();q[2]=q[1]+[0.,-10.,0.]
    assert not route_audit(q,s)['checks']['maximum_turn']
    q=pts.copy();q[2,2]=-1.
    assert not route_audit(q,s)['checks']['minimum_height']
    shorter=Scenario(waypoint_count=3,hills=(),zones=(),min_clearance=0.,max_length_factor=1.01)
    assert not route_audit(np.array([[10.,10.,10.],[20.,100.,10.],[30.,15.,10.],[40.,100.,10.],[190.,190.,10.]]),shorter)['checks']['maximum_length']


def test_exact_zone_segment_intersection_not_missed_by_sampling():
    s=Scenario(waypoint_count=1,hills=(),zones=((100.,100.,.01),),min_clearance=0.)
    route=np.array([[10.,10.,10.],[100.,100.,10.],[190.,190.,10.]])
    z=route_audit(route,s,samples=3)
    assert z['zone_margin']<0
    assert not z['checks']['restricted_airspace']


def test_terrain_clearance_independent_from_absolute_floor():
    s=Scenario(waypoint_count=1,hills=((100.,100.,120.,15.),),zones=(),min_clearance=5.)
    r=np.array([[10.,10.,10.],[100.,100.,25.],[190.,190.,10.]])
    z=route_audit(r,s)
    assert z['checks']['minimum_height']
    assert not z['checks']['terrain_clearance']


@pytest.mark.parametrize('encoding',ENCODINGS)
@pytest.mark.parametrize('initialization',INITIALIZATIONS)
def test_objective_finite_and_reproducible(encoding,initialization):
    s=Scenario(min_clearance=5.)
    p=route_from_controls(np.arange(30)*6.5,s,encoding,initialization)
    a=route_objective(p,s,FAST)
    assert np.isfinite(a) and a==route_objective(p,s,FAST)


def test_validation_input_shape():
    s=Scenario(min_clearance=5.)
    with pytest.raises(ValueError): route_from_controls(np.zeros(20),s)
    with pytest.raises(ValueError): route_audit(np.zeros((11,3)),s)
