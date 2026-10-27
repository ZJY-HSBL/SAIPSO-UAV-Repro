"""Paper-style 10-interior-waypoint 3-D encoding in civilian airspace.

This is a route encoding and constraint verification benchmark, NOT the
unpublished MATLAB route planner. Military threat functions are excluded.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .model import Scenario, Weights, terrain_height
from .feasible import segment_circle_clearance, seed_corridor

ENCODINGS = ('full_xyz', 'fixed_x')
INITIALIZATIONS = ('straight', 'corridor')


def route_from_controls(controls: np.ndarray, scenario: Scenario,
                        encoding: str = 'full_xyz', initialization: str = 'straight') -> np.ndarray:
    """Decode a 3*K decision vector as x, y, z waypoint controls.

    Both arms intentionally expose the same 3*K optimization dimensions and
    initial random numbers; fixed_x ignores the x controls (neutral variables).
    This pairs initialization across the two encodings but gives fixed_x
    an explicitly disclosed neutral-dimension disadvantage.
    
    0..200 controls map into bounded *offsets* about a prespecified baseline;
    this is NOT unrestricted optimization of the authors' 3-D coordinates.
    """
    if encoding not in ENCODINGS or initialization not in INITIALIZATIONS:
        raise ValueError('unknown route encoding or baseline')
    k = scenario.waypoint_count
    u = np.asarray(controls, dtype=float)
    if u.shape != (3*k,) or not np.all(np.isfinite(u)):
        raise ValueError('expected a finite 3*K control vector')
    start, goal = np.asarray(scenario.start), np.asarray(scenario.goal)
    fractions = np.linspace(0., 1., k+2)[1:-1]
    baseline = start[None, :] + fractions[:, None] * (goal-start)[None, :]
    x0 = baseline[:, 0]
    if initialization == 'corridor':
        corridor = seed_corridor(scenario,17.,70.)
        baseline[:, 1] = corridor[:k]
        baseline[:, 2] = corridor[k:]
        ys, zs = 14., 20.
    else:
        baseline[:, 2] = np.maximum(34., terrain_height(x0, baseline[:, 1],scenario)+23.)
        ys, zs = 42., 32.
    ux,uy,uz = (u.reshape(3,k)-100.)/100.
    step = abs(goal[0]-start[0])/(k+1)
    xs = np.clip(x0 + (ux*.32*step if encoding == 'full_xyz' else 0.), *scenario.world_xy)
    ys = np.clip(baseline[:,1] + uy*ys, *scenario.world_xy)
    zs = np.clip(baseline[:,2] + uz*zs, 5., scenario.max_altitude)
    return np.vstack((start, np.column_stack((xs,ys,zs)),goal))


def route_audit(route: np.ndarray, scenario: Scenario, samples: int = 201) -> dict:
    """Paper's five flight constraints + independent civilian safety checks.

    Terrain clearance is tested at `samples` points PER segment, not proven
    continuously. Civilian circular-zone intersections use exact segment-disk
    geometry; no defense-system evasion model is used.
    """
    route = np.asarray(route,dtype=float)
    if route.shape != (scenario.waypoint_count+2,3) or not np.all(np.isfinite(route)):
        raise ValueError('invalid 3-D polyline geometry')
    if samples < 3:
        raise ValueError('minimum three samples per segment')
    edges = np.diff(route,axis=0)
    lengths = np.linalg.norm(edges,axis=1)
    length = float(lengths.sum())
    direct = float(np.linalg.norm(route[-1]-route[0]))
    turns=np.degrees(np.arccos(np.clip(np.sum(edges[:-1]*edges[1:],axis=1)/
                 np.maximum(lengths[:-1]*lengths[1:],1e-12),-1.,1.)))
    climb=np.degrees(np.arctan2(np.abs(edges[:,2]),np.maximum(np.linalg.norm(edges[:,:2],axis=1),1e-12)))
    t=np.linspace(0.,1.,samples,endpoint=False)
    pts=(route[:-1,None,:]+t[None,:,None]*edges[:,None,:]).reshape(-1,3)
    pts=np.vstack((pts,route[-1]))
    clearance=pts[:,2]-terrain_height(pts[:,0],pts[:,1],scenario)
    zone_margin=min((segment_circle_clearance(route,*z) for z in scenario.zones),default=200.)
    min_segment=float(np.min(lengths)); max_turn=float(np.max(turns)); max_climb=float(np.max(climb))
    min_height=float(np.min(pts[:,2])); min_clearance=float(np.min(clearance)); max_height=float(np.max(pts[:,2]))
    max_len=scenario.max_length_factor*direct
    violation={
        'minimum_segment':max(0.,scenario.min_segment-min_segment)/max(scenario.min_segment,1.),
        'maximum_turn':max(0.,max_turn-scenario.max_turn_degrees)/90.,
        'maximum_length':max(0.,length-max_len)/max(max_len,1.),
        'maximum_climb':max(0.,max_climb-scenario.max_climb_degrees)/90.,
        'minimum_height':max(0.,10.-min_height)/10.,
        # Civilian baseline: additional ground clearance, airspace and ceiling.
        'terrain_clearance':max(0.,scenario.min_clearance-min_clearance)/50.,
        'restricted_airspace':max(0.,-zone_margin)/25.,
        'altitude_ceiling':max(0.,max_height-scenario.max_altitude)/50.,
    }
    checks={k:v <= 1e-8 for k,v in violation.items()}
    return dict(feasible=bool(all(checks.values())),checks=checks,components=violation,
                total_violation=float(sum(violation.values())),length=length,
                length_ratio=length/direct,min_segment=min_segment,max_turn_deg=max_turn,
                max_climb_deg=max_climb,min_height=min_height,min_clearance=min_clearance,
                zone_margin=float(zone_margin),max_altitude=max_height)


def route_objective(route: np.ndarray, scenario: Scenario, weights: Weights,
                    samples: int = 13) -> float:
    """Transparent civilian objective with a fixed exterior penalty.

    Only flight/terrain/restricted-airspace geometry; feasibility is audited
    again using denser sampling outside the counted optimization FE budget.
    """
    a=route_audit(route,scenario,samples=samples)
    edges=np.diff(route,axis=0)
    t=np.linspace(0,1,samples,endpoint=False)
    points=(route[:-1,None,:]+t[None,:,None]*edges[:,None,:]).reshape(-1,3)
    clearance=points[:,2]-terrain_height(points[:,0],points[:,1],scenario)
    near=np.maximum(0,scenario.comfort_clearance-clearance)/max(scenario.comfort_clearance,1.)
    near=np.minimum(near,8.)
    civilian_risk=float(np.mean(near*near))
    for x,y,r in scenario.zones:
        rr=np.maximum(0,1.-np.hypot(points[:,0]-x,points[:,1]-y)/r)
        civilian_risk+=float(np.mean(rr*rr))
    # Constraints remain explicit and have a common penalty across arms.
    return (weights.length*a['length_ratio']+weights.exposure*civilian_risk+
            weights.constraints*a['total_violation']+12.*a['total_violation'])
