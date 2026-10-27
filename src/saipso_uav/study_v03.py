"""Pre-specified FE-matched ablations for civilian feasibility priority (v0.3)."""
from __future__ import annotations
import argparse
import csv
import json
import platform
import sys
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np

from .feasible import ARMS, optimize_feasible, feasibility_metrics, segment_circle_clearance
from .model import Scenario, WEIGHTS, fitness_details, vector_to_route
from .study import civilian_scenarios, paired_bootstrap
from .validation import dense_validation


def scenes() -> dict[str, Scenario]:
    previous = civilian_scenarios()
    # The v0.2 holdout was inspected during method development. Accordingly,
    # it is now a *development stress test*, not a claim of untouched data.
    novel = replace(previous['reference'], name='frozen-novel-scene', hills=(
        (48., 90., 205., 24.), (112., 135., 270., 22.), (152., 154., 130., 17.),
    ), zones=((52., 57., 19.), (152., 67., 23.), (110., 170., 13.)))
    return {'reference': previous['reference'], 'development': previous['holdout'],
            'frozen_novel': novel}


def independent_audit(vector: np.ndarray, scenario: Scenario) -> dict:
    """Densely sampled terrain, exact segment-to-disk restricted-zone geometry."""
    dense = dense_validation(vector, scenario, samples=201)
    route = vector_to_route(vector, scenario)
    zone_margin = min((segment_circle_clearance(route, *z) for z in scenario.zones), default=float('inf'))
    dense['checks']['restricted_airspace'] = bool(zone_margin >= -1e-9)
    dense['sampled_feasible'] = bool(all(dense['checks'].values()))
    dense['zone_signed_margin'] = float(zone_margin)
    return dense


def run(output: Path, budget: int = 500, seeds: list[int] | None = None, plot: bool = True) -> dict:
    if seeds is None:
        seeds = list(range(300, 330))  # 30 new pre-specified seeds
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0 or budget < 20:
        raise ValueError('distinct nonnegative seeds and budget>=20 required')
    scs = scenes()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    hist: dict[tuple[str, str, str], list[np.ndarray]] = {}
    hist_v: dict[tuple[str, str, str], list[np.ndarray]] = {}
    total = len(scs)*len(WEIGHTS)*len(ARMS)*len(seeds)
    for sc_name, scenario in scs.items():
        for task, weights in WEIGHTS.items():
            for seed in seeds:
                for arm in ARMS:
                    result = optimize_feasible(scenario, weights, budget, seed, arm)
                    detail = fitness_details(result.vector, scenario, weights)
                    audit = independent_audit(result.vector, scenario)
                    # Train-grid feasibility is intentionally reported separately.
                    row = {'scene': sc_name, 'task': task, 'seed': seed,
                           'arm': arm, 'budget': budget, 'evaluations': result.evaluations,
                           'fitness': result.score, 'violation_61': result.violation,
                           'feasible_61': result.feasible,
                           'audited_feasible': audit['sampled_feasible'],
                           'first_feasible_fe_61': result.first_feasible_fe,
                           'best_feasible_fitness_61': result.best_feasible_score,
                           'sa_calls': result.sa_calls,
                           'min_dense_clearance': audit['min_dense_clearance'],
                           'zone_signed_margin': audit['zone_signed_margin'],
                           'min_sampled_violation_61': float(np.min(result.history_violation)),
                           **{f'pass_{name}': yes for name, yes in audit['checks'].items()},
                           'length': detail['length'], 'length_ratio': detail['length_ratio']}
                    rows.append(row)
                    hist.setdefault((sc_name, task, arm), []).append(result.history_feasible)
                    hist_v.setdefault((sc_name, task, arm), []).append(result.history_violation)
                    if len(rows) % 60 == 0:
                        print(f'v0.3: {len(rows)}/{total} runs completed',flush=True)
    with (output/'metrics.csv').open('w', encoding='utf-8', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    analyses = []
    comparisons = []
    for scene in scs:
        for task in WEIGHTS:
            group = [r for r in rows if r['scene'] == scene and r['task'] == task]
            for arm in ARMS:
                rs = [r for r in group if r['arm'] == arm]
                obj = np.array([r['fitness'] for r in rs],dtype=float)
                feas = [r for r in rs if r['audited_feasible']]
                analyses.append({'scene':scene,'task':task,'arm':arm,'n':len(rs),
                                 'audited_feasible_count':len(feas),
                                 'audited_feasible_rate':len(feas)/len(rs),
                                 'mean_objective_all':float(np.mean(obj)),
                                 'std_objective_all':float(np.std(obj,ddof=1)) if len(obj)>1 else None,
                                 'mean_feasible_objective':float(np.mean([r['fitness'] for r in feas])) if feas else None,
                                 'mean_violation_61':float(np.mean([r['violation_61'] for r in rs])),
                                 'median_first_feasible_fe_61':float(np.median([r['first_feasible_fe_61'] for r in rs
                                   if r['first_feasible_fe_61'] is not None])) if any(r['first_feasible_fe_61'] is not None for r in rs) else None})
            # Paired differences under identical seeds; binary feasibility is primary.
            ctrl = {r['seed']: r for r in group if r['arm']=='soft-saipso'}
            for arm in ARMS[1:]:
                cur = {r['seed']:r for r in group if r['arm']==arm}
                gains = np.asarray([float(cur[s]['audited_feasible'])-float(ctrl[s]['audited_feasible']) for s in seeds])
                lo, hi = paired_bootstrap(gains, reps=5000, seed=8614)
                comparisons.append({'scene':scene,'task':task, 'contrast':f'{arm}-soft-saipso',
                                    'paired_n':len(seeds), 'feasibility_gain':float(gains.mean()),
                                    'bootstrap_95_exploratory':[lo,hi]})
    manifest = {'version':'v0.3.0', 'purpose':'civilian feasibility priority, NOT military evasion or flight qualification',
                'method_status':'independent algorithm extension; not original paper formulation',
                'seed_origin':'fresh seeds 300..329, selected before v0.3 evaluation',
                'objective_evaluations_per_run':budget,'total_runs':len(rows), 'seeds':seeds,
                'arms':list(ARMS), 'runtime':{'python':sys.version.split()[0], 'numpy':np.__version__, 'platform':platform.platform()}, 'search_constraint_samples_per_segment':61, 'audit_constraint_samples_per_segment':201, 'scenarios':{k:asdict(s) for k,s in scs.items()},
                'primary_metric':'independent audited feasibility rate (201 segment samples and exact circular-zone intersections)',
                'secondary_metrics':['best feasible objective','first feasible FE','constraint violation','wall time not standardized'],
                'summary':analyses,'contrasts':comparisons,
                'limitations':['The fixed detour heuristic is domain-specific and is responsible for immediate feasibility in some scenes',
                               'v0.2 holdout was used for development, renamed development here',
                               'frozen_novel scenario synthetic, not real-world independent distribution',
                               '201 samples per straight segment do not formally certify continuous terrain clearance',
                               'fixed evaluation count does not equalize internal CPU cost',
                               'contrast confidence intervals are exploratory and unadjusted for multiplicity']}
    (output/'summary.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    if plot:
        from .study_v03_plots import plots
        plots(output, rows, analyses, hist, hist_v, scs, budget)
    return manifest


def main():
    p=argparse.ArgumentParser(description='Civilian equal-FE feasibility-first ablation')
    p.add_argument('--budget',type=int,default=500)
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(300,330)))
    p.add_argument('--output',type=Path,default=Path('results/feasibility_v03'))
    p.add_argument('--no-plots',action='store_true')
    a=p.parse_args()
    x=run(a.output,a.budget,a.seeds,not a.no_plots)
    print(f"v0.3 completed: {x['total_runs']} runs, {x['objective_evaluations_per_run']} FE each")

if __name__ == '__main__':
    main()
