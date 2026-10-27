"""v0.5: civilian 3-D 10-waypoint representation and five-flight-constraint audit.

Neither task cost nor baseline geometry reproduces unpublished MATLAB maps.
"""
from __future__ import annotations
import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
from .model import WEIGHTS
from .paper_core import CoreConfig, optimize_function
from .route3d import (route_from_controls,route_audit,route_objective,
                       ENCODINGS, INITIALIZATIONS)
from .study_v03 import scenes

ARMS=('pso','ipso','saipso')
SCENES=('reference','frozen_novel')


def run(output: Path, seeds: list[int] | None=None, budget: int=600,
        make_plots: bool=True) -> dict:
    seeds=list(range(800,810)) if seeds is None else list(seeds)
    if not seeds or len(set(seeds))!=len(seeds) or any(x<0 for x in seeds) or budget<20:
        raise ValueError('distinct nonnegative seeds and budget>=20 required')
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    traces={}
    paths={}
    maps={k:scenes()[k] for k in SCENES}
    total=len(maps)*len(WEIGHTS)*len(INITIALIZATIONS)*len(ENCODINGS)*len(ARMS)*len(seeds)
    for scene_name,scene in maps.items():
        for task,weight in WEIGHTS.items():
            for baseline in INITIALIZATIONS:
                for encoding in ENCODINGS:
                    for arm in ARMS:
                        for seed in seeds:
                            cfg=replace(CoreConfig(),dimension=3*scene.waypoint_count)
                            def fn(u):
                                path=route_from_controls(u,scene,encoding,baseline)
                                return route_objective(path,scene,weight,samples=13)
                            result=optimize_function(fn,0.,200.,arm,seed,cfg,fe_budget=budget)
                            route=route_from_controls(result.x,scene,encoding,baseline)
                            audit=route_audit(route,scene,samples=201)
                            record=dict(scene=scene_name,task=task,initialization=baseline,
                                encoding=encoding,algorithm=arm,seed=seed,
                                budget=budget,evaluations=result.fe,
                                iterations=result.iterations,sa_triggers=result.sa_triggers,
                                best_objective=result.value,initial_population_best=float(result.history[cfg.population-1]),
                                gain_over_initial=float(result.history[cfg.population-1]-result.value),
                                audited_feasible=audit['feasible'],
                                audit_violation=audit['total_violation'],
                                length=audit['length'],length_ratio=audit['length_ratio'],
                                min_clearance=audit['min_clearance'],zone_margin=audit['zone_margin'],
                                max_turn_deg=audit['max_turn_deg'],max_climb_deg=audit['max_climb_deg'],
                                **{f'pass_{k}':v for k,v in audit['checks'].items()})
                            rows.append(record)
                            traces[(scene_name,task,baseline,encoding,arm,seed)]=result.history
                            # store best path in each group, purely for illustrative visuals
                            key=(scene_name,task,baseline,encoding,arm)
                            if key not in paths or record['best_objective']<paths[key][0]:
                                paths[key]=(record['best_objective'],route)
                            if len(rows)%30==0:
                                print(f'v0.5 {len(rows)}/{total} runs',flush=True)
    with (output/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[]
    paired=[]
    for scene in maps:
        for task in WEIGHTS:
            for initialization in INITIALIZATIONS:
                groups={}
                for encoding in ENCODINGS:
                    for algorithm in ARMS:
                        selected=[r for r in rows if r['scene']==scene and r['task']==task and r['initialization']==initialization and r['encoding']==encoding and r['algorithm']==algorithm]
                        groups[(encoding,algorithm)]={r['seed']:r for r in selected}
                        summary.append(dict(scene=scene,task=task,initialization=initialization,
                            encoding=encoding,algorithm=algorithm,n=len(selected),
                            mean_objective=float(np.mean([r['best_objective'] for r in selected])),
                            feasible_count=sum(r['audited_feasible'] for r in selected),
                            feasible_rate=float(np.mean([r['audited_feasible'] for r in selected])),
                            mean_improvement=float(np.mean([r['gain_over_initial'] for r in selected])),
                            median_violation=float(np.median([r['audit_violation'] for r in selected]))))
                for arm in ARMS:
                    full=groups[('full_xyz',arm)];locked=groups[('fixed_x',arm)]
                    diffs=[full[s]['best_objective']-locked[s]['best_objective'] for s in seeds]
                    feas=[int(full[s]['audited_feasible'])-int(locked[s]['audited_feasible']) for s in seeds]
                    paired.append(dict(scene=scene,task=task,initialization=initialization,
                                       algorithm=arm,n=len(seeds),mean_full_minus_fixed=float(np.mean(diffs)),
                                       feasibility_full_minus_fixed=float(np.mean(feas))))
    manifest=dict(version='v0.5.0',scope='civilian benchmark; full xyz vs fixed x 10-interior-waypoint representation',
        author_experiment_reproduced=False,source_paper='Fan Weilun, Ship Electronic Engineering 2026 (6)',
        total_runs=len(rows),equal_fe_runs=len(rows),evaluation_budget=budget,seeds=seeds,
        task_weights={k:asdict(w) for k,w in WEIGHTS.items()},
        optimization=asdict(replace(CoreConfig(),dimension=30)),
        scene_definitions={k:asdict(v) for k,v in maps.items()},
        encodings=list(ENCODINGS),initializations=list(INITIALIZATIONS),algorithms=list(ARMS),
        audit_samples_per_segment=201,optimization_samples_per_segment=13,
        summary=summary,paired_contrasts=paired,
        limitations=[
            'The paper encodes each intermediate waypoint in 3D; 30 controls for 10 intermediate points are an explicit interpretation, not author source.',
            'Full x coordinates vary within +/-32% of nominal x spacing, not unrestricted physical coordinates.',
            'The fixed-x ablation keeps 30 controls and ignores 10 neutral x controls to share initial RNG; the added neutral dimensions may disadvantage the ablation.',
            'Two corridor/straight baselines are chosen heuristic initializations; some results may be caused by the initialization rather than optimization.',
            'The same 500/600 FE budget does not equalize CPU time or initialization feasibility across baselines.',
            'Published paper has five flight constraints; study separately adds terrain clearance, civilian restricted-airspace and height ceiling.',
            'Civilian exposure is not radar, missile or artillery threat and should not be compared numerically to paper figures.',
            'Terrain clearance is sampled, not mathematically proven between samples; circular-zone segment intersection is exact.',
            'Reference and frozen_novel are both previously inspected in v0.3.0, NOT statistically independent unseen terrain.',
            'Only exploratory paired means; sample size insufficient for claims of significance or general superiority.',
            'Motion constraints are simplified polyline proxies; flight dynamics and real-world safety are not certified.',
        ])
    (output/'summary.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    if make_plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from .model import terrain_height
        fig,axs=plt.subplots(1,2,figsize=(12,5))
        for j,baseline in enumerate(INITIALIZATIONS):
            lab=[s for s in summary if s['scene']=='reference' and s['task']=='fast' and s['initialization']==baseline]
            names=[x['encoding']+' / '+x['algorithm'] for x in lab]
            axs[j].barh(names,[x['feasible_rate'] for x in lab]);axs[j].set(xlim=(0,1),xlabel='Audited feasible fraction',title=f'Reference / fast / {baseline}')
        fig.tight_layout();fig.savefig(output/'feasibility.svg');plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,5.3))
        for algorithm in ARMS:
            for enc in ENCODINGS:
                h=np.stack([traces[('reference','fast','corridor',enc,algorithm,seed)] for seed in seeds])
                ax.plot(np.arange(1,budget+1),np.median(h,axis=0),label=f'{algorithm} {enc}')
        ax.set(xlabel='Objective evaluation (FE)',ylabel='Median best-so-far penalized cost',title=f'10 waypoint civilian benchmark / corridor / n={len(seeds)}')
        ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(output/'convergence.svg');plt.close(fig)
        fig=plt.figure(figsize=(12,7.5));ax=fig.add_subplot(projection='3d')
        sc=maps['reference']; mesh=np.linspace(*sc.world_xy,95);gx,gy=np.meshgrid(mesh,mesh)
        gz=terrain_height(gx,gy,sc)
        ax.plot_surface(gx,gy,gz,cmap='terrain',alpha=.48,linewidth=0,antialiased=False)
        for encoding in ENCODINGS:
            path=paths[('reference','safe','corridor',encoding,'saipso')][1]
            ax.plot(*path.T,label=f'{encoding} SA-IPSO')
        ax.scatter([sc.start[0],sc.goal[0]],[sc.start[1],sc.goal[1]],[sc.start[2],sc.goal[2]],s=26,color='black')
        ax.set(xlabel='X',ylabel='Y',zlabel='Height',title='Illustrative civilian terrain, selected optimized paths')
        ax.legend();fig.tight_layout();fig.savefig(output/'routes_3d.svg');plt.close(fig)
    return manifest


def main():
    p=argparse.ArgumentParser(description='Civilian SA-IPSO full XYZ waypoint encoding study')
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(800,810)))
    p.add_argument('--budget',type=int,default=600)
    p.add_argument('--output',type=Path,default=Path('results/route3d_v05'))
    p.add_argument('--no-plots',action='store_true')
    a=p.parse_args();d=run(a.output,a.seeds,a.budget,not a.no_plots)
    print('completed',d['total_runs'],'runs')

if __name__=='__main__':main()
