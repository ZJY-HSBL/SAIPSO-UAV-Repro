"""Paper-parameter mechanism reproduction, evaluated on generic benchmarks.

Intentionally isolates optimization mechanisms from military UAV scenarios.
No claims of author code, MATLAB figures, or unique undocumented operators.
"""
from __future__ import annotations
import argparse
import csv
import json
import platform
from pathlib import Path
from dataclasses import asdict
from statistics import mean, stdev
import numpy as np
from .paper_core import ARMS, CoreConfig, benchmark, optimize_function

FUNCTIONS=('sphere','rastrigin','rosenbrock')


def paired_bootstrap(values, repetitions=3000, seed=1947):
    diffs=np.asarray(values,dtype=float)
    rng=np.random.default_rng(seed)
    picks=rng.integers(0,len(diffs),size=(repetitions,len(diffs)))
    return np.percentile(np.mean(diffs[picks],axis=1),[2.5,97.5]).tolist()


def run(out: Path, seeds: list[int] | None = None, budget: int = 2500,
        make_plots: bool = True, paper_iterations: bool = True) -> dict:
    """Two protocols: strict 1000 iterations (one seed per objective)
    and exact 2500 FE comparisons (paired fresh seeds)."""
    seeds=list(range(700,712)) if seeds is None else list(seeds)
    if not seeds or len(set(seeds))!=len(seeds) or min(seeds)<0 or budget<20:
        raise ValueError('invalid seeds or FE budget')
    out.mkdir(parents=True,exist_ok=True)
    cfg=CoreConfig()  # all paper-stated numeric parameters
    data=[]
    traces={}
    for fun in FUNCTIONS:
        fn, lo, hi=benchmark(fun,cfg.dimension)
        for seed in seeds:
            for arm in ARMS:
                r=optimize_function(fn,lo,hi,arm,seed,cfg,fe_budget=budget)
                data.append(dict(protocol='equal_fe',function=fun,seed=seed,algorithm=arm,
                                 best=r.value,fe=r.fe,iterations=r.iterations,
                                 sa_triggers=r.sa_triggers,pbest_worse_accepted=r.pbest_worse_accepted))
                traces.setdefault((fun,arm),[]).append(r.history)
        print(f'equal-FE {fun}: {len(seeds)*len(ARMS)} runs completed',flush=True)
    if paper_iterations:
        # Exact 1000-iteration protocol, same original hyperparameters for each arm.
        # SA/IPSO arms expend extra FEs: never compare them as equal-budget.
        for fun in FUNCTIONS:
            fn,lo,hi=benchmark(fun,cfg.dimension)
            for arm in ('pso','ipso','saipso'):
                r=optimize_function(fn,lo,hi,arm,seed=700,config=cfg)
                data.append(dict(protocol='paper_1000_iterations',function=fun,seed=700,
                                 algorithm=arm,best=r.value,fe=r.fe,iterations=r.iterations,
                                 sa_triggers=r.sa_triggers,pbest_worse_accepted=r.pbest_worse_accepted))
            print(f'paper-1000 {fun}: 3 runs completed',flush=True)
    with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]))
        w.writeheader();w.writerows(data)
    summary=[]
    contrasts=[]
    for fun in FUNCTIONS:
        by_arm={arm: [d for d in data if d['protocol']=='equal_fe' and d['function']==fun and d['algorithm']==arm] for arm in ARMS}
        for arm, rows in by_arm.items():
            xs=[r['best'] for r in rows]
            summary.append(dict(function=fun,algorithm=arm,n=len(xs),mean=float(mean(xs)),
                                sd=float(stdev(xs)) if len(xs)>1 else None,
                                median=float(np.median(xs)),min=float(min(xs)),max=float(max(xs)),
                                mean_sa_triggers=float(mean(r['sa_triggers'] for r in rows)),
                                mean_pbest_worse_accept=float(mean(r['pbest_worse_accepted'] for r in rows))))
        base={r['seed']:r['best'] for r in by_arm['saipso']}
        for arm in ARMS:
            if arm=='saipso':continue
            cmp={r['seed']:r['best'] for r in by_arm[arm]}
            changes=[base[seed]-cmp[seed] for seed in seeds]
            contrasts.append(dict(function=fun,comparison='saipso_minus_'+arm,
                                  n=len(changes),mean_difference=float(np.mean(changes)),
                                  bootstrap_95_exploratory=paired_bootstrap(changes)))
    obj=dict(version='v0.4.0',mode='paper-parameter algorithmic verification, generic functions',
             paper_hyperparameters=asdict(cfg),seeds=seeds,equal_fe_budget=budget,
             equal_fe_runs=len(seeds)*len(ARMS)*len(FUNCTIONS),
             fixed_iteration_runs=9 if paper_iterations else 0,
             total_runs=len(data),functions=list(FUNCTIONS),arms=list(ARMS),
             runtime=dict(python=platform.python_version(),numpy=np.__version__),
             summary=summary,paired_contrasts=contrasts,
             clarification='No original MATLAB implementation, terrain map, military threat reproduction, or author reported numerical replication.',
             assumptions=['Immune-archive size and elite replacement are specified only qualitatively in article',
                          'SA neighborhood Gaussian perturbation, 2 trials per temperature, and memory injection of two candidates are independently defined',
                          'Affinities with nonpositive fitness and zero diversity are protected numerically',
                          'Fixed-iteration protocols compare algorithms with different total FEs',
                          'Ablation intervals are exploratory, unadjusted, and 12 paired seeds are insufficient for strong significance conclusions'])
    (out/'summary.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    if make_plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for fun in FUNCTIONS:
            fig,ax=plt.subplots(figsize=(8.5,5))
            for arm in ('pso','ipso','saipso','saipso-no-local-sa'):
                cs=np.stack(traces[(fun,arm)])
                ax.plot(np.arange(1,budget+1),np.median(cs,axis=0),label=arm)
            ax.set(xlabel='Function evaluations (FE)', ylabel='Median best-ever objective',
                   title=f'{fun.capitalize()} | matched {budget} FE | n={len(seeds)}')
            if fun!='rosenbrock':ax.set_yscale('symlog',linthresh=1e-4)
            ax.grid(alpha=.25)
            ax.legend();fig.tight_layout()
            fig.savefig(out/f'{fun}_fe.svg');plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,5))
        positions=np.arange(len(FUNCTIONS))
        width=.18
        for j,arm in enumerate(('pso','ipso','saipso','saipso-no-local-sa')):
            means=[next(z['mean'] for z in summary if z['function']==fun and z['algorithm']==arm) for fun in FUNCTIONS]
            ax.bar(positions+(j-1.5)*width,means,width,label=arm)
        ax.set_xticks(positions,FUNCTIONS)
        ax.set(yscale='symlog',ylabel='Mean final best objective (lower better)',title='Equal-FE comparisons; scales differ by function')
        ax.grid(axis='y',alpha=.25);ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(out/'means.svg');plt.close(fig)
    return obj


def main():
    p=argparse.ArgumentParser(description='Verify paper SA-IPSO algorithm on numerical functions')
    p.add_argument('--budget',type=int,default=2500)
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(700,712)))
    p.add_argument('--output',type=Path,default=Path('results/paper_core'))
    p.add_argument('--no-plots',action='store_true')
    p.add_argument('--skip-paper-iterations',action='store_true')
    x=p.parse_args()
    d=run(x.output,x.seeds,x.budget,not x.no_plots,not x.skip_paper_iterations)
    print('Completed',d['total_runs'],'numerical optimization runs')

if __name__=='__main__':main()
