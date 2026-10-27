"""Vector data figures for independent v0.3 study (no trajectory smoothing)."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .feasible import ARMS


def plots(output: Path, rows: list[dict], analyses: list[dict], hist: dict, hist_v: dict, scenes: dict, budget: int):
    group_order = [(s,t) for s in scenes for t in ('fast','safe')]
    labels=[f'{s}/{t}' for s,t in group_order]
    fig,ax=plt.subplots(figsize=(12,5.8))
    xx=np.arange(len(group_order));width=.13
    for i,arm in enumerate(ARMS):
        rates=[next(q['audited_feasible_rate'] for q in analyses if q['scene']==s and q['task']==t and q['arm']==arm) for s,t in group_order]
        ax.bar(xx+(i-2.5)*width,rates,width,label=arm)
    ax.set_xticks(xx);ax.set_xticklabels(labels,rotation=15,ha='right')
    ax.set_ylim(0,1.05);ax.set_ylabel('Dense-audit feasible fraction (higher is better)')
    ax.set_title('Constrained feasibility ablation | 201-sample terrain + exact airspace intersection')
    ax.legend(ncol=3,fontsize=8);ax.grid(axis='y',alpha=.18)
    fig.tight_layout();fig.savefig(output/'feasibility_rate.svg');plt.close(fig)
    # Traces are objective evaluations, not iterations; all methods share x-axis.
    for sc,t in group_order:
        fig,ax=plt.subplots(figsize=(10,5.6))
        for arm in ARMS:
            vals=np.stack(hist_v[(sc,t,arm)])
            mean=vals.mean(axis=0)
            ax.plot(np.arange(1,budget+1),mean,label=arm,lw=1.6)
        ax.set(xlabel='Objective+constraint evaluations (equal FE budget)',ylabel='Mean best normalized violation',
               title=f'Feasibility convergence | {sc}/{t} (lower is better)')
        ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=2);fig.tight_layout()
        fig.savefig(output/f'violation_{sc}_{t}.svg');plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,5.6))
        for arm in ARMS:
            vals=np.stack(hist[(sc,t,arm)])
            with np.errstate(all='ignore'):
                frac=np.isfinite(vals).mean(axis=0)
            ax.plot(np.arange(1,budget+1),frac,label=arm,lw=1.6)
        ax.set(xlabel='Objective+constraint evaluations',ylabel='Fraction with feasible incumbent',ylim=(0,1.05),
               title=f'First-feasible progress (61-sample search oracle) | {sc}/{t}')
        ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=2);fig.tight_layout()
        fig.savefig(output/f'first_feasible_{sc}_{t}.svg');plt.close(fig)
