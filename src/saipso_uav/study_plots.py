"""Matplotlib study figures, saved as vector SVG without data smoothing."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .equal_budget import ALGORITHMS


def plots(output: Path, rows: list[dict], traces: dict, scenarios: dict, budget: int):
    for scenario in scenarios:
        for task in ("fast", "safe"):
            fig, ax = plt.subplots(figsize=(9, 5.3))
            for alg in ALGORITHMS:
                values = np.vstack(traces[(scenario, task, alg)])
                mean = values.mean(axis=0)
                se = values.std(axis=0, ddof=1) / np.sqrt(len(values)) if len(values)>1 else np.zeros(budget)
                x = np.arange(1, budget+1)
                ax.plot(x, mean, label=alg, linewidth=1.8)
                ax.fill_between(x, mean-1.96*se, mean+1.96*se, alpha=0.13)
            ax.set(xlabel="Function evaluations (equal budget)", ylabel="Best-ever fitness (lower is better)",
                   title=f"Civilian {scenario} / {task} (mean ± 1.96 SE; exploratory)")
            ax.grid(alpha=.2)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output/f"fe_convergence_{scenario}_{task}.svg",format="svg")
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(9, 5.3))
            d = [[r["fitness"] for r in rows if r["scenario"]==scenario and r["task"]==task and r["algorithm"]==alg] for alg in ALGORITHMS]
            ax.boxplot(d, showmeans=True)
            ax.set_xticks(np.arange(1, len(ALGORITHMS)+1))
            ax.set_xticklabels(list(ALGORITHMS))
            ax.set(ylabel="Final best-ever fitness", title=f"{scenario} / {task}: paired seeds, n={len(d[0])}")
            ax.grid(axis="y",alpha=.2)
            fig.tight_layout()
            fig.savefig(output/f"fe_distribution_{scenario}_{task}.svg",format="svg")
            plt.close(fig)
    # A quick feasibility summary. A nonzero mean violation is never called safe.
    fig, ax = plt.subplots(figsize=(10,5.3))
    labels = []
    matrix = []
    for scenario in scenarios:
        for task in ("fast", "safe"):
            labels.append(scenario+"/"+task)
            matrix.append([np.mean([r["sampled_feasible"] for r in rows if r["scenario"]==scenario and r["task"]==task and r["algorithm"]==alg]) for alg in ALGORITHMS])
    x=np.arange(len(labels));width=.19
    for j, alg in enumerate(ALGORITHMS):
        ax.bar(x+(j-1.5)*width,np.array(matrix)[:,j],width,label=alg)
    ax.set(xticks=x,xticklabels=labels,ylabel="Fraction passing sampled feasibility checks",ylim=(0,1.05),
           title="Dense sampled validation: not a flight safety certificate")
    ax.grid(axis="y",alpha=.2);ax.legend();fig.tight_layout()
    fig.savefig(output/"feasibility_by_task.svg",format="svg")
    plt.close(fig)
