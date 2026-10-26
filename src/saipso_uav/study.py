"""Frozen equal-FE civil terrain study: reproducible evidence and SVGs."""
import argparse
import csv
import json
from pathlib import Path
from dataclasses import asdict, replace

import numpy as np
from .equal_budget import ALGORITHMS, optimize_budget
from .model import Scenario, WEIGHTS, fitness_details
from .optimizer import OptimizerConfig
from .validation import dense_validation


def civilian_scenarios() -> dict[str, Scenario]:
    base = Scenario(min_clearance=5.0)  # Endpoint clearance 10 minus nonzero terrain: baseline H=10 was inconsistent
    holdout = replace(base, name="held-out-terrain", hills=(
        (53., 105., 195., 21.), (111., 122., 275., 26.), (157., 150., 145., 19.),
    ), zones=((47., 53., 20.), (150., 64., 25.), (128., 165., 14.)))
    return {"reference": base, "holdout": holdout}


def paired_bootstrap(differences: np.ndarray, seed: int = 8264,
                     reps: int = 10000) -> tuple[float, float]:
    """Paired resampling of mean differences; exploratory 95% percentile CI."""
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(differences), size=(reps, len(differences)))
    samples = differences[indexes].mean(axis=1)
    return tuple(map(float, np.quantile(samples, [0.025, 0.975])))


def run(output: Path, budget: int = 1200, seeds: list[int] | None = None,
        plot: bool = True) -> list[dict]:
    if seeds is None:
        seeds = list(range(100, 120))
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    output.mkdir(parents=True, exist_ok=True)
    scenarios = civilian_scenarios()
    cfg = OptimizerConfig()
    rows: list[dict] = []
    traces: dict[tuple[str, str, str], list[np.ndarray]] = {}
    for scenario_name, scenario in scenarios.items():
        for mode, weights in WEIGHTS.items():
            for seed in seeds:
                for algorithm in ALGORITHMS:
                    result = optimize_budget(scenario, weights, budget, seed, algorithm, cfg)
                    details = fitness_details(result.vector, scenario, weights)
                    checked = dense_validation(result.vector, scenario)
                    rows.append({"scenario": scenario_name, "task": mode, "seed": seed,
                                 "algorithm": algorithm, "budget": budget,
                                 "evaluations": result.evaluations,
                                 "iterations": result.iterations, "sa_calls": result.sa_calls,
                                 "temperature_scale": result.temperature_scale,
                                 **details,
                                 "sampled_feasible": checked["sampled_feasible"],
                                 "min_dense_clearance": checked["min_dense_clearance"],
                                 "restricted_sample_fraction": checked["restricted_sample_fraction"],
                                 **{f"passes_{k}": v for k, v in checked["checks"].items()}})
                    traces.setdefault((scenario_name, mode, algorithm), []).append(result.trace)
                    if len(rows) % 40 == 0:
                        print(f"Completed {len(rows)} runs", flush=True)
    with (output/"metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    analyses = []
    for scenario in scenarios:
        for task in WEIGHTS:
            subset = [r for r in rows if r["scenario"] == scenario and r["task"] == task]
            for algorithm in ALGORITHMS:
                ours = [r for r in subset if r["algorithm"] == algorithm]
                values = np.array([r["fitness"] for r in ours])
                analyses.append(dict(scenario=scenario, task=task, algorithm=algorithm,
                                     n=len(ours), mean=float(values.mean()),
                                     std=float(values.std(ddof=1)) if len(values)>1 else None,
                                     feasible_count=sum(r["sampled_feasible"] for r in ours)))
            # Paired differences vs PSO on identical seeds; smaller is better.
            pso = {r["seed"]: r["fitness"] for r in subset if r["algorithm"] == "pso"}
            for algorithm in ALGORITHMS[1:]:
                diff = np.array([r["fitness"] - pso[r["seed"]] for r in subset if r["algorithm"] == algorithm])
                lo, hi = paired_bootstrap(diff)
                analyses.append(dict(scenario=scenario, task=task, comparison=f"{algorithm}-pso",
                                     mean_diff=float(diff.mean()), ci95=[lo, hi],
                                     paired_seeds=len(diff)))
    manifest = {"provenance": "Independent civilian terrain extension, NOT original-paper results",
                "budget": budget, "seeds": seeds, "config": asdict(cfg),
                "scenarios": {key: asdict(value) for key,value in scenarios.items()},
                "analyses": analyses,
                "limitations": ["Temperature calibration is an original adaptation, not from source paper",
                                "Fitness penalties are soft; no guarantee of continuous feasibility",
                                "Dense feasibility is sampled, not formal safety verification",
                                "Synthetic terrain, not original paper geometry or real flight data",
                                "Exploratory small-sample bootstrap, not confirmatory inference"]}
    (output/"summary.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),encoding="utf-8")
    if plot:
        from .study_plots import plots
        plots(output, rows, traces, scenarios, budget)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Independent civilian equal-evaluation optimization study")
    parser.add_argument("--budget", type=int, default=1200)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(100, 120)))
    parser.add_argument("--output", type=Path, default=Path("results/equal_budget"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    rows = run(args.output, args.budget, args.seeds, not args.no_plots)
    print(f"{len(rows)} runs, all at {args.budget} FEs, exported to {args.output}")


if __name__ == "__main__":
    main()
