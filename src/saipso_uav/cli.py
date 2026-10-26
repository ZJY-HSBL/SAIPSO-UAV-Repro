"""Command-line runner for paired-seed civilian benchmark studies."""
import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
from .model import Scenario, WEIGHTS, fitness_details
from .optimizer import OptimizerConfig, optimize
from .visualize import plot_route, plot_convergence


def run(output: Path, iterations: int, seeds: list[int], plot: bool = True) -> list[dict]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Supply unique seeds")
    output.mkdir(parents=True, exist_ok=True)
    scenario = Scenario()
    config = replace(OptimizerConfig(), iterations=iterations)
    rows = []
    for mode, weights in WEIGHTS.items():
        first_seed_curves = {}
        for seed in seeds:
            for algorithm in ("pso", "ipso", "saipso"):
                result = optimize(scenario, weights, config, seed, algorithm)
                details = fitness_details(result.best_vector, scenario, weights)
                row = {"mode": mode, "seed": seed, "algorithm": algorithm, "evaluations": result.evaluations,
                       "sa_calls": result.sa_calls, **details}
                rows.append(row)
                if seed == seeds[0]:
                    first_seed_curves[algorithm.upper()] = result.history
                    if plot and algorithm == "saipso":
                        plot_route(result.best_vector, scenario, output / f"route_{mode}.svg", f"Civilian 3D route: {mode}")
                np.savetxt(output / f"curve_{mode}_{algorithm}_seed{seed}.csv", result.history,
                           delimiter=",", header="best_so_far_fitness", comments="")
        if plot:
            plot_convergence(first_seed_curves, output / f"convergence_{mode}.svg",
                             f"{mode.title()} objective — seed {seeds[0]} (not paper's numeric result)")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {}
    for mode in WEIGHTS:
        summary[mode] = {}
        for algorithm in ("pso", "ipso", "saipso"):
            values = [r["fitness"] for r in rows if r["mode"] == mode and r["algorithm"] == algorithm]
            summary[mode][algorithm] = {"n": len(values), "mean_fitness": float(np.mean(values)),
                                         "std_fitness_ddof1": float(np.std(values, ddof=1)) if len(values) > 1 else None}
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump({"provenance": "independent civilian scenario; not exact paper reproduction",
                   "config": asdict(config), "scenario": asdict(scenario),
                   "seeds": seeds, "summary": summary}, stream, indent=2, ensure_ascii=False)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Paper-informed SA-IPSO civilian benchmark")
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    rows = run(args.output, args.iterations, args.seeds, plot=not args.no_plots)
    for r in rows:
        print(f"{r['mode']:>4s} seed={r['seed']:>3} {r['algorithm']:>6} fitness={r['fitness']:.5f} "
              f"length={r['length']:.1f} evals={r['evaluations']}")
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
