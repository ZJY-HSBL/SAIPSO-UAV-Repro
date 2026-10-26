# Reproducibility / 复现说明

Reference: FAN Weilun, *UAV Route Planning Based on SA-IPSO Algorithm*, Ship Electronic Engineering 46(6), 2026. DOI: 10.3969/j.issn.1672-9730.2026.06.009.

## Paper-derived information

Section 2: personal-best simulated annealing acceptance; decreasing inertia; immune memory; affinity and fitness-difference concentration (beta=0.5); annealing when two successive iterations are stagnant. Section 4: endpoints (10,10,10) and (190,190,10), M=20, N=10, n=10, c1=c2=2.05, inertia 0.9 to 0.4, vmax=20, epsilon=1e-4, T0=2000, Tmin=10, alpha=0.8, tmax=1000. Outer AHP task weights: fast (0.6370,0.2583,0.1047); safe (0.0879,0.6694,0.2426).

## Independently chosen implementation

- Synthetic civilian Gaussian terrain and circular restricted-airspace zones, **not** the paper's original maps or military threat models.
- Ten intermediate waypoints with fixed x coordinates, optimizable y and z (20 continuous variables), instead of an unknown full 3-D encoding.
- Soft-penalty aircraft feasibility objective; collisions and constraint violations can remain in some computed paths. Output is **not** flight-ready.
- Memory: max 10 nonidentical archived candidates; 2 memory injections per iteration with 1.5% span jitter. These operators and the SA neighborhood are implementation choices, not directly specified by the article.
- SA: 2 trials per temperature level, max 24 cooling levels, 8% span initial neighborhood step. Original absolute temperatures may be poorly calibrated to this benchmark's normalized fitness.
- Initial study: 120 iterations, rather than the paper's 1000. Three methods use different fitness evaluation budgets and do not support fair fixed-evaluation comparisons.

## Protocol

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python -m saipso_uav --iterations 120 --seeds 0 1 2 3 4 --output docs/figures
```

Output: all paired-seed rows in `metrics.csv`, complete best-so-far traces in `curve_*.csv`, summary and config in `summary.json`, SVG plots in `route_*.svg` and `convergence_*.svg`. Report all results, including cases where hybrid optimization loses.

## Next validation stage

Run equal fitness-evaluation budgets, out-of-sample civilian terrains, at least 30 fresh random seeds, physical constraint feasibility checks, parameter sensitivity/ablation, uncertainty intervals, and corrected significance tests. No global optimality or algorithmic superiority is claimed based on this preliminary 5-seed study.
