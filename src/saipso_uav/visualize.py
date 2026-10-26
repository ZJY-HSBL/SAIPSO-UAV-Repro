"""Publication-ready route and convergence SVG plots (noninteractive)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .model import Scenario, terrain_height, vector_to_route


def plot_route(vector: np.ndarray, scenario: Scenario, output: Path, title: str):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    route = vector_to_route(vector, scenario)
    x = np.linspace(0, 200, 95)
    y = np.linspace(0, 200, 95)
    xx, yy = np.meshgrid(x, y)
    zz = terrain_height(xx, yy, scenario)
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, zz, cmap="viridis", alpha=0.77, linewidth=0, antialiased=True)
    ax.plot(route[:, 0], route[:, 1], route[:, 2], color="crimson", lw=2.8, label="Planned route")
    ax.scatter(route[[0,-1],0], route[[0,-1],1], route[[0,-1],2], s=40, c=["tab:blue","tab:orange"])
    ax.set(xlabel="X", ylabel="Y", zlabel="Altitude", title=title, zlim=(0, 450))
    ax.view_init(elev=29, azim=-65)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_convergence(results: dict[str, np.ndarray], output: Path, title: str):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for label, history in results.items():
        ax.plot(np.arange(len(history)), history, label=label, lw=2)
    ax.set(xlabel="Iterations", ylabel="Best-so-far fitness (lower is better)", title=title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)
