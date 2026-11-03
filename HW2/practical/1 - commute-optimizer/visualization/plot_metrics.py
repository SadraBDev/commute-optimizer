"""
visualization/plot_metrics.py
Plots optimization cost convergence profiles over search steps or generations.
"""

from typing import Dict, List, Optional
import matplotlib.pyplot as plt


def plot_convergence(
    histories: Dict[str, List[float]],
    title: str = "Search Convergence Comparison",
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plots objective cost f(x) over iterations for one or multiple solvers.

    Args:
        histories: Dictionary mapping algorithm name -> list of recorded objective scores.
        title: Figure title.
        save_path: Optional output path to save PNG/PDF figure.
        show: Whether to display interactive plot window.
    """
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    palette = {
        "Hill Climbing": "#E67E22",
        "Simulated Annealing": "#9B59B6",
        "Local Beam Search": "#2980B9",
        "Genetic Algorithm": "#27AE60"
    }

    for algo_name, cost_history in histories.items():
        if not cost_history:
            continue
        color = palette.get(algo_name, None)
        iterations = list(range(1, len(cost_history) + 1))
        plt.plot(
            iterations,
            cost_history,
            label=algo_name,
            linewidth=2.2,
            alpha=0.9,
            color=color
        )

    plt.xlabel("Iteration / Generation Step", fontsize=11, fontweight="bold")
    plt.ylabel("Commute Duration $f(x)$ (minutes)", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
