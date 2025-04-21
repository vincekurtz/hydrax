"""Visualization utilities for benchmark results."""

from typing import Dict, List, Any
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from hydrax import ROOT

# Set up output directory
RESULTS_DIR = Path(ROOT) / "benchmark" / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def plot_results(results_list: List[Dict[str, Any]], task_name: str) -> None:
    """Create plots from the benchmark results for a specific task.

    Args:
        results_list: List of result dictionaries.
        task_name: Name of the task that was benchmarked.
    """
    # Extract data for plotting
    controllers = []
    avg_costs = []
    avg_plan_times = []

    # Sort results by average cost
    results_sorted = sorted(
        results_list, key=lambda x: x.get("avg_cost", float("inf"))
    )

    for result in results_sorted:
        if "controller" in result and "avg_cost" in result:
            controllers.append(result["controller"])
            avg_costs.append(result["avg_cost"])
            avg_plan_times.append(result["avg_plan_time"])

    # Performance comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Set common style
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 12})

    # Plot avg cost
    colors = cm.viridis(np.linspace(0, 1, len(controllers)))
    bars1 = ax1.bar(controllers, avg_costs, color=colors)
    ax1.set_title(f"{task_name} - Average Cost per Controller")
    ax1.set_ylabel("Average Cost")

    # Plot runtime
    bars2 = ax2.bar(
        controllers,
        avg_plan_times,
        color=cm.plasma(np.linspace(0, 1, len(controllers))),
    )
    ax2.set_title(f"{task_name} - Average Plan Time per Controller")
    ax2.set_ylabel("Average Plan Time (s)")
    ax2.set_xlabel("Controller")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{task_name}_comparison.png", dpi=300)
    plt.close()

    # Cost vs Performance scatter plot
    plt.figure(figsize=(10, 8))

    # Create scatter plot
    plt.scatter(avg_plan_times, avg_costs, s=100, alpha=0.7)

    # Add controller name labels to each point
    for i, controller in enumerate(controllers):
        plt.annotate(
            controller,
            (avg_plan_times[i], avg_costs[i]),
            fontsize=10,
            xytext=(5, 5),
            textcoords="offset points",
        )

    # Add labels and title
    plt.xlabel("Average Planning Time (s)")
    plt.ylabel("Average Cost")
    plt.title(f"{task_name} - Cost vs Performance Trade-off")

    # Add grid and improve layout
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    plt.savefig(RESULTS_DIR / f"{task_name}_cost_vs_performance.png", dpi=300)
    plt.close()


def plot_cost_over_time(
    results_list: List[Dict[str, Any]], task_name: str
) -> None:
    """Create plot showing cost over time for the task.

    Args:
        results_list: List of result dictionaries with cost trajectories.
        task_name: Name of the task that was benchmarked.
    """
    plt.figure(figsize=(14, 8))

    # Get a different color for each controller
    num_controllers = len(results_list)
    colors = cm.viridis(np.linspace(0, 1, num_controllers))

    # Plot cost trajectories for each controller
    for i, result in enumerate(results_list):
        controller_name = result["controller"]

        # Skip if no valid trajectories
        if not result["costs"] or not result["timestamps"]:
            continue

        # Plot the cost trajectory
        plt.plot(
            result["timestamps"],
            result["costs"],
            label=controller_name,
            color=colors[i],
            linewidth=1.5,
        )

    plt.title(f"{task_name} - Cost Over Time")
    plt.xlabel("Simulation Time (s)")
    plt.ylabel("Cost")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{task_name}_cost_over_time.png", dpi=300)
    plt.close()
