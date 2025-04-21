"""Visualization utilities for benchmark results."""

from typing import Dict, List, Any
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from hydrax import ROOT

# Set up output directory
RESULTS_DIR = Path(ROOT) / "benchmark" / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def create_summary_dataframe(
    results_list: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Create a summary DataFrame from the results list.

    Args:
        results_list: List of result dictionaries

    Returns:
        DataFrame with summary data
    """
    summary_data = []
    for result in results_list:
        if "controller" in result and "avg_cost" in result:
            summary_data.append(
                {
                    "controller": result["controller"],
                    "task": result["task"],
                    "avg_cost": result["avg_cost"],
                    "avg_plan_time": result["avg_plan_time"],
                    "total_time": result["total_time"],
                }
            )

    return pd.DataFrame(summary_data)


def plot_results(results_list: List[Dict[str, Any]], task_name: str) -> None:
    """Create plots from the benchmark results for a specific task.

    Args:
        results_list: List of result dictionaries.
        task_name: Name of the task that was benchmarked.
    """
    # Create summary DataFrame for bar plots
    results_df = create_summary_dataframe(results_list)

    # Set plot style
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("talk")

    # Sort controllers by performance (avg_cost)
    task_data = results_df.sort_values("avg_cost")

    # Performance comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot avg cost
    sns.barplot(
        x="controller",
        y="avg_cost",
        hue="controller",
        data=task_data,
        ax=ax1,
        palette="viridis",
        legend=False,
    )
    ax1.set_title(f"{task_name} - Average Cost per Controller")
    ax1.set_ylabel("Average Cost")

    # Plot runtime
    sns.barplot(
        x="controller",
        y="avg_plan_time",
        hue="controller",
        data=task_data,
        ax=ax2,
        palette="rocket",
        legend=False,
    )
    ax2.set_title(f"{task_name} - Average Plan Time per Controller")
    ax2.set_ylabel("Average Plan Time (s)")
    ax2.set_xlabel("Controller")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{task_name}_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 8))

    # Create scatter plot
    plt.scatter(
        results_df["avg_plan_time"], results_df["avg_cost"], s=100, alpha=0.7
    )

    # Add controller name labels to each point
    for _, row in results_df.iterrows():
        plt.annotate(
            row["controller"],
            (row["avg_plan_time"], row["avg_cost"]),
            fontsize=10,
            xytext=(5, 5),
            textcoords="offset points",
        )

    # Add labels and title
    plt.xlabel("Average Planning Time (s)")
    plt.ylabel("Average Cost")
    plt.title(f"{task_name} - Cost vs Performance Trade-off")

    # Add grid and improve layout
    plt.grid(True, which="both", ls="--", alpha=0.3)
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
    colors = plt.cm.viridis(np.linspace(0, 1, num_controllers))

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
