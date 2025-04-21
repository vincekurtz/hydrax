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


def results_to_dataframe(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert results list to DataFrame for plotting.

    Args:
        results_list: List of result dictionaries

    Returns:
        DataFrame with extracted summary data
    """
    # Extract all fields except trajectories
    results_for_df = []
    for result in results_list:
        # Make sure result is a dictionary before trying to access items
        if isinstance(result, dict):
            filtered_result = {
                k: v
                for k, v in result.items()
                if k not in ["cost_trajectories", "time_trajectories"]
            }
            results_for_df.append(filtered_result)

    return pd.DataFrame(results_for_df)


def plot_results(results_list: List[Dict[str, Any]], task_name: str) -> None:
    """Create plots from the benchmark results for a specific task.

    Args:
        results_list: List of result dictionaries.
        task_name: Name of the task that was benchmarked.
    """
    # Convert to DataFrame for plotting
    results_df = results_to_dataframe(results_list)

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

    plt.figure(figsize=(14, 8))

    # Plot cost trajectories for each controller
    for result in results_list:
        controller_name = result["controller"]

        # Skip if no valid trajectories
        if not result["cost_trajectories"] or all(
            len(traj) == 0 for traj in result["cost_trajectories"]
        ):
            continue

        # Average costs across episodes
        cost_trajectories = result["cost_trajectories"]
        time_trajectories = result["time_trajectories"]

        # Find the shortest trajectory length to align data
        min_length = min(
            len(traj) for traj in cost_trajectories if len(traj) > 0
        )

        # Truncate all trajectories to the same length for averaging
        aligned_costs = [
            traj[:min_length] for traj in cost_trajectories if len(traj) > 0
        ]
        aligned_times = [
            traj[:min_length] for traj in time_trajectories if len(traj) > 0
        ]

        # Average across episodes
        avg_costs = np.mean(aligned_costs, axis=0)
        avg_times = np.mean(aligned_times, axis=0)

        # Plot the average cost trajectory
        plt.plot(avg_times, avg_costs, label=controller_name, linewidth=2)

    plt.title(f"{task_name} - Cost over Time")
    plt.xlabel("Simulation Time (s)")
    plt.ylabel("Cost")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{task_name}_cost_over_time.png", dpi=300)
    plt.close()
