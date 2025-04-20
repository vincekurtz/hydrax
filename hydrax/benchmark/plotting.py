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


def plot_results(results_df: pd.DataFrame) -> None:
    """Create plots from the benchmark results.

    Args:
        results_df: DataFrame with benchmark results.
    """
    # Set plot style
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("talk")

    # 1. Runtime comparison
    plt.figure(figsize=(14, 10))
    runtime_pivot = results_df.pivot(
        index="task", columns="controller", values="avg_plan_time"
    )
    sns.heatmap(
        runtime_pivot,
        annot=True,
        fmt=".5f",
        cmap="viridis",
        cbar_kws={"label": "Average step time (s)"},
    )
    plt.title("Average step time (seconds) per controller per task")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "avg_plan_time_heatmap.png", dpi=300)
    plt.close()

    # 2. Total runtime comparison
    plt.figure(figsize=(14, 10))
    total_runtime_pivot = results_df.pivot(
        index="task", columns="controller", values="total_time"
    )
    sns.heatmap(
        total_runtime_pivot,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        cbar_kws={"label": "Total time (s)"},
    )
    plt.title("Total runtime (seconds) per controller per task")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "total_time_heatmap.png", dpi=300)
    plt.close()

    # 3. Average cost comparison
    plt.figure(figsize=(14, 10))
    avg_cost_pivot = results_df.pivot(
        index="task", columns="controller", values="avg_cost"
    )
    # Use log scale for better visualization if costs vary widely
    sns.heatmap(
        np.log1p(avg_cost_pivot),
        annot=avg_cost_pivot,
        fmt=".2f",
        cmap="rocket_r",
        cbar_kws={"label": "Log(1+Average cost)"},
    )
    plt.title("Average cost per controller per task")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "avg_cost_heatmap.png", dpi=300)
    plt.close()

    # 4. Final cost comparison
    plt.figure(figsize=(14, 10))
    final_cost_pivot = results_df.pivot(
        index="task", columns="controller", values="final_cost"
    )
    sns.heatmap(
        np.log1p(final_cost_pivot),
        annot=final_cost_pivot,
        fmt=".2f",
        cmap="rocket_r",
        cbar_kws={"label": "Log(1+Final cost)"},
    )
    plt.title("Final cost per controller per task")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "final_cost_heatmap.png", dpi=300)
    plt.close()

    # 5. Performance vs Runtime scatter plot
    plt.figure(figsize=(14, 10))
    for task in results_df["task"].unique():
        task_data = results_df[results_df["task"] == task]
        plt.scatter(
            task_data["avg_plan_time"],
            task_data["avg_cost"],
            label=task,
            s=100,
            alpha=0.7,
        )

    for _, row in results_df.iterrows():
        plt.annotate(
            row["controller"],
            (row["avg_plan_time"], row["avg_cost"]),
            fontsize=8,
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Average step time (log scale)")
    plt.ylabel("Average cost (log scale)")
    plt.title("Performance vs Runtime Trade-off")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "performance_vs_runtime.png", dpi=300)
    plt.close()

    # 6. Bar plots for each task
    for task in results_df["task"].unique():
        task_data = results_df[results_df["task"] == task].sort_values(
            "avg_cost"
        )

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
        ax1.set_title(f"{task} - Average Cost per Controller")
        ax1.set_ylabel("Average Cost")
        ax1.set_yscale("log")

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
        ax2.set_title(f"{task} - Average Step Time per Controller")
        ax2.set_ylabel("Average Step Time (s)")
        ax2.set_xlabel("Controller")
        ax2.set_yscale("log")

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"{task}_comparison.png", dpi=300)
        plt.close()


def plot_cost_over_time(results_list: List[Dict[str, Any]]) -> None:
    """Create plots showing cost over time for each task.

    Args:
        results_list: List of result dictionaries with cost trajectories.
    """
    # Group results by task
    task_results = {}
    for result in results_list:
        task_name = result["task"]
        if task_name not in task_results:
            task_results[task_name] = []
        task_results[task_name].append(result)

    # Create a plot for each task
    for task_name, results in task_results.items():
        plt.figure(figsize=(14, 8))

        # Plot cost trajectories for each controller
        for result in results:
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
        plt.yscale("log")  # Log scale for costs that vary widely
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"{task_name}_cost_over_time.png", dpi=300)
        plt.close()
