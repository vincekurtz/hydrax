#!/usr/bin/env python3
"""
Benchmark script that tests all Hydrax controllers against all tasks,
measuring performance and runtime metrics.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import jax
import jax.numpy as jnp
import evosax
from mujoco import mjx
from typing import Dict, List, Tuple, Any, Type
import importlib
import inspect
import os
import mujoco
import argparse

# Import all controllers and tasks
import hydrax.algs as algs
import hydrax.tasks as tasks
from hydrax.alg_base import SamplingBasedController
from hydrax.task_base import Task
from hydrax.algs.cem import CEM
from hydrax.algs.mppi import MPPI
from hydrax.algs.predictive_sampling import PredictiveSampling
from hydrax.algs.evosax import Evosax
from hydrax.risk import WorstCase, AverageCost
from hydrax import ROOT

# Set up output directory
RESULTS_DIR = Path(ROOT) / "benchmark_results"
RESULTS_DIR.mkdir(exist_ok=True)

# Number of episodes and steps to run for each benchmark
NUM_EPISODES = 1
TOTAL_STEPS = 500


def get_all_tasks() -> Dict[str, Type[Task]]:
    """Dynamically find all task classes."""
    task_dict = {}

    # Iterate through all modules in the tasks package
    for file in os.listdir(os.path.dirname(tasks.__file__)):
        if file.endswith(".py") and not file.startswith("__"):
            module_name = file[:-3]
            module = importlib.import_module(f"hydrax.tasks.{module_name}")

            # Find classes in the module that are Tasks
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, Task)
                    and obj != Task
                    and name != "Task"
                ):
                    task_dict[name] = obj

    return task_dict


def get_default_controller_configs() -> Dict[str, Dict[str, Any]]:
    """Get default configurations for all controllers."""
    return {
        "PredictiveSampling": {
            "class": PredictiveSampling,
            "params": {
                "num_samples": 512,
                "noise_level": 0.1,
                "num_randomizations": 10,
                "plan_horizon": 0.5,
                "spline_type": "zero",
                "num_knots": 11,
            },
        },
        "MPPI": {
            "class": MPPI,
            "params": {
                "num_samples": 512,
                "noise_level": 0.1,
                "temperature": 0.01,
                "plan_horizon": 0.25,
                "spline_type": "zero",
                "num_knots": 11,
            },
        },
        "CEM": {
            "class": CEM,
            "params": {
                "num_samples": 512,
                "num_elites": 20,
                "sigma_start": 0.3,
                "sigma_min": 0.05,
                "explore_fraction": 0.5,
                "plan_horizon": 0.25,
                "spline_type": "zero",
                "num_knots": 11,
            },
        },
        # "CMA_ES": {
        #     "class": Evosax,
        #     "params": {
        #         "optimizer": evosax.CMA_ES,
        #         "num_samples": 16,
        #         "plan_horizon": 0.25,
        #         "spline_type": "zero",
        #         "num_knots": 11,
        #     },
        # },
        # "Sep_CMA_ES": {
        #     "class": Evosax,
        #     "params": {
        #         "optimizer": evosax.Sep_CMA_ES,
        #         "num_samples": 16,
        #         "plan_horizon": 0.25,
        #         "spline_type": "zero",
        #         "num_knots": 11,
        #     },
        # },
        # "SAMR_GA": {
        #     "class": Evosax,
        #     "params": {
        #         "optimizer": evosax.SAMR_GA,
        #         "num_samples": 16,
        #         "plan_horizon": 0.25,
        #         "spline_type": "zero",
        #         "num_knots": 11,
        #     },
        # },
        # "DE": {
        #     "class": Evosax,
        #     "params": {
        #         "optimizer": evosax.DE,
        #         "num_samples": 16,
        #         "plan_horizon": 0.25,
        #         "spline_type": "zero",
        #         "num_knots": 11,
        #     },
        # },
        # "GLD": {
        #     "class": Evosax,
        #     "params": {
        #         "optimizer": evosax.GLD,
        #         "num_samples": 16,
        #         "plan_horizon": 0.25,
        #         "spline_type": "zero",
        #         "num_knots": 11,
        #     },
        # },
    }


def run_benchmark_episode(
    controller: SamplingBasedController, task: Task, TOTAL_STEPS: int
) -> Dict[str, Any]:
    """Run a single benchmark episode with the given controller and task."""

    # Initialize results dict
    results = {
        "plan_times": [],
        "costs": [],
        "final_cost": 0.0,
    }

    # Create MuJoCo model and data
    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)

    # Set up simulation parameters
    frequency = 50  # Control frequency in Hz
    replan_period = 1.0 / frequency
    sim_steps_per_replan = max(int(replan_period / mj_model.opt.timestep), 1)

    # Initialize the controller and JIT functions
    mjx_data = mjx.put_data(mj_model, mj_data)
    policy_params = controller.init_params()
    jit_optimize = jax.jit(controller.optimize)
    jit_interp_func = jax.jit(controller.interp_func)

    # Run the episode
    for _ in range(TOTAL_STEPS):
        # Set the start state for the controller
        mjx_data = mjx_data.replace(
            qpos=jnp.array(mj_data.qpos),
            qvel=jnp.array(mj_data.qvel),
            mocap_pos=jnp.array(mj_data.mocap_pos),
            mocap_quat=jnp.array(mj_data.mocap_quat),
            time=mj_data.time,
        )

        plan_start_time = time.time()
        # Do a replanning step
        policy_params, rollouts = jit_optimize(mjx_data, policy_params)
        # Record time for this control step
        plan_end_time = time.time()
        results["plan_times"].append(plan_end_time - plan_start_time)

        # Interpolate control signals at simulation frequency
        sim_dt = mj_model.opt.timestep
        t_curr = mj_data.time
        tq = jnp.arange(0, sim_steps_per_replan) * sim_dt + t_curr
        tk = policy_params.tk
        knots = policy_params.mean[None, ...]
        us = np.asarray(jit_interp_func(tq, tk, knots))[0]  # (ss, nu)
        # Track the cost of this step (running cost)
        results["costs"].append(task.running_cost(mjx_data, us[0]))

        # Simulate the system for the replan period
        for i in range(sim_steps_per_replan):
            mj_data.ctrl[:] = np.array(us[i])
            mujoco.mj_step(mj_model, mj_data)

    # Convert final MuJoCo state to MJX
    final_mjx_data = mjx.put_data(mj_model, mj_data)
    # Record final cost
    results["final_cost"] = float(task.terminal_cost(final_mjx_data))

    return results


def benchmark_controller_on_task(
    controller_name: str,
    controller_config: Dict[str, Any],
    task_name: str,
    task_class: Type[Task],
    num_episodes: int,
    total_steps: int,
) -> Dict[str, Any]:
    """Benchmark a controller on a task for multiple episodes."""
    print(f"Benchmarking {controller_name} on {task_name}...")

    # Results for this controller-task pair
    results = {
        "controller": controller_name,
        "task": task_name,
        "total_time": [],
        "avg_plan_time": [],
        "avg_cost": [],
        "final_cost": [],
    }

    # Run multiple episodes
    for episode in range(num_episodes):
        print(f"  Episode {episode + 1}/{num_episodes}")

        # Create task instance
        task = task_class()

        # Create controller
        controller_class = controller_config["class"]
        controller = controller_class(task, **controller_config["params"])

        # Run the episode
        try:
            episode_start = time.time()
            episode_results = run_benchmark_episode(
                controller, task, total_steps
            )
            episode_end = time.time()

            # Record results
            results["total_time"].append(episode_end - episode_start)
            results["avg_plan_time"].append(
                np.mean(episode_results["plan_times"])
            )
            results["avg_cost"].append(np.mean(episode_results["costs"]))
            results["final_cost"].append(episode_results["final_cost"])

        except Exception as e:
            print(f"  Error in episode {episode + 1}: {e}")
            # If an episode fails, record poor performance
            results["total_time"].append(float("inf"))
            results["avg_plan_time"].append(float("inf"))
            results["avg_cost"].append(float("inf"))
            results["final_cost"].append(float("inf"))

    # Calculate averages across episodes
    for key in ["total_time", "avg_plan_time", "avg_cost", "final_cost"]:
        results[key] = np.mean(results[key])

    return results


def run_full_benchmark() -> pd.DataFrame:
    """Run the full benchmark of all controllers on all tasks."""
    # Get all tasks
    all_tasks = get_all_tasks()
    print(f"Found {len(all_tasks)} tasks: {', '.join(all_tasks.keys())}")

    # Get all controller configurations
    controller_configs = get_default_controller_configs()
    print(
        f"Found {len(controller_configs)} controllers: {', '.join(controller_configs.keys())}"
    )

    # Store results
    all_results = []

    # Benchmark each controller on each task
    for controller_name, config in controller_configs.items():
        for task_name, task_class in all_tasks.items():
            try:
                result = benchmark_controller_on_task(
                    controller_name,
                    config,
                    task_name,
                    task_class,
                    NUM_EPISODES,
                    TOTAL_STEPS,
                )
                all_results.append(result)
            except Exception as e:
                print(
                    f"Error benchmarking {controller_name} on {task_name}: {e}"
                )
                # Add failure result
                all_results.append(
                    {
                        "controller": controller_name,
                        "task": task_name,
                        "total_time": float("inf"),
                        "avg_plan_time": float("inf"),
                        "avg_cost": float("inf"),
                        "final_cost": float("inf"),
                    }
                )

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    # Save results to CSV
    results_df.to_csv(RESULTS_DIR / "benchmark_results.csv", index=False)

    return results_df


def plot_results(results_df: pd.DataFrame) -> None:
    """Create plots from the benchmark results."""
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

    # 6. Performance vs Runtime scatter plot
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

    # 7. Bar plots for each task
    for task in results_df["task"].unique():
        task_data = results_df[results_df["task"] == task].sort_values(
            "avg_cost"
        )

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Plot avg cost - fixed for deprecation warning
        sns.barplot(
            x="controller",
            y="avg_cost",
            hue="controller",  # Add hue parameter
            data=task_data,
            ax=ax1,
            palette="viridis",
            legend=False,  # Add legend=False
        )
        ax1.set_title(f"{task} - Average Cost per Controller")
        ax1.set_ylabel("Average Cost")
        ax1.set_yscale("log")

        # Plot runtime - fixed for deprecation warning
        sns.barplot(
            x="controller",
            y="avg_plan_time",
            hue="controller",  # Add hue parameter
            data=task_data,
            ax=ax2,
            palette="rocket",
            legend=False,  # Add legend=False
        )
        ax2.set_title(f"{task} - Average Step Time per Controller")
        ax2.set_ylabel("Average Step Time (s)")
        ax2.set_xlabel("Controller")
        ax2.set_yscale("log")

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"{task}_comparison.png", dpi=300)
        plt.close()


def run_single_task_benchmark(task_name: str) -> pd.DataFrame:
    """Run the benchmark for a single task with all controllers."""
    # Get all tasks
    all_tasks = get_all_tasks()
    if task_name not in all_tasks:
        available_tasks = ", ".join(all_tasks.keys())
        raise ValueError(
            f"Task '{task_name}' not found. Available tasks: {available_tasks}"
        )

    print(f"Benchmarking task: {task_name}")
    task_class = all_tasks[task_name]

    # Get all controller configurations
    controller_configs = get_default_controller_configs()
    print(
        f"Found {len(controller_configs)} controllers: {', '.join(controller_configs.keys())}"
    )

    # Store results
    all_results = []

    # Benchmark each controller on the selected task
    for controller_name, config in controller_configs.items():
        try:
            result = benchmark_controller_on_task(
                controller_name,
                config,
                task_name,
                task_class,
                NUM_EPISODES,
                TOTAL_STEPS,
            )
            all_results.append(result)
        except Exception as e:
            print(f"Error benchmarking {controller_name} on {task_name}: {e}")
            # Add failure result
            all_results.append(
                {
                    "controller": controller_name,
                    "task": task_name,
                    "total_time": float("inf"),
                    "avg_plan_time": float("inf"),
                    "avg_cost": float("inf"),
                    "final_cost": float("inf"),
                }
            )

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    # Save results to CSV
    results_df.to_csv(
        RESULTS_DIR / f"benchmark_results_{task_name}.csv", index=False
    )

    return results_df


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Benchmark controllers on tasks"
    )
    parser.add_argument(
        "--task", type=str, help="Run benchmark for a specific task only"
    )
    args = parser.parse_args()

    # Run benchmark for a single task or all tasks
    if args.task:
        results = run_single_task_benchmark(args.task)
        print(f"Single task benchmark for {args.task} complete!")
    else:
        # Run the full benchmark
        results = run_full_benchmark()

    # Plot the results
    plot_results(results)

    print(f"Benchmark complete! Results saved to {RESULTS_DIR}")
