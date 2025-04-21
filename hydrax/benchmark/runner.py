"""Benchmarking runner for controllers on a specific task."""

import time
from typing import Dict, Any, List, Tuple, Type

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
import pandas as pd
from pathlib import Path
import os
import inspect
import importlib

import hydrax.tasks as tasks
from hydrax.task_base import Task
from hydrax.alg_base import SamplingBasedController
from hydrax.task_base import Task
from hydrax import ROOT

from controllers import get_default_controller_configs

# Set up output directory
RESULTS_DIR = Path(ROOT) / "benchmark" / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_benchmark_episode(
    controller: SamplingBasedController, task: Task, total_steps: int
) -> Dict[str, Any]:
    """Run a single benchmark episode with the given controller and task.

    Args:
        controller: The controller to benchmark.
        task: The task to run.
        total_steps: Number of control steps to run.

    Returns:
        Dictionary with benchmark results.
    """
    # Initialize results dict
    results = {
        "plan_times": [],
        "costs": [],
        "final_cost": 0.0,
        "timestamps": [],
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
    for _ in range(total_steps):
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

        # Record current timestamp
        results["timestamps"].append(mj_data.time)

        # Interpolate control signals at simulation frequency
        sim_dt = mj_model.opt.timestep
        t_curr = mj_data.time
        tq = jnp.arange(0, sim_steps_per_replan) * sim_dt + t_curr
        tk = policy_params.tk
        knots = policy_params.mean[None, ...]
        us = np.asarray(jit_interp_func(tq, tk, knots))[0]  # (ss, nu)

        # Track the running cost at this step
        results["costs"].append(float(task.running_cost(mjx_data, us[0])))

        # Simulate the system for the replan period
        for i in range(sim_steps_per_replan):
            mj_data.ctrl[:] = np.array(us[i])
            mujoco.mj_step(mj_model, mj_data)

    # Convert final MuJoCo state to MJX
    final_mjx_data = mjx.put_data(mj_model, mj_data)

    # Record final cost
    final_cost = task.terminal_cost(final_mjx_data)
    if hasattr(final_cost, "shape") and final_cost.shape:
        final_cost = np.mean(final_cost)
    results["final_cost"] = float(final_cost)

    return results


def benchmark_controller_on_task(
    controller_name: str,
    controller_config: Dict[str, Any],
    task_name: str,
    task_class: Task,
    num_episodes: int = 3,
    total_steps: int = 500,
) -> Dict[str, Any]:
    """Benchmark a controller on a task for multiple episodes.

    Args:
        controller_name: Name of the controller.
        controller_config: Configuration for the controller.
        task_name: Name of the task.
        task_class: Task class.
        num_episodes: Number of episodes to run.
        total_steps: Number of steps per episode.

    Returns:
        Dictionary with benchmark results.
    """
    print(f"Benchmarking {controller_name} on {task_name}...")

    # Results for this controller-task pair
    results = {
        "controller": controller_name,
        "task": task_name,
        "total_time": [],
        "avg_plan_time": [],
        "avg_cost": [],
        "final_cost": [],
        "cost_trajectories": [],
        "time_trajectories": [],
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

            # Store trajectories
            results["cost_trajectories"].append(episode_results["costs"])
            results["time_trajectories"].append(episode_results["timestamps"])

        except Exception as e:
            print(f"  Error in episode {episode + 1}: {e}")
            # If an episode fails, record poor performance
            results["total_time"].append(float("inf"))
            results["avg_plan_time"].append(float("inf"))
            results["avg_cost"].append(float("inf"))
            results["final_cost"].append(float("inf"))
            results["cost_trajectories"].append([])
            results["time_trajectories"].append([])

    # Calculate averages across episodes
    for key in ["total_time", "avg_plan_time", "avg_cost", "final_cost"]:
        results[key] = np.mean(results[key])

    return results


def get_all_tasks() -> Dict[str, Type[Task]]:
    """Dynamically find all task classes.
    Returns:
        Dictionary mapping task names to task classes.
    """
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


def run_task_benchmark(
    task_name: str, num_episodes: int = 3, total_steps: int = 500
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Run the benchmark for a single task with all controllers.

    Args:
        task_name: Name of the task to benchmark.
        num_episodes: Number of episodes per benchmark.
        total_steps: Number of steps per episode.

    Returns:
        Tuple of (DataFrame with results, List of full result dictionaries)
    """
    # Get task class
    all_tasks = get_all_tasks()
    task_class = all_tasks[task_name]

    # Get all controller configurations
    controller_configs = get_default_controller_configs()
    print(
        f"Benchmarking {len(controller_configs)} controllers on task: {task_name}"
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
                num_episodes,
                total_steps,
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
                    "cost_trajectories": [],
                    "time_trajectories": [],
                }
            )

    # Convert to DataFrame (excluding trajectories)
    results_for_df = [
        {
            k: v
            for k, v in result.items()
            if k not in ["cost_trajectories", "time_trajectories"]
        }
        for result in all_results
    ]
    results_df = pd.DataFrame(results_for_df)

    return results_df, all_results
