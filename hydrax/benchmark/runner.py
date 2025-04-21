"""Benchmarking runner for controllers on a specific task."""

import time
from typing import Dict, Any, List, Type

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
from hydrax import ROOT

from controllers import get_default_controller_configs

# Set up output directory
RESULTS_DIR = Path(ROOT) / "benchmark" / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_benchmark(
    controller_name: str,
    controller_config: Dict[str, Any],
    task_name: str,
    task_class: Type[Task],
    total_steps: int = 500,
) -> Dict[str, Any]:
    """Run a benchmark for a controller on a task.

    Args:
        controller_name: Name of the controller.
        controller_config: Configuration for the controller.
        task_name: Name of the task.
        task_class: Task class.
        total_steps: Number of steps to run.

    Returns:
        Dictionary with benchmark results.
    """
    print(f"Benchmarking {controller_name} on {task_name}...")

    # Create task instance
    task = task_class()

    # Create controller
    controller_class = controller_config["class"]
    controller = controller_class(task, **controller_config["params"])

    # Initialize results
    results = {
        "controller": controller_name,
        "task": task_name,
        "plan_times": [],
        "costs": [],
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

    # Record start time for total runtime calculation
    total_start_time = time.time()

    try:
        # Run the simulation
        for step in range(total_steps):
            # Set the start state for the controller
            mjx_data = mjx_data.replace(
                qpos=jnp.array(mj_data.qpos),
                qvel=jnp.array(mj_data.qvel),
                mocap_pos=jnp.array(mj_data.mocap_pos),
                mocap_quat=jnp.array(mj_data.mocap_quat),
                time=mj_data.time,
            )

            # Record planning time
            plan_start_time = time.time()
            policy_params, rollouts = jit_optimize(mjx_data, policy_params)
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

        # Calculate summary statistics
        results["avg_plan_time"] = np.mean(results["plan_times"])
        results["avg_cost"] = np.mean(results["costs"])
        results["total_time"] = time.time() - total_start_time

    except Exception as e:
        print(f"Error running benchmark for {controller_name}: {e}")
        # Record failure with infinite cost
        results["avg_plan_time"] = float("inf")
        results["avg_cost"] = float("inf")
        results["total_time"] = float("inf")

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
    task_name: str, total_steps: int = 500
) -> List[Dict[str, Any]]:
    """Run the benchmark for a single task with all controllers.

    Args:
        task_name: Name of the task to benchmark.
        total_steps: Number of steps to run.

    Returns:
        List of result dictionaries with benchmark data
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
            result = run_benchmark(
                controller_name,
                config,
                task_name,
                task_class,
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
                    "plan_times": [],
                    "costs": [],
                    "timestamps": [],
                    "avg_plan_time": float("inf"),
                    "avg_cost": float("inf"),
                    "total_time": float("inf"),
                }
            )

    return all_results
