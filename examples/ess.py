"""Test Effective Sample Size (ESS) statistics as a performance predictor."""

from typing import Tuple

import jax.numpy as jnp
import numpy as np

from hydrax.algs.predictive_sampling import PredictiveSampling
from hydrax.benchmarking import benchmark
from hydrax.tasks.pendulum import Pendulum


def run_pendulum_benchmark(
    total_time: float = 5.0,
    num_samples: int = 32,
    save_path: str = "data/pendulum_benchmark.npz",
) -> None:
    """Set up a pendulum swing-up controller and benchmark it.

    Records the closed-loop running cost and the per-step rollout total
    costs, then saves both arrays to `save_path` for later ESS analysis.
    """
    task = Pendulum()
    ctrl = PredictiveSampling(
        task,
        num_samples=num_samples,
        noise_level=0.1,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=11,
    )

    # Start hanging straight down — the standard swing-up initial condition.
    initial_state = task.make_data()
    initial_state = initial_state.replace(
        qpos=jnp.array([0.0]), qvel=jnp.array([0.0])
    )

    running_costs, rollout_costs = benchmark(
        task,
        ctrl,
        initial_state,
        total_time=total_time,
        save_path=save_path,
    )

    print(f"Saved benchmark results to {save_path}")
    print(
        f"  running_costs: {running_costs.shape}, "
        f"sum = {float(jnp.sum(running_costs)):.3f}"
    )
    print(f"  rollout_costs: {rollout_costs.shape}")


def compute_ess(
    save_path: str, temperature: float
) -> Tuple[float, float, float]:
    """Load benchmark results from `save_path` and compute ESS statistics.

    Args:
        save_path: Path to the .npz file containing benchmark results.
        temperature: The temperature parameter for weighting of the
                     rollout costs when computing ESS.

    Returns:
        Total running cost
        Effective sample size (ESS) relative to the optimal policy
        Dataset size (num_steps * num_samples)
    """
    data = np.load(save_path)
    running_costs = data["running_costs"]
    rollout_costs = data["rollout_costs"]

    # Compute the total running cost, as an overall performance metric.
    total_running_cost = np.sum(running_costs)

    # Flatten the rollout costs to shape (num_steps * num_samples,).
    rollout_costs_flat = rollout_costs.flatten()

    # Shift costs so that the minimum cost is zero to improve numerics
    min_cost = np.min(rollout_costs_flat)
    rollout_costs_flat -= min_cost

    # Compute the ESS
    weights = np.exp(-rollout_costs_flat / temperature)
    ess = (np.sum(weights) ** 2) / np.sum(weights**2)

    # Compute the dataset size
    dataset_size = rollout_costs.size

    return total_running_cost, ess, dataset_size


if __name__ == "__main__":
    print("Running pendulum benchmark...")
    run_pendulum_benchmark(num_samples=32)

    total_cost, ess, dataset_size = compute_ess(
        "data/pendulum_benchmark.npz", temperature=1.0
    )
    print(f"Total running cost: {total_cost:.3f}")
    print(f"Effective Sample Size (ESS): {ess:.1f}")
    print(f"Dataset size (num_steps * num_samples): {dataset_size}")
    print(f"ESS / Dataset size: {ess / dataset_size:.4f}")
