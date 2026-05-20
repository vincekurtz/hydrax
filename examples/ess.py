"""Test Effective Sample Size (ESS) statistics as a performance predictor."""

import glob
import os
from typing import Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
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


def sweep_pendulum_hyperparams(
    num_runs: int,
    total_time: float = 5.0,
    output_dir: str = "data/pendulum_sweep",
    seed: int = 0,
    num_samples_range: Tuple[int, int] = (2, 128),
    noise_level_range: Tuple[float, float] = (0.1, 0.1),
    plan_horizon_range: Tuple[float, float] = (1.0, 1.0),
    num_knots_range: Tuple[int, int] = (8, 8),
) -> None:
    """Sweep pendulum benchmark hyperparameters with random sampling.

    For each run, samples `num_samples`, `noise_level`, `plan_horizon`, and
    `num_knots` uniformly from the supplied (inclusive) ranges using `seed`
    so the sweep is fully reproducible. Each run is written to its own file
    `output_dir/run_<i>.npz`, containing both the cost arrays produced by
    `benchmark` and the hyperparameters used. That format means
    `compute_ess` can be applied to any single run directly, and a future
    aggregator can simply glob the directory and read the hyperparameters
    out of each file.

    Args:
        num_runs: Number of randomly-sampled benchmark runs.
        total_time: Simulated time in seconds for each run.
        output_dir: Directory to write per-run .npz files into. Created if
                    missing.
        seed: Seed for both hyperparameter sampling and per-run controller
              RNG initialization.
        num_samples_range: Inclusive (low, high) for the number of rollouts.
        noise_level_range: (low, high) for the sampling noise scale.
        plan_horizon_range: (low, high) for the planning horizon in seconds.
        num_knots_range: Inclusive (low, high) for the number of spline
                         knots.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    task = Pendulum()
    initial_state = task.make_data()
    initial_state = initial_state.replace(
        qpos=jnp.array([0.0]), qvel=jnp.array([0.0])
    )

    for i in range(num_runs):
        num_samples = int(
            rng.integers(
                num_samples_range[0], num_samples_range[1], endpoint=True
            )
        )
        noise_level = float(
            rng.uniform(noise_level_range[0], noise_level_range[1])
        )
        plan_horizon = float(
            rng.uniform(plan_horizon_range[0], plan_horizon_range[1])
        )
        num_knots = int(
            rng.integers(num_knots_range[0], num_knots_range[1], endpoint=True)
        )

        ctrl = PredictiveSampling(
            task,
            num_samples=num_samples,
            noise_level=noise_level,
            plan_horizon=plan_horizon,
            spline_type="zero",
            num_knots=num_knots,
            seed=seed + i,
        )
        # Use a per-run seed for the sampling RNG so each run is
        # individually reproducible but distinct from its neighbors.
        initial_params = ctrl.init_params(seed=seed + i)

        running_costs, rollout_costs = benchmark(
            task,
            ctrl,
            initial_state,
            total_time=total_time,
            initial_params=initial_params,
        )

        path = os.path.join(output_dir, f"run_{i:04d}.npz")
        np.savez(
            path,
            running_costs=np.asarray(running_costs),
            rollout_costs=np.asarray(rollout_costs),
            num_samples=num_samples,
            noise_level=noise_level,
            plan_horizon=plan_horizon,
            num_knots=num_knots,
            total_time=total_time,
        )
        print(
            f"[{i + 1}/{num_runs}] {path} "
            f"(num_samples={num_samples}, noise_level={noise_level:.3f}, "
            f"plan_horizon={plan_horizon:.3f}, num_knots={num_knots}), "
            f"total_running_cost={float(jnp.sum(running_costs)):.3f}"
        )


def list_sweep_runs(sweep_dir: str) -> list[str]:
    """Return the sorted list of per-run .npz paths in a sweep directory.

    Useful for an analysis function that wants to iterate over runs and
    apply `compute_ess` (plus pull hyperparameters out of each file) to
    build total-cost / ESS / dataset-size arrays for plotting.
    """
    return sorted(glob.glob(os.path.join(sweep_dir, "run_*.npz")))


def plot_sweep(
    sweep_dir: str = "data/pendulum_sweep",
    temperature: float = 1.0,
    logscale: bool = False,
) -> None:
    """Aggregate a hyperparameter sweep and plot three diagnostic scatters.

    For every `run_*.npz` in `sweep_dir`, computes the total running cost,
    ESS, and dataset size via `compute_ess`, then makes three scatter plots:

        1. dataset size       vs total running cost
        2. ESS                vs total running cost
        3. ESS / dataset size vs total running cost

    Args:
        sweep_dir: Directory containing per-run .npz files.
        temperature: Temperature used in the ESS weighting.
        logscale: Whether to use log scale on both axes of all three plots.
    """
    files = list_sweep_runs(sweep_dir)
    if not files:
        raise FileNotFoundError(f"No run_*.npz files found in {sweep_dir!r}")

    total_costs = np.zeros(len(files))
    esses = np.zeros(len(files))
    dataset_sizes = np.zeros(len(files))

    for i, path in enumerate(files):
        total, ess, size = compute_ess(path, temperature)
        total_costs[i] = total
        esses[i] = ess
        dataset_sizes[i] = size

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].scatter(dataset_sizes, total_costs)
    axes[0].set_xlabel("Dataset size (num_steps * num_samples)")
    axes[0].set_ylabel("Total running cost")
    axes[0].set_title("Dataset size vs total cost")
    if logscale:
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].grid(True, which="both", ls="--", lw=0.5)

    axes[1].scatter(esses, total_costs)
    axes[1].set_xlabel("Effective Sample Size (ESS)")
    axes[1].set_ylabel("Total running cost")
    axes[1].set_title("ESS vs total cost")
    if logscale:
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].grid(True, which="both", ls="--", lw=0.5)

    axes[2].scatter(esses / dataset_sizes, total_costs)
    axes[2].set_xlabel("ESS / dataset size")
    axes[2].set_ylabel("Total running cost")
    axes[2].set_title("ESS / dataset size vs total cost")
    if logscale:
        axes[2].set_xscale("log")
        axes[2].set_yscale("log")
        axes[2].grid(True, which="both", ls="--", lw=0.5)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # sweep_pendulum_hyperparams(num_runs=10, output_dir="data/pendulum_sweep")
    plot_sweep("data/pendulum_sweep", temperature=1e-1, logscale=False)
