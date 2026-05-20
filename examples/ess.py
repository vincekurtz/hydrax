"""Test Effective Sample Size (ESS) statistics as a performance predictor."""

import jax.numpy as jnp

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


if __name__ == "__main__":
    run_pendulum_benchmark(num_samples=128)
