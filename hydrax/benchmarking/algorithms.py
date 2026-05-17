"""Factory functions for constructing each algorithm with a fixed budget.

All algorithms share the same `num_samples`, `num_knots`, `spline_type`, and
`plan_horizon`. Algorithm-specific hyperparameters (noise level, temperature,
elite count, etc.) use defaults chosen to be reasonable across the benchmark
tasks. Tuning per-task is intentionally out of scope: the goal of the
benchmark is an apples-to-apples comparison at a fixed compute budget.
"""

from typing import Callable, Dict

from evosax.algorithms.distribution_based import CMA_ES

from hydrax.alg_base import SamplingBasedController
from hydrax.algs import (
    CEM,
    DIAL,
    MPPI,
    ErCma,
    Evosax,
    MppiCma,
    PredictiveSampling,
)
from hydrax.task_base import Task

AlgFactory = Callable[..., SamplingBasedController]


def _make_ps(
    task: Task,
    num_samples: int,
    plan_horizon: float,
    num_knots: int,
    spline_type: str,
    seed: int,
) -> SamplingBasedController:
    """Predictive Sampling factory."""
    return PredictiveSampling(
        task,
        num_samples=num_samples,
        noise_level=0.3,
        plan_horizon=plan_horizon,
        spline_type=spline_type,
        num_knots=num_knots,
        seed=seed,
    )


def _make_mppi(
    task: Task,
    num_samples: int,
    plan_horizon: float,
    num_knots: int,
    spline_type: str,
    seed: int,
) -> SamplingBasedController:
    """MPPI factory."""
    return MPPI(
        task,
        num_samples=num_samples,
        noise_level=0.3,
        temperature=0.1,
        plan_horizon=plan_horizon,
        spline_type=spline_type,
        num_knots=num_knots,
        seed=seed,
    )


def _make_cem(
    task: Task,
    num_samples: int,
    plan_horizon: float,
    num_knots: int,
    spline_type: str,
    seed: int,
) -> SamplingBasedController:
    """Cross-Entropy Method factory."""
    num_elites = max(num_samples // 8, 2)
    return CEM(
        task,
        num_samples=num_samples,
        num_elites=num_elites,
        sigma_start=0.3,
        sigma_min=0.05,
        explore_fraction=0.25,
        plan_horizon=plan_horizon,
        spline_type=spline_type,
        num_knots=num_knots,
        seed=seed,
    )


def _make_dial(
    task: Task,
    num_samples: int,
    plan_horizon: float,
    num_knots: int,
    spline_type: str,
    seed: int,
) -> SamplingBasedController:
    """DIAL-MPC factory."""
    return DIAL(
        task,
        num_samples=num_samples,
        noise_level=0.4,
        beta_opt_iter=1.0,
        beta_horizon=1.0,
        temperature=0.1,
        plan_horizon=plan_horizon,
        spline_type=spline_type,
        num_knots=num_knots,
        seed=seed,
    )


def _make_cmaes(
    task: Task,
    num_samples: int,
    plan_horizon: float,
    num_knots: int,
    spline_type: str,
    seed: int,
) -> SamplingBasedController:
    """CMA-ES via Evosax factory."""
    return Evosax(
        task,
        CMA_ES,
        num_samples=num_samples,
        plan_horizon=plan_horizon,
        spline_type=spline_type,
        num_knots=num_knots,
        seed=seed,
    )


def _make_mppi_cma(
    task: Task,
    num_samples: int,
    plan_horizon: float,
    num_knots: int,
    spline_type: str,
    seed: int,
) -> SamplingBasedController:
    """MPPI-CMA factory."""
    return MppiCma(
        task,
        num_samples=num_samples,
        initial_noise_level=0.3,
        minimum_noise_level=1e-3,
        covariance_adaptation_rate=0.1,
        temperature=0.1,
        plan_horizon=plan_horizon,
        spline_type=spline_type,
        num_knots=num_knots,
        seed=seed,
    )


def _make_er_cma(
    task: Task,
    num_samples: int,
    plan_horizon: float,
    num_knots: int,
    spline_type: str,
    seed: int,
) -> SamplingBasedController:
    """ER-CMA factory."""
    return ErCma(
        task,
        num_samples=num_samples,
        initial_noise_level=0.3,
        minimum_noise_level=1e-3,
        maximum_noise_level=1.0,
        initial_entropy_bonus=0.3,
        final_entropy_bonus=0.5,
        covariance_adaptation_rate=0.1,
        temperature=0.1,
        plan_horizon=plan_horizon,
        spline_type=spline_type,
        num_knots=num_knots,
        seed=seed,
    )


ALGORITHMS: Dict[str, AlgFactory] = {
    "ps": _make_ps,
    "mppi": _make_mppi,
    "cem": _make_cem,
    "dial": _make_dial,
    "cmaes": _make_cmaes,
    "mppi_cma": _make_mppi_cma,
    "er_cma": _make_er_cma,
}


def make_algorithm(
    name: str,
    task: Task,
    num_samples: int,
    plan_horizon: float,
    num_knots: int,
    spline_type: str,
    seed: int,
) -> SamplingBasedController:
    """Instantiate an algorithm by name with a shared compute budget."""
    if name not in ALGORITHMS:
        raise KeyError(
            f"Unknown algorithm '{name}'. Available: {sorted(ALGORITHMS)}"
        )
    return ALGORITHMS[name](
        task, num_samples, plan_horizon, num_knots, spline_type, seed
    )
