"""Minimal unit tests for the open-loop benchmarking utility."""

import jax
import jax.numpy as jnp
import pytest
from mujoco import mjx

from hydrax.algs import CEM, MPPI
from hydrax.benchmarking import BenchmarkResult, run_open_loop_benchmark
from hydrax.tasks.pendulum import Pendulum

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def task() -> Pendulum:
    """Pendulum task instance shared across tests."""
    return Pendulum()


@pytest.fixture(scope="module")
def initial_state(task: Pendulum) -> mjx.Data:
    """Default initial state for the pendulum task."""
    return task.make_data()


@pytest.fixture(scope="module")
def ctrl(task: Pendulum) -> MPPI:
    """Small MPPI controller suitable for fast tests."""
    return MPPI(
        task,
        num_samples=16,
        noise_level=0.2,
        temperature=0.1,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=4,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_output_shape_and_type(ctrl: MPPI, initial_state: mjx.Data) -> None:
    """BenchmarkResult arrays must have the right shape and dtype."""
    N = 5
    result = run_open_loop_benchmark(ctrl, initial_state, iterations=N, seed=0)

    assert isinstance(result, BenchmarkResult)
    assert result.costs.shape == (N,)
    assert result.costs.dtype in (jnp.float32, jnp.float64)


def test_values_finite(ctrl: MPPI, initial_state: mjx.Data) -> None:
    """All returned cost values must be finite."""
    result = run_open_loop_benchmark(ctrl, initial_state, iterations=5, seed=0)

    assert jnp.all(jnp.isfinite(result.costs))


def test_determinism(ctrl: MPPI, initial_state: mjx.Data) -> None:
    """Identical seeds must produce identical cost curves."""
    r1 = run_open_loop_benchmark(ctrl, initial_state, iterations=5, seed=42)
    r2 = run_open_loop_benchmark(ctrl, initial_state, iterations=5, seed=42)

    assert jnp.allclose(r1.costs, r2.costs)


def test_different_seeds_differ(ctrl: MPPI, initial_state: mjx.Data) -> None:
    """Different seeds should (almost certainly) produce different results."""
    r1 = run_open_loop_benchmark(ctrl, initial_state, iterations=5, seed=0)
    r2 = run_open_loop_benchmark(ctrl, initial_state, iterations=5, seed=99)

    # It is astronomically unlikely for all values to match by chance
    assert not jnp.allclose(r1.costs, r2.costs)


def test_parity_with_manual_loop(
    task: Pendulum, initial_state: mjx.Data
) -> None:
    """lax.scan benchmark must match manual params.mean rollout costs."""
    ctrl = MPPI(
        task,
        num_samples=16,
        noise_level=0.2,
        temperature=0.1,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=4,
    )
    N = 4
    seed = 7

    # --- benchmark (lax.scan path) ---
    result = run_open_loop_benchmark(
        ctrl, initial_state, iterations=N, seed=seed
    )

    # --- manual Python loop (equivalent) ---
    params = ctrl.init_params(seed=seed)
    jit_optimize = jax.jit(ctrl.optimize)
    manual_costs = []
    for _ in range(N):
        params, _ = jit_optimize(initial_state, params)
        mean_knots = jnp.clip(params.mean, ctrl.task.u_min, ctrl.task.u_max)
        mean_rollout = ctrl.rollout_with_randomizations(
            initial_state,
            params.tk,
            mean_knots[None, ...],
            params.rng,
        )
        manual_costs.append(jnp.sum(mean_rollout.costs[0], axis=-1))
    manual_costs = jnp.stack(manual_costs)

    assert jnp.allclose(
        result.costs, manual_costs, rtol=1e-5, atol=1e-5
    )


def test_custom_algorithm(task: Pendulum, initial_state: mjx.Data) -> None:
    """Benchmark must work with algorithms other than MPPI (e.g. CEM)."""
    ctrl = CEM(
        task,
        num_samples=16,
        num_elites=4,
        sigma_start=0.5,
        sigma_min=0.05,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=4,
    )
    result = run_open_loop_benchmark(ctrl, initial_state, iterations=3, seed=0)

    assert result.costs.shape == (3,)
    assert jnp.all(jnp.isfinite(result.costs))


def test_preinitialized_params(ctrl: MPPI, initial_state: mjx.Data) -> None:
    """Passing explicit params should override seed and give same result."""
    params = ctrl.init_params(seed=5)
    r1 = run_open_loop_benchmark(
        ctrl, initial_state, iterations=3, params=params
    )
    # Re-run with the same params object; seed arg should be ignored
    r2 = run_open_loop_benchmark(
        ctrl, initial_state, iterations=3, seed=99, params=params
    )

    assert jnp.allclose(r1.costs, r2.costs)


def test_invalid_iterations(ctrl: MPPI, initial_state: mjx.Data) -> None:
    """Iterations < 1 must raise ValueError."""
    with pytest.raises(ValueError, match="iterations"):
        run_open_loop_benchmark(ctrl, initial_state, iterations=0)


if __name__ == "__main__":
    _task = Pendulum()
    _state = _task.make_data()
    _ctrl = MPPI(
        _task,
        num_samples=16,
        noise_level=0.2,
        temperature=0.1,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=4,
    )
    test_output_shape_and_type(_ctrl, _state)
    test_values_finite(_ctrl, _state)
    test_determinism(_ctrl, _state)
    test_different_seeds_differ(_ctrl, _state)
    test_parity_with_manual_loop(_task, _state)
    test_custom_algorithm(_task, _state)
    test_preinitialized_params(_ctrl, _state)
    test_invalid_iterations(_ctrl, _state)
    print("All tests passed!")
