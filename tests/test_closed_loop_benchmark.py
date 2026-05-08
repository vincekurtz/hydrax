"""Minimal unit tests for the closed-loop benchmarking utility."""

import jax
import jax.numpy as jnp
import pytest
from mujoco import mjx

from hydrax.algs import CEM, MPPI
from hydrax.benchmarking import (
    ClosedLoopBenchmarkResult,
    run_closed_loop_benchmark,
)
from hydrax.tasks.pendulum import Pendulum


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


def _num_replans_for(
    dt: float,
    total_time: float,
    replan_frequency: float,
) -> int:
    requested_replan_period = 1.0 / replan_frequency
    sim_steps_per_replan = max(int(round(requested_replan_period / dt)), 1)
    replan_period = sim_steps_per_replan * dt
    return max(int(jnp.ceil(total_time / replan_period)), 1)


def test_output_shape_and_type(ctrl: MPPI, initial_state: mjx.Data) -> None:
    """Closed-loop result arrays must have expected shapes and dtypes."""
    total_time = 0.2
    frequency = 25.0
    n = _num_replans_for(float(ctrl.task.dt), total_time, frequency)

    result = run_closed_loop_benchmark(
        ctrl,
        initial_state,
        total_time=total_time,
        replan_frequency=frequency,
        seed=0,
    )

    assert isinstance(result, ClosedLoopBenchmarkResult)
    assert result.costs.shape == (n,)
    assert result.times.shape == (n,)
    assert result.costs.dtype in (jnp.float32, jnp.float64)


def test_values_finite(ctrl: MPPI, initial_state: mjx.Data) -> None:
    """All returned values must be finite."""
    result = run_closed_loop_benchmark(
        ctrl,
        initial_state,
        total_time=0.2,
        replan_frequency=25.0,
        seed=0,
    )
    assert jnp.all(jnp.isfinite(result.costs))
    assert jnp.all(jnp.isfinite(result.times))


def test_determinism(ctrl: MPPI, initial_state: mjx.Data) -> None:
    """Identical seeds must produce identical benchmark curves."""
    kwargs = {
        "total_time": 0.2,
        "replan_frequency": 25.0,
    }
    r1 = run_closed_loop_benchmark(ctrl, initial_state, seed=42, **kwargs)
    r2 = run_closed_loop_benchmark(ctrl, initial_state, seed=42, **kwargs)

    assert jnp.allclose(r1.costs, r2.costs)
    assert jnp.allclose(r1.times, r2.times)


def test_different_seeds_differ(ctrl: MPPI, initial_state: mjx.Data) -> None:
    """Different seeds should almost surely produce different trajectories."""
    kwargs = {
        "total_time": 0.2,
        "replan_frequency": 25.0,
    }
    r1 = run_closed_loop_benchmark(ctrl, initial_state, seed=0, **kwargs)
    r2 = run_closed_loop_benchmark(ctrl, initial_state, seed=99, **kwargs)

    assert not jnp.allclose(r1.costs, r2.costs)


def test_parity_with_manual_loop(
    task: Pendulum,
    initial_state: mjx.Data,
) -> None:
    """JIT scan benchmark must match a manual equivalent MPC loop."""
    ctrl = MPPI(
        task,
        num_samples=16,
        noise_level=0.2,
        temperature=0.1,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=4,
    )

    total_time = 0.2
    replan_frequency = 25.0
    seed = 7

    # --- benchmark path ---
    result = run_closed_loop_benchmark(
        ctrl,
        initial_state,
        total_time=total_time,
        replan_frequency=replan_frequency,
        seed=seed,
    )

    # --- manual path ---
    sim_dt = float(ctrl.task.dt)
    requested_replan_period = 1.0 / replan_frequency
    sim_steps_per_replan = max(int(round(requested_replan_period / sim_dt)), 1)
    replan_period = sim_steps_per_replan * sim_dt
    num_replans = max(int(jnp.ceil(total_time / replan_period)), 1)

    params = ctrl.init_params(seed=seed)
    state = initial_state
    jit_optimize = jax.jit(ctrl.optimize)

    manual_costs = []
    cumulative = jnp.array(0.0)

    for _ in range(num_replans):
        params, _ = jit_optimize(state, params)

        tq = state.time + jnp.arange(sim_steps_per_replan) * sim_dt
        knots = jnp.clip(params.mean, ctrl.task.u_min, ctrl.task.u_max)[None]
        controls = ctrl.interp_func(tq, params.tk, knots)[0]

        for u in controls:
            state = state.replace(ctrl=u)
            state = mjx.step(ctrl.task.model, state)
            cumulative = cumulative + sim_dt * ctrl.task.running_cost(state, u)

        manual_costs.append(cumulative)

    manual_costs = jnp.stack(manual_costs)
    assert jnp.allclose(result.costs, manual_costs, rtol=1e-5, atol=1e-5)


def test_custom_algorithm(task: Pendulum, initial_state: mjx.Data) -> None:
    """Benchmark must work for non-MPPI algorithms too."""
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
    result = run_closed_loop_benchmark(
        ctrl,
        initial_state,
        total_time=0.2,
        replan_frequency=25.0,
        seed=0,
    )

    assert result.costs.shape[0] >= 1
    assert jnp.all(jnp.isfinite(result.costs))


def test_preinitialized_params(ctrl: MPPI, initial_state: mjx.Data) -> None:
    """Explicit params should override seed and yield identical results."""
    params = ctrl.init_params(seed=5)
    kwargs = {
        "total_time": 0.2,
        "replan_frequency": 25.0,
        "params": params,
    }
    r1 = run_closed_loop_benchmark(ctrl, initial_state, **kwargs)
    r2 = run_closed_loop_benchmark(
        ctrl,
        initial_state,
        seed=99,
        **kwargs,
    )

    assert jnp.allclose(r1.costs, r2.costs)
    assert jnp.allclose(r1.times, r2.times)


def test_invalid_total_time(ctrl: MPPI, initial_state: mjx.Data) -> None:
    """Non-positive total time should raise ValueError."""
    with pytest.raises(ValueError, match="total_time"):
        run_closed_loop_benchmark(
            ctrl,
            initial_state,
            total_time=0.0,
            replan_frequency=25.0,
        )


def test_invalid_replan_frequency(ctrl: MPPI, initial_state: mjx.Data) -> None:
    """Non-positive replanning frequency should raise ValueError."""
    with pytest.raises(ValueError, match="replan_frequency"):
        run_closed_loop_benchmark(
            ctrl,
            initial_state,
            total_time=0.2,
            replan_frequency=0.0,
        )
