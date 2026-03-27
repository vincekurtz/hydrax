import mujoco
import numpy as np
import pytest
from mujoco import mjx

from hydrax.algs import CEM, MPPI, PredictiveSampling
from hydrax.simulation.deterministic import run_headless
from hydrax.tasks.cart_pole import CartPole
from hydrax.tasks.pendulum import Pendulum


def _make_mjx(task, qpos=None, qvel=None):
    """Helper to create mjx model/data from a task."""
    mj_model = task.mj_model
    mjx_model_sim = mjx.put_model(mj_model)
    mjx_data_sim = mjx.make_data(mj_model)
    if qpos is not None:
        mjx_data_sim = mjx_data_sim.replace(qpos=qpos)
    if qvel is not None:
        mjx_data_sim = mjx_data_sim.replace(qvel=qvel)
    return mjx_model_sim, mjx_data_sim


def test_headless_pendulum_basic() -> None:
    """Test basic headless simulation with pendulum task."""
    task = Pendulum()
    ctrl = PredictiveSampling(task, num_samples=8, noise_level=0.1)
    mjx_model_sim, mjx_data_sim = _make_mjx(task)

    result = run_headless(ctrl, mjx_model_sim, mjx_data_sim, frequency=50, duration=0.5)

    # Verify trajectory was recorded with correct shape
    nq = task.mj_model.nq
    assert result["qpos"].shape[1] == nq
    assert result["qpos"].shape[0] > 0
    # Verify all entries are finite (no leftover NaNs)
    assert np.all(np.isfinite(result["qpos"]))
    # Verify state changed (pendulum should swing)
    assert not np.allclose(result["qpos"][0], result["qpos"][-1])


def test_headless_cart_pole() -> None:
    """Test headless simulation with cart pole task."""
    task = CartPole()
    ctrl = CEM(
        task,
        num_samples=32,
        num_elites=3,
        sigma_start=0.5,
        sigma_min=0.1,
        plan_horizon=1.0,
        num_knots=4,
    )
    mjx_model_sim, mjx_data_sim = _make_mjx(task)

    result = run_headless(ctrl, mjx_model_sim, mjx_data_sim, frequency=50, duration=1.0)

    # Verify trajectory shape
    nq = task.mj_model.nq
    nv = task.mj_model.nv
    assert result["qpos"].shape[1] == nq
    assert result["qvel"].shape[1] == nv
    assert result["qpos"].shape[0] == result["qvel"].shape[0]
    # State should have changed
    assert not np.allclose(result["qpos"][0], result["qpos"][-1])


def test_headless_duration_steps() -> None:
    """Test that the number of recorded steps matches the expected duration."""
    task = Pendulum()
    ctrl = PredictiveSampling(task, num_samples=8, noise_level=0.1)
    mjx_model_sim, mjx_data_sim = _make_mjx(task)

    duration = 0.3
    frequency = 50
    result = run_headless(ctrl, mjx_model_sim, mjx_data_sim, frequency=frequency, duration=duration)

    sim_dt = task.mj_model.opt.timestep
    sim_steps_per_replan = max(int((1.0 / frequency) / sim_dt), 1)
    step_dt = sim_steps_per_replan * sim_dt
    num_replan_steps = int(np.ceil(duration / step_dt))
    expected_total = num_replan_steps * sim_steps_per_replan

    assert result["qpos"].shape[0] == expected_total
    assert result["qvel"].shape[0] == expected_total


def test_headless_mppi() -> None:
    """Test headless simulation with MPPI controller."""
    task = Pendulum()
    ctrl = MPPI(
        task,
        num_samples=32,
        noise_level=0.2,
        temperature=0.1,
        plan_horizon=0.5,
        num_knots=4,
    )
    mjx_model_sim, mjx_data_sim = _make_mjx(task)

    result = run_headless(ctrl, mjx_model_sim, mjx_data_sim, frequency=50, duration=0.2)

    assert np.all(np.isfinite(result["qpos"]))
    assert np.all(np.isfinite(result["qvel"]))
    assert not np.allclose(result["qvel"][0], result["qvel"][-1])


def test_headless_nonzero_initial_state() -> None:
    """Test headless simulation starting from a non-zero initial state."""
    task = Pendulum()
    ctrl = PredictiveSampling(task, num_samples=8, noise_level=0.1)

    initial_qpos = np.array([1.0])
    initial_qvel = np.array([0.5])
    mjx_model_sim, mjx_data_sim = _make_mjx(task, qpos=initial_qpos, qvel=initial_qvel)

    result = run_headless(ctrl, mjx_model_sim, mjx_data_sim, frequency=50, duration=0.3)

    # State should have evolved from the non-zero initial condition
    assert not np.allclose(result["qpos"][-1], initial_qpos)


def test_headless_qvel_recorded() -> None:
    """Test that qvel trajectory is properly recorded."""
    task = Pendulum()
    ctrl = PredictiveSampling(task, num_samples=8, noise_level=0.1)
    mjx_model_sim, mjx_data_sim = _make_mjx(task)

    result = run_headless(ctrl, mjx_model_sim, mjx_data_sim, frequency=50, duration=0.5)

    # qvel should be finite throughout
    assert np.all(np.isfinite(result["qvel"]))
    # qvel should change over time
    assert not np.allclose(result["qvel"][0], result["qvel"][-1])


def test_headless_short_duration() -> None:
    """Test headless simulation with a very short duration."""
    task = Pendulum()
    ctrl = PredictiveSampling(task, num_samples=8, noise_level=0.1)
    mjx_model_sim, mjx_data_sim = _make_mjx(task)

    result = run_headless(ctrl, mjx_model_sim, mjx_data_sim, frequency=50, duration=0.02)

    # Should still have recorded steps
    assert result["qpos"].shape[0] > 0
    assert np.all(np.isfinite(result["qpos"]))


def test_headless_high_frequency() -> None:
    """Test headless simulation with a high control frequency."""
    task = Pendulum()
    ctrl = PredictiveSampling(task, num_samples=8, noise_level=0.1)
    mjx_model_sim, mjx_data_sim = _make_mjx(task)

    result = run_headless(ctrl, mjx_model_sim, mjx_data_sim, frequency=200, duration=0.2)

    assert np.all(np.isfinite(result["qpos"]))
    assert np.all(np.isfinite(result["qvel"]))


def test_headless_low_frequency() -> None:
    """Test headless simulation with a low control frequency."""
    task = Pendulum()
    ctrl = PredictiveSampling(task, num_samples=8, noise_level=0.1)
    mjx_model_sim, mjx_data_sim = _make_mjx(task)

    result = run_headless(ctrl, mjx_model_sim, mjx_data_sim, frequency=10, duration=0.5)

    assert np.all(np.isfinite(result["qpos"]))
    assert np.all(np.isfinite(result["qvel"]))


def test_headless_state_finite() -> None:
    """Test that simulation state remains finite (no NaN/inf divergence)."""
    task = CartPole()
    ctrl = CEM(
        task,
        num_samples=32,
        num_elites=3,
        sigma_start=0.5,
        sigma_min=0.1,
        plan_horizon=1.0,
        num_knots=4,
    )
    mjx_model_sim, mjx_data_sim = _make_mjx(task)

    result = run_headless(ctrl, mjx_model_sim, mjx_data_sim, frequency=50, duration=2.0)

    assert np.all(np.isfinite(result["qpos"]))
    assert np.all(np.isfinite(result["qvel"]))


if __name__ == "__main__":
    test_headless_pendulum_basic()
    test_headless_cart_pole()
    test_headless_duration_steps()
    test_headless_mppi()
    test_headless_nonzero_initial_state()
    test_headless_qvel_recorded()
    test_headless_short_duration()
    test_headless_high_frequency()
    test_headless_low_frequency()
    test_headless_state_finite()
    print("All tests passed!")
