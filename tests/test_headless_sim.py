import mujoco
import numpy as np
import pytest

from hydrax.algs import CEM, MPPI, PredictiveSampling
from hydrax.simulation.deterministic import run_headless
from hydrax.tasks.cart_pole import CartPole
from hydrax.tasks.pendulum import Pendulum


def test_headless_pendulum_basic() -> None:
    """Test basic headless simulation with pendulum task."""
    task = Pendulum()
    ctrl = PredictiveSampling(task, num_samples=8, noise_level=0.1)

    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)

    initial_qpos = mj_data.qpos.copy()

    # Run for 0.5 seconds
    run_headless(ctrl, mj_model, mj_data, frequency=50, duration=0.5)

    # Verify simulation time advanced
    assert mj_data.time >= 0.5
    # Verify state changed (pendulum should swing)
    assert not np.allclose(mj_data.qpos, initial_qpos)


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

    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)

    initial_qpos = mj_data.qpos.copy()
    initial_time = mj_data.time

    # Run for 1 second
    run_headless(ctrl, mj_model, mj_data, frequency=50, duration=1.0)

    # Verify simulation progressed
    assert mj_data.time > initial_time
    assert mj_data.time >= 1.0
    # State should have changed
    assert not np.allclose(mj_data.qpos, initial_qpos)


def test_headless_duration_termination() -> None:
    """Test that simulation stops at specified duration."""
    task = Pendulum()
    ctrl = PredictiveSampling(task, num_samples=8, noise_level=0.1)

    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)

    duration = 0.3
    run_headless(ctrl, mj_model, mj_data, frequency=50, duration=duration)

    # Verify we stopped near the requested duration (allow small tolerance)
    assert duration <= mj_data.time < duration + 0.1


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

    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)

    initial_qvel = mj_data.qvel.copy()

    run_headless(ctrl, mj_model, mj_data, frequency=50, duration=0.2)

    # Verify simulation progressed and state changed
    assert mj_data.time >= 0.2
    assert not np.allclose(mj_data.qvel, initial_qvel)


def test_headless_nonzero_initial_state() -> None:
    """Test headless simulation starting from a non-zero initial state."""
    task = Pendulum()
    ctrl = PredictiveSampling(task, num_samples=8, noise_level=0.1)

    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)

    # Set a non-zero initial state
    mj_data.qpos[:] = 1.0
    mj_data.qvel[:] = 0.5

    initial_qpos = mj_data.qpos.copy()

    run_headless(ctrl, mj_model, mj_data, frequency=50, duration=0.3)

    # State should have evolved from the non-zero initial condition
    assert mj_data.time >= 0.3
    assert not np.allclose(mj_data.qpos, initial_qpos)


def test_headless_qvel_synced() -> None:
    """Test that qvel is properly synced back to mj_data after simulation."""
    task = Pendulum()
    ctrl = PredictiveSampling(task, num_samples=8, noise_level=0.1)

    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)

    initial_qvel = mj_data.qvel.copy()

    run_headless(ctrl, mj_model, mj_data, frequency=50, duration=0.5)

    # qvel should be synced back and have changed
    assert not np.allclose(mj_data.qvel, initial_qvel)
    # qvel should be finite (no NaNs or infs)
    assert np.all(np.isfinite(mj_data.qvel))


def test_headless_short_duration() -> None:
    """Test headless simulation with a very short duration."""
    task = Pendulum()
    ctrl = PredictiveSampling(task, num_samples=8, noise_level=0.1)

    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)

    run_headless(ctrl, mj_model, mj_data, frequency=50, duration=0.02)

    # Should still advance time even for very short runs
    assert mj_data.time >= 0.02


def test_headless_high_frequency() -> None:
    """Test headless simulation with a high control frequency."""
    task = Pendulum()
    ctrl = PredictiveSampling(task, num_samples=8, noise_level=0.1)

    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)

    # High frequency means more replanning steps per second
    run_headless(ctrl, mj_model, mj_data, frequency=200, duration=0.2)

    assert mj_data.time >= 0.2
    assert np.all(np.isfinite(mj_data.qpos))
    assert np.all(np.isfinite(mj_data.qvel))


def test_headless_low_frequency() -> None:
    """Test headless simulation with a low control frequency."""
    task = Pendulum()
    ctrl = PredictiveSampling(task, num_samples=8, noise_level=0.1)

    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)

    # Low frequency means more sim steps between replanning
    run_headless(ctrl, mj_model, mj_data, frequency=10, duration=0.5)

    assert mj_data.time >= 0.5
    assert np.all(np.isfinite(mj_data.qpos))
    assert np.all(np.isfinite(mj_data.qvel))


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

    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)

    run_headless(ctrl, mj_model, mj_data, frequency=50, duration=2.0)

    # After 2 seconds, state should still be finite
    assert np.all(np.isfinite(mj_data.qpos))
    assert np.all(np.isfinite(mj_data.qvel))
    assert np.isfinite(mj_data.time)


if __name__ == "__main__":
    test_headless_pendulum_basic()
    test_headless_cart_pole()
    test_headless_duration_termination()
    test_headless_mppi()
    test_headless_nonzero_initial_state()
    test_headless_qvel_synced()
    test_headless_short_duration()
    test_headless_high_frequency()
    test_headless_low_frequency()
    test_headless_state_finite()
    print("All tests passed!")
