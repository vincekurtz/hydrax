import glob
import os
import shutil

import mujoco
import numpy as np
import pytest

from hydrax import ROOT
from hydrax.algs import CEM, MPPI, PredictiveSampling
from hydrax.simulation.deterministic import run_headless
from hydrax.tasks.cart_pole import CartPole
from hydrax.tasks.pendulum import Pendulum


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


def test_headless_video_recording() -> None:
    """Test that video recording works in headless mode."""
    recordings_dir = os.path.join(ROOT, "recordings")

    # Clean up any existing recordings
    if os.path.exists(recordings_dir):
        shutil.rmtree(recordings_dir)

    task = Pendulum()
    ctrl = PredictiveSampling(task, num_samples=8, noise_level=0.1)

    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)

    # Run with video recording enabled
    run_headless(
        ctrl, mj_model, mj_data, frequency=50, duration=0.2, record_video=True
    )

    # Verify recordings directory was created
    assert os.path.exists(recordings_dir), "Recordings directory was not created"

    # Check that at least one video file was created
    video_files = glob.glob(os.path.join(recordings_dir, "*.mp4"))
    assert len(video_files) > 0, "No video files were created"

    # Verify video file has non-zero size
    for video_file in video_files:
        file_size = os.path.getsize(video_file)
        assert file_size > 0, f"Video file {video_file} is empty"


if __name__ == "__main__":
    test_headless_cart_pole()
    test_headless_duration_termination()
    test_headless_mppi()
    test_headless_video_recording()