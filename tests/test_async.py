import multiprocessing as mp
import time

import mujoco
import numpy as np

from hydrax import ROOT
from hydrax.algs.predictive_sampling import PredictiveSampling
from hydrax.simulation.asynchronous import (
    SharedMemoryMujocoData,
    SharedMemoryNumpyArray,
    run_controller,
    run_interactive,
    run_simulator,
)
from hydrax.tasks.pendulum import Pendulum


def _write_to_shared_nparray(shared: SharedMemoryNumpyArray) -> None:
    """Write to a shared-memory numpy array."""
    shared[0] = 4.0


def test_shared_nparray() -> None:
    """Test reading and writing a shared-memory numpy array."""
    ctx = mp.get_context("spawn")

    original = np.array([0.0, 1.0, 2.0, 3.0])
    shared = SharedMemoryNumpyArray(original, ctx)

    proc = ctx.Process(target=_write_to_shared_nparray, args=(shared,))
    proc.start()
    proc.join()

    assert shared[0] == 4.0
    assert original[0] == 0.0
    assert np.all(shared[1:] == original[1:])


def test_controller() -> None:
    """Test running the controller in a separate process."""
    ctx = mp.get_context("spawn")

    mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/pendulum/scene.xml")
    mj_data = mujoco.MjData(mj_model)

    task = Pendulum()
    ctrl = PredictiveSampling(task, num_samples=8, noise_level=0.1)

    shared_mjdata = SharedMemoryMujocoData(mj_data, ctx)
    assert shared_mjdata.ctrl[0] == 0.0

    ready = ctx.Event()
    finished = ctx.Event()
    proc = ctx.Process(
        target=run_controller, args=(ctrl, shared_mjdata, ready, finished)
    )

    proc.start()
    time.sleep(5)
    finished.set()
    proc.join()

    assert shared_mjdata.ctrl[0] != 0.0
    assert not proc.is_alive()
    assert ready.is_set()
    assert finished.is_set()


def manual_test_simulator() -> None:
    """Test running the simulator in a separate process.

    Doesn't run as part of the test suite because it opens a window.
    """
    ctx = mp.get_context("spawn")

    mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/pendulum/scene.xml")
    mj_data = mujoco.MjData(mj_model)
    shared_mjdata = SharedMemoryMujocoData(mj_data, ctx)
    assert shared_mjdata.ctrl[0] == 0.0
    assert shared_mjdata.qpos[0] == 0.0

    ready = ctx.Event()
    finished = ctx.Event()

    sim = ctx.Process(
        target=run_simulator,
        args=(mj_model, mj_data, shared_mjdata, ready, finished),
    )

    sim.start()
    ready.set()
    time.sleep(1)
    shared_mjdata.ctrl[0] = 1.0
    sim.join()  # Runs until the gui is closed

    assert shared_mjdata.qpos[0] != 0.0
    assert finished.is_set()


def manual_test_interactive() -> None:
    """Test running an interactive simulation.

    Note that this does not run as a normal test, only when called directly. It
    opens a window and would block the test suite.
    """
    mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/pendulum/scene.xml")
    mj_data = mujoco.MjData(mj_model)

    task = Pendulum()
    ctrl = PredictiveSampling(task, num_samples=64, noise_level=0.1)

    run_interactive(ctrl, mj_model, mj_data)


if __name__ == "__main__":
    test_shared_nparray()
    test_controller()
    manual_test_simulator()
    manual_test_interactive()
