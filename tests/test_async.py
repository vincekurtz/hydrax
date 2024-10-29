import time
from multiprocessing import Event, Process

import mujoco
import numpy as np

from hydrax import ROOT
from hydrax.algs.predictive_sampling import PredictiveSampling
from hydrax.simulation.asynchronous import (
    SharedMemoryMujocoData,
    SharedMemoryNumpyArray,
    run_controller,
    run_interactive,
)
from hydrax.tasks.pendulum import Pendulum


def test_shared_nparray() -> None:
    """Test reading and writing a shared-memory numpy array."""
    original = np.array([0.0, 1.0, 2.0, 3.0])
    shared = SharedMemoryNumpyArray(original)

    def _write(shared: SharedMemoryNumpyArray) -> None:
        shared[0] = 4.0

    proc = Process(target=_write, args=(shared,))
    proc.start()
    proc.join()

    assert shared[0] == 4.0
    assert original[0] == 0.0
    assert np.all(shared[1:] == original[1:])


def test_controller() -> None:
    """Test running the controller in a separate process."""
    mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/pendulum/scene.xml")
    mj_data = mujoco.MjData(mj_model)

    def _setup_fn() -> PredictiveSampling:
        """Set up the controller."""
        task = Pendulum()
        return PredictiveSampling(task, num_samples=8, noise_level=0.1)

    shared_mjdata = SharedMemoryMujocoData(mj_data)
    ready = Event()
    finished = Event()
    proc = Process(
        target=run_controller, args=(_setup_fn, shared_mjdata, ready, finished)
    )

    proc.start()
    time.sleep(5)
    finished.set()
    proc.join()

    assert not proc.is_alive()
    assert ready.is_set()
    assert finished.is_set()


def manual_test_interactive() -> None:
    """Test running an interactive simulation.

    Note that this does not as a normal test, only when called directly, because
    this will open a window and block the test suite.
    """
    mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/pendulum/scene.xml")
    mj_data = mujoco.MjData(mj_model)

    def _setup_fn() -> PredictiveSampling:
        """Set up the controller."""
        task = Pendulum()
        return PredictiveSampling(task, num_samples=8, noise_level=0.1)

    run_interactive(_setup_fn, mj_model, mj_data)


if __name__ == "__main__":
    test_shared_nparray()
    test_controller()
    manual_test_interactive()
