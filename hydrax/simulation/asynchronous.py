import time
from multiprocessing import Event, Lock, shared_memory
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import mjx

from hydrax.alg_base import SamplingBasedController

"""
Utilities for asynchronous simulation, with the simulator and controller running
in separate processes. The controller runs as fast as possible, while the
simulator runs in real time.

This is more realistic than deterministic simulation, but supports a limited set
of features (e.g., no trace visualization, zero-order-hold interpolation only).
"""


class SharedMemoryNumpyArray:
    """Helper class to store a numpy array in shared memory."""

    def __init__(self, arr: np.ndarray):
        """Create a shared memory numpy array.

        Args:
            arr: The numpy array to store in shared memory. Size and dtype must
                 be fixed.
        """
        self.shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        self.data = np.ndarray(arr.shape, dtype=arr.dtype, buffer=self.shm.buf)
        self.data[:] = arr[:]
        self.lock = Lock()

    def __getitem__(self, key: int) -> np.ndarray:
        """Get an item from the shared array."""
        return self.data[key]

    def __setitem__(self, key: int, value: np.ndarray) -> None:
        """Set an item in the shared array."""
        with self.lock:
            self.data[key] = value

    def __str__(self) -> str:
        """Return the string representation of the shared array."""
        return str(self.data)

    def __del__(self) -> None:
        """Clean up the shared memory on deletion."""
        self.shm.close()
        self.shm.unlink()

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the shared array."""
        return self.shared_arr.shape


class SharedMemoryMujocoData:
    """Helper class for passing mujoco data between concurrent processes."""

    def __init__(self, mj_data: mujoco.MjData):
        """Create shared memory objects for state and control data.

        Note that this does not copy the full mj_data object, only those fields
        that we want to share between the simulator and controller.

        Args:
            mj_data: The mujoco data object to store in shared memory.
        """
        # N.B. we use float32 to match JAX's default precision
        self.qpos = SharedMemoryNumpyArray(
            np.array(mj_data.qpos, dtype=np.float32)
        )
        self.qvel = SharedMemoryNumpyArray(
            np.array(mj_data.qvel, dtype=np.float32)
        )
        self.ctrl = SharedMemoryNumpyArray(
            np.zeros(mj_data.ctrl.shape, dtype=np.float32)
        )

        if len(mj_data.mocap_pos) > 0:
            self.mocap_pos = SharedMemoryNumpyArray(
                np.array(mj_data.mocap_pos, dtype=np.float32)
            )
            self.mocap_quat = SharedMemoryNumpyArray(
                np.array(mj_data.mocap_quat, dtype=np.float32)
            )
        else:
            # We need non-empty arrays, even if they are not used
            self.mocap_pos = SharedMemoryNumpyArray(
                np.zeros((3,), dtype=np.float32)
            )
            self.mocap_quat = SharedMemoryNumpyArray(
                np.zeros((4,), dtype=np.float32)
            )


def controller(
    setup_fn: Callable[[], SamplingBasedController],
    shm_data: SharedMemoryMujocoData,
    ready: Event,
    finished: Event,
) -> None:
    """Run the controller, communicating with the simulator over shared memory.

    Note: we need to create the controller within this process, otherwise
    JAX will complain about sharing data across processes. That's why we take
    a callable `setup_fn` to create the controller, rather than the controller
    itself.

    Args:
        setup_fn: Function to set up the controller.
        shm_data: Shared memory object for state and control action data.
        ready: Shared flag for signaling that the controller is ready.
        finished: Shared flag for stopping the simulation.
    """
    # Set up the controller
    ctrl = setup_fn()
    mjx_data = mjx.make_data(ctrl.task.model)
    policy_params = ctrl.init_params()

    # Jit the optimizer step, then signal that we're ready to go
    print("Jitting controller...")
    st = time.time()
    jit_optimize = jax.jit(
        lambda d, p: ctrl.optimize(d, p)[0], donate_argnums=(1,)
    )
    get_action = jax.jit(ctrl.get_action)
    policy_params = jit_optimize(mjx_data, policy_params)
    print(f"Time to jit: {time.time() - st}")

    # Signal that we're ready to start
    ready.set()

    while not finished.is_set():
        st = time.time()

        # Set the start state for the controller, reading the lastest state info
        # from shared memory
        mjx_data = mjx_data.replace(
            qpos=jnp.array(shm_data.qpos.data),
            qvel=jnp.array(shm_data.qvel.data),
            mocap_pos=jnp.array(shm_data.mocap_pos.data),
            mocap_quat=jnp.array(shm_data.mocap_quat.data),
        )

        # Do a planning step
        policy_params = jit_optimize(mjx_data, policy_params)

        # Send the action to the simulator
        shm_data.ctrl[:] = np.array(
            get_action(policy_params, 0.0), dtype=np.float32
        )

        # Print the current planning frequency
        print(f"Controller running at {1/(time.time() - st):.2f} Hz")
