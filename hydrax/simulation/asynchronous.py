from typing import Any, Callable, Tuple
import jax
import time
from multiprocessing import Process, shared_memory, Lock, Event, Queue
import numpy as np
import jax.numpy as jnp

import mujoco
import mujoco.viewer
from mujoco import mjx

from hydrax.alg_base import SamplingBasedController
from hydrax.task_base import Task


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

    def __getitem__(self, key):
        """Get an item from the shared array."""
        return self.data[key]

    def __setitem__(self, key, value):
        """Set an item in the shared array."""
        with self.lock:
            self.data[key] = value

    def __str__(self):
        """Return the string representation of the shared array."""
        return str(self.data)

    def __del__(self):
        """Clean up the shared memory on deletion."""
        self.shm.close()
        self.shm.unlink()

    @property
    def shape(self):
        """Return the shape of the shared array."""
        return self.shared_arr.shape