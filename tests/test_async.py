import numpy as np
from multiprocessing import Process

from hydrax.simulation.asynchronous import SharedMemoryNumpyArray


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


if __name__ == "__main__":
    test_shared_nparray()