import jax.numpy as jnp
import pytest
from mujoco import mjx

from hydrax.tasks.pendulum import Pendulum


@pytest.mark.parametrize("impl", ["jax", "warp"])
def test_pendulum(impl: str) -> None:
    """Make sure we can instantiate the Pendulum task.

    Args:
        impl: Which implementation to use ("jax" or "warp")
    """
    task = Pendulum(impl=impl)
    assert isinstance(task, Pendulum)

    state = task.make_data()
    assert isinstance(state, mjx.Data)

    ell = task.running_cost(state, jnp.zeros(1))
    assert ell.shape == ()

    phi = task.terminal_cost(state)
    assert phi.shape == ()
    assert phi >= 0.0


if __name__ == "__main__":
    test_pendulum("jax")
    test_pendulum("warp")
