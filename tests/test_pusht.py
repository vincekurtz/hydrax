import jax
import jax.numpy as jnp
import pytest
from mujoco import mjx

from hydrax.tasks.pusht import PushT


@pytest.mark.parametrize("impl", ["jax", "warp"])
def test_task(impl: str) -> None:
    """Set up the push T task.

    Args:
        impl: Which implementation to use ("jax" or "warp").
    """
    task = PushT(impl=impl)

    state = task.make_data()
    assert isinstance(state, mjx.Data)
    state = state.replace(mocap_quat=jnp.array([[0.0, 1.0, 0.0, 0.0]]))
    state = jax.jit(mjx.forward)(task.model, state)

    pos = task._get_position_err(state)
    assert pos.shape == (3,)

    ori = task._get_orientation_err(state)
    assert ori.shape == (3,)

    ell = task.running_cost(state, jnp.zeros(2))
    assert ell.shape == ()

    phi = task.terminal_cost(state)
    assert phi.shape == ()


if __name__ == "__main__":
    test_task("jax")
    test_task("warp")
