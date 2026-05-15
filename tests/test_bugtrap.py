import jax.numpy as jnp
import pytest
from mujoco import mjx

from hydrax.tasks.bugtrap import BugTrap


@pytest.mark.parametrize("impl", ["jax", "warp"])
def test_bugtrap(impl: str) -> None:
    """Smoke-test the bug-trap navigation task."""
    task = BugTrap(impl=impl)
    assert task.pointmass_id >= 0
    assert task._wall_pos.shape == (3, 2)
    assert task._wall_size.shape == (3, 2)

    state = task.make_data()
    state = state.replace(mocap_pos=jnp.array([[0.25, 0.0, 0.01]]))
    assert isinstance(state, mjx.Data)
    state = mjx.forward(task.model, state)

    ell = task.running_cost(state, jnp.zeros(2))
    assert ell.shape == ()
    assert ell > 0.0

    phi = task.terminal_cost(state)
    assert phi.shape == ()
    assert phi > 0.0


if __name__ == "__main__":
    test_bugtrap("jax")
    test_bugtrap("warp")
