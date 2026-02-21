import jax.numpy as jnp
import pytest
from mujoco import mjx

from hydrax.tasks.particle import Particle


@pytest.mark.parametrize("impl", ["jax", "warp"])
def test_particle(impl: str) -> None:
    """Make sure we can instantiate and get basic info about the task.

    Args:
        impl: Which implementation to use ("jax" or "warp")
    """
    task = Particle(impl=impl)
    assert task.pointmass_id >= 0

    state = task.make_data()
    state = state.replace(mocap_pos=jnp.array([[0.0, 0.1, 0.0]]))
    assert isinstance(state, mjx.Data)
    assert state.site_xpos.shape == (1, 3)
    state = mjx.forward(task.model, state)  # compute site positions
    assert not jnp.all(state.site_xpos == 0.0)

    ell = task.running_cost(state, jnp.zeros(2))
    assert ell.shape == ()
    assert ell > 0.0

    phi = task.terminal_cost(state)
    assert phi.shape == ()
    assert phi > 0.0


if __name__ == "__main__":
    test_particle("jax")
    test_particle("warp")
