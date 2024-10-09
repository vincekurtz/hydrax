import jax.numpy as jnp
from mujoco import mjx

from hydra.tasks.particle import Particle


def test_particle() -> None:
    """Make sure we can instantiate and get basic info about the task."""
    task = Particle()
    assert task.pointmass_id >= 0

    state = mjx.make_data(task.model)
    state = state.replace(mocap_pos=jnp.zeros((1, 3)))
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
    test_particle()
