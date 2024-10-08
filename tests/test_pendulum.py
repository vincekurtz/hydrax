import jax.numpy as jnp
from mujoco import mjx

from hydra.tasks.pendulum import Pendulum


def test_pendulum() -> None:
    """Make sure we can instantiate the Pendulum task."""
    task = Pendulum()
    assert isinstance(task, Pendulum)

    state = mjx.make_data(task.model)
    assert isinstance(state, mjx.Data)

    ell = task.running_cost(state, jnp.zeros(1))
    assert ell.shape == ()

    phi = task.terminal_cost(state)
    assert phi.shape == ()
    assert phi > 0.0

    y = task.get_obs(state)
    assert y.shape == (2,)


if __name__ == "__main__":
    test_pendulum()
