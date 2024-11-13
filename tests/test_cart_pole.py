import jax.numpy as jnp
from mujoco import mjx

from hydrax.tasks.cart_pole import CartPole


def test_cart_pole() -> None:
    """Make sure we can instantiate the task."""
    task = CartPole()
    assert isinstance(task, CartPole)

    state = mjx.make_data(task.model)
    assert isinstance(state, mjx.Data)

    ell = task.running_cost(state, jnp.zeros(1))
    assert ell.shape == ()

    phi = task.terminal_cost(state)
    assert phi.shape == ()
    assert phi >= 0.0


if __name__ == "__main__":
    test_cart_pole()
