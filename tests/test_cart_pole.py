import jax.numpy as jnp
import pytest
from mujoco import mjx

from hydrax.tasks.cart_pole import CartPole


@pytest.mark.parametrize("impl", ["jax", "warp"])
def test_cart_pole(impl: str) -> None:
    """Make sure we can instantiate the task.

    Args:
        impl: Which implementation to use ("jax" or "warp")
    """
    task = CartPole(impl=impl)
    assert isinstance(task, CartPole)

    state = task.make_data()
    assert isinstance(state, mjx.Data)

    ell = task.running_cost(state, jnp.zeros(1))
    assert ell.shape == ()

    phi = task.terminal_cost(state)
    assert phi.shape == ()
    assert phi >= 0.0


if __name__ == "__main__":
    test_cart_pole("jax")
    test_cart_pole("warp")
