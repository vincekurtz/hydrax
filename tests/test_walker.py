import jax.numpy as jnp
import pytest
from mujoco import mjx

from hydrax.tasks.walker import Walker


@pytest.mark.parametrize("impl", ["jax", "warp"])
def test_walker(impl: str) -> None:
    """Make sure we can instantiate the Walker task.

    Args:
        impl: Which implementation to use ("jax" or "warp").
    """
    task = Walker(impl=impl)
    assert isinstance(task, Walker)
    assert task.torso_position_sensor >= 0
    assert task.torso_velocity_sensor >= 0
    assert task.torso_zaxis_sensor >= 0

    state = task.make_data()
    assert isinstance(state, mjx.Data)

    # Check sensor measurements
    state = mjx.forward(task.model, state)
    pz = task._get_torso_height(state)
    vx = task._get_torso_velocity(state)
    oz = task._get_torso_deviation_from_upright(state)
    assert pz > 0.0
    assert vx == 0.0
    assert oz == 0.0

    # Check costs
    ell = task.running_cost(state, jnp.zeros(task.model.nu))
    assert ell.shape == ()
    assert ell > 0.0

    phi = task.terminal_cost(state)
    assert phi.shape == ()
    assert phi > 0.0


if __name__ == "__main__":
    test_walker("jax")
    test_walker("warp")
