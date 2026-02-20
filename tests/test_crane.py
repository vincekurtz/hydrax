import jax.numpy as jnp
import pytest
from mujoco import mjx

from hydrax.tasks.crane import Crane


@pytest.mark.parametrize("impl", ["jax", "warp"])
def test_crane(impl: str) -> None:
    """Make sure we can instantiate the luffing crane task.

    Args:
        impl: Which implementation to use ("jax" or "warp").
    """
    task = Crane(impl=impl)
    assert isinstance(task, Crane)
    assert task.payload_pos_sensor_adr >= 0
    assert task.payload_vel_sensor_adr >= 0

    state = task.make_data()
    state = state.replace(
        mocap_pos=jnp.array([[0.1, 0.1, 0.1]]),
        mocap_quat=jnp.array([[1.0, 0.0, 0.0, 0.0]]),
    )
    assert isinstance(state, mjx.Data)

    # Check sensor measurements
    state = mjx.forward(task.model, state)
    pos = task._get_payload_position(state)
    vel = task._get_payload_velocity(state)
    assert pos.shape == (3,)
    assert vel.shape == (3,)
    assert not jnp.all(pos == 0.0)

    # Check costs
    ell = task.running_cost(state, jnp.zeros(task.model.nu))
    assert ell.shape == ()
    assert ell > 0.0

    phi = task.terminal_cost(state)
    assert phi.shape == ()
    assert phi > 0.0


if __name__ == "__main__":
    test_crane("jax")
    test_crane("warp")
