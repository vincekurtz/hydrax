import jax.numpy as jnp
from mujoco import mjx

from hydrax.tasks.crane import Crane


def test_crane() -> None:
    """Make sure we can instantiate the luffing crane task."""
    task = Crane()
    assert isinstance(task, Crane)
    assert task.payload_pos_sensor_adr >= 0
    assert task.payload_vel_sensor_adr >= 0

    state = mjx.make_data(task.model)
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
    test_crane()
