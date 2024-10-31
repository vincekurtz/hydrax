import jax.numpy as jnp
from mujoco import mjx

from hydrax.tasks.walker import Walker


def test_walker() -> None:
    """Make sure we can instantiate the Pendulum task."""
    task = Walker()
    assert isinstance(task, Walker)
    assert task.torso_position_sensor >= 0
    assert task.torso_velocity_sensor >= 0
    assert task.torso_zaxis_sensor >= 0

    state = mjx.make_data(task.model)
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

    # Check observations
    obs = task.get_obs(state)
    assert obs.shape == (task.model.nq + task.model.nv - 1,)


if __name__ == "__main__":
    test_walker()
