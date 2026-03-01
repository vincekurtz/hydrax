import jax.numpy as jnp
from mujoco import mjx

from hydrax.tasks.double_pendulum import DoublePendulum


def test_double_pendulum() -> None:
    """Make sure we can instantiate the task."""
    task = DoublePendulum()
    assert isinstance(task, DoublePendulum)

    state = mjx.make_data(task.model)
    state = state.replace(qpos=jnp.array([jnp.pi, jnp.pi]))  # θ₁, θ₂
    state = mjx.forward(task.model, state)
    tip_pos = state.site_xpos[task.tip_id]
    assert tip_pos[0] != 0.0  # x position
    assert tip_pos[1] != 0.0  # y position
    # Allow for some z tolerance
    assert tip_pos[2] > task.max_height - 1e-6
    assert tip_pos[2] < task.max_height + 1e-6

    ell = task.running_cost(state, jnp.zeros(2))
    assert ell.shape == ()

    phi = task.terminal_cost(state)
    assert phi.shape == ()
    assert phi >= 0.0


if __name__ == "__main__":
    test_double_pendulum()
