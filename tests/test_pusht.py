import jax
import jax.numpy as jnp
from mujoco import mjx

from hydrax.tasks.pusht import PushT


def test_task() -> None:
    """Set up the push T task."""
    task = PushT()

    state = mjx.make_data(task.model)
    assert isinstance(state, mjx.Data)
    state = state.replace(mocap_quat=jnp.array([[0.0, 1.0, 0.0, 0.0]]))
    state = jax.jit(mjx.forward)(task.model, state)

    pos = task._get_position_err(state)
    assert pos.shape == (3,)

    ori = task._get_orientation_err(state)
    assert ori.shape == (3,)

    ell = task.running_cost(state, jnp.zeros(2))
    assert ell.shape == ()

    phi = task.terminal_cost(state)
    assert phi.shape == ()


if __name__ == "__main__":
    test_task()
