import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class DoublePendulum(Task):
    """A double inverted pendulum swingup task."""

    def __init__(self):
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/double_pendulum/scene.xml"
        )

        super().__init__(
            mj_model,
            trace_sites=["tip"],
        )

        self.tip_id = mj_model.site("tip").id
        self.max_height = 1.0  # Desired height of tip

    def _height_cost(self, state: mjx.Data) -> jax.Array:
        """Cost based on the height of the tip."""
        return jnp.square(state.site_xpos[self.tip_id, 2] - self.max_height)

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        velocity_cost = jnp.sum(jnp.square(state.qvel))
        control_cost = jnp.sum(jnp.square(control))
        return (
            1e2 * self._height_cost(state)
            + 1e-1 * velocity_cost
            + 1e-2 * control_cost
        )

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        vel_cost = jnp.sum(jnp.square(state.qvel))
        return 1e2 * self._height_cost(state) + 1e-1 * vel_cost
