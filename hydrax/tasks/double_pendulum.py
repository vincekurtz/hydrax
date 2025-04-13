import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class DoublePendulum(Task):
    """A double inverted pendulum swingup task."""

    def __init__(
        self,
        planning_horizon: int = 10,
        sim_steps_per_control_step: int = 8,
    ):
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/double_pendulum/scene.xml"
        )

        super().__init__(
            mj_model,
            planning_horizon=planning_horizon,
            sim_steps_per_control_step=sim_steps_per_control_step,
            trace_sites=["tip"],
        )

        self.tip_id = mj_model.site("tip").id

    def _height_cost(self, state: mjx.Data) -> jax.Array:
        """Cost based on the height of the tip."""
        return jnp.square(state.site_xpos[self.tip_id, 2] - 1.1)

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        height_cost = self._height_cost(state)
        # Penalize velocity
        vel_cost = 0.01 * (
            jnp.square(state.qvel[0]) + jnp.square(state.qvel[1])
        )
        # Penalize control effort
        control_cost = 0.001 * jnp.sum(jnp.square(control))
        return 10 * height_cost + vel_cost + control_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        # Primarily care about final height and low velocity
        height_cost = self._height_cost(state)
        vel_cost = 0.01 * (
            jnp.square(state.qvel[0]) + jnp.square(state.qvel[1])
        )
        # Higher weight on terminal height cost
        return 100.0 * height_cost + vel_cost
