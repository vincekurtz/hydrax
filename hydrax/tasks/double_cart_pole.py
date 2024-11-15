import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class DoubleCartPole(Task):
    """A swing-up task for a double pendulum on a cart."""

    def __init__(
        self, planning_horizon: int = 10, sim_steps_per_control_step: int = 10
    ):
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/double_cart_pole/scene.xml"
        )

        super().__init__(
            mj_model,
            planning_horizon=planning_horizon,
            sim_steps_per_control_step=sim_steps_per_control_step,
            trace_sites=["tip"],
        )

        self.tip_id = mj_model.site("tip").id

    def _distance_to_upright(self, state: mjx.Data) -> jax.Array:
        """Get a measure of distance to the upright position."""
        tip_z = state.site_xpos[self.tip_id, 2]
        return jnp.square(tip_z - 4.0)

    def _distance_to_centered(self, state: mjx.Data) -> jax.Array:
        """Distance to center is measured at the tip, not the cart."""
        tip_x = state.site_xpos[self.tip_id, 0]
        return jnp.square(tip_x)

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        theta_cost = self._distance_to_upright(state)
        centering_cost = self._distance_to_centered(state)
        velocity_cost = 0.1 * jnp.sum(jnp.square(state.qvel[1:]))
        control_cost = 0.01 * jnp.sum(jnp.square(control))
        return theta_cost + centering_cost + velocity_cost + control_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        theta_cost = 10 * self._distance_to_upright(state)
        centering_cost = 10 * self._distance_to_centered(state)
        velocity_cost = 0.1 * jnp.sum(jnp.square(state.qvel[1:]))
        return theta_cost + centering_cost + velocity_cost
