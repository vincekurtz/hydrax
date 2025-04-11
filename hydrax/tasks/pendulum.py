import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class Pendulum(Task):
    """An inverted pendulum swingup task."""

    def __init__(self) -> None:
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/pendulum/scene.xml"
        )
        super().__init__(mj_model, trace_sites=["tip"])

    def _distance_to_upright(self, state: mjx.Data) -> jax.Array:
        """Get a measure of distance to the upright position."""
        theta = state.qpos[0] - jnp.pi
        theta_err = jnp.array([jnp.cos(theta) - 1, jnp.sin(theta)])
        return jnp.sum(jnp.square(theta_err))

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        theta_cost = self._distance_to_upright(state)
        theta_dot_cost = 0.01 * jnp.square(state.qvel[0])
        control_cost = 0.001 * jnp.sum(jnp.square(control))
        total_cost = theta_cost + theta_dot_cost + control_cost
        return total_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        theta_cost = self._distance_to_upright(state)
        theta_dot_cost = 0.01 * jnp.square(state.qvel[0])
        return theta_cost + theta_dot_cost
