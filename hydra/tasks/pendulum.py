import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydra import ROOT
from hydra.base import Task


class Pendulum(Task):
    """An inverted pendulum swingup task."""

    def __init__(self, planning_horizon: int = 20):
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/pendulum/scene.xml"
        )

        sim_steps_per_control_step = 5
        self.dt = mj_model.opt.timestep * sim_steps_per_control_step

        super().__init__(
            mjx.put_model(mj_model),
            planning_horizon,
            sim_steps_per_control_step,
        )

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
        return self.dt * total_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        theta_cost = self._distance_to_upright(state)
        theta_dot_cost = 0.01 * jnp.square(state.qvel[0])
        return theta_cost + theta_dot_cost
