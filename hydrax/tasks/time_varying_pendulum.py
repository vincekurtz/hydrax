import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class TimeVaryingPendulum(Task):
    """A time-varying pendulum task, where the pendulum tracks reference."""

    def __init__(
        self, planning_horizon: int = 20, sim_steps_per_control_step: int = 5
    ):
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/pendulum/scene.xml"
        )

        super().__init__(
            mj_model,
            planning_horizon=planning_horizon,
            sim_steps_per_control_step=sim_steps_per_control_step,
            trace_sites=["tip"],
        )

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        t = state.time
        theta_nom = 0.5 * jnp.sin(2 * jnp.pi * t / 4)
        theta_cost = 0.5 * jnp.square(state.qpos[0] - theta_nom)
        theta_dot_cost = 0.00 * jnp.square(state.qvel[0])
        control_cost = 0.000 * jnp.sum(jnp.square(control))
        total_cost = theta_cost + theta_dot_cost + control_cost
        return total_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        return self.running_cost(state, jnp.zeros(1))
