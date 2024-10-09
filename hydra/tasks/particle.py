import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydra import ROOT
from hydra.base import Task


class Particle(Task):
    """A velocity-controlled planar point mass chases a target position."""

    def __init__(self):
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/particle/scene.xml"
        )

        super().__init__(
            mj_model,
            planning_horizon=10,
            sim_steps_per_control_step=5,
            u_max=1.0,
            trace_sites=["pointmass"],
        )

        self.pointmass_id = mj_model.site("pointmass").id

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ) encourages target tracking."""
        state_cost = self.terminal_cost(state)
        control_cost = jnp.sum(jnp.square(control))
        return state_cost + 0.1 * control_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        position_cost = jnp.sum(
            jnp.square(state.site_xpos[self.pointmass_id] - state.mocap_pos[0])
        )
        velocity_cost = jnp.sum(jnp.square(state.qvel))
        return 5.0 * position_cost + 0.1 * velocity_cost
