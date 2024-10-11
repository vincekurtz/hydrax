import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class LeapHand(Task):
    """Cube rotation with the LEAP hand."""

    def __init__(self):
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/leap/scene.xml")

        super().__init__(
            mj_model,
            planning_horizon=4,
            sim_steps_per_control_step=3,
            u_max=jnp.inf,
            trace_sites=[],
        )

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        return 0.0

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        return jnp.sum(jnp.square(state.qvel))
