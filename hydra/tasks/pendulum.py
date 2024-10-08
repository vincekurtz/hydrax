import jax
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

        super().__init__(
            mjx.put_model(mj_model),
            planning_horizon,
            sim_steps_per_control_step,
        )

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ) penalizes large torques."""
        pass

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T) penalizes distance from the upright."""
        pass
