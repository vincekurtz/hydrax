import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class Crane(Task):
    """A luffing crane moves a payload to a target position."""

    def __init__(
        self, planning_horizon: int = 2, sim_steps_per_control_step: int = 40
    ):
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/crane/scene.xml"
        )

        super().__init__(
            mj_model,
            planning_horizon=planning_horizon,
            sim_steps_per_control_step=sim_steps_per_control_step,
            trace_sites=["payload_end"],
        )

        self.payload_pos_sensor_adr = mj_model.sensor_adr[
            mj_model.sensor("payload_pos").id
        ]
        self.payload_vel_sensor_adr = mj_model.sensor_adr[
            mj_model.sensor("payload_vel").id
        ]

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ) encourages payload tracking."""
        # Get the position and velocity of the payload relative to the target
        payload_pos = state.sensordata[
            self.payload_pos_sensor_adr : self.payload_pos_sensor_adr + 3
        ]
        payload_vel = state.sensordata[
            self.payload_vel_sensor_adr : self.payload_vel_sensor_adr + 3
        ]

        # Compute cost terms
        position_cost = jnp.sum(jnp.square(payload_pos))
        velocity_cost = jnp.sum(jnp.square(payload_vel))
        return position_cost + 0.1 * velocity_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """Terminal cost is the same as running cost."""
        return self.running_cost(state, jnp.zeros(self.model.nu))
