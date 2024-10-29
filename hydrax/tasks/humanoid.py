import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class Humanoid(Task):
    """Locomotion with the Unitree G1 humanoid."""

    def __init__(self):
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/g1/scene.xml")

        super().__init__(
            mj_model,
            planning_horizon=2,
            sim_steps_per_control_step=50,
            trace_sites=["imu", "left_foot", "right_foot"],
        )

        # Get sensor and site ids
        self.orientation_sensor_id = mj_model.sensor("imu-body-quat").id
        self.velocity_sensor_id = mj_model.sensor("imu-body-linvel").id
        self.torso_id = mj_model.site("imu").id

        # Set the target velocity (m/s) and height
        self.target_velocity = 0.0
        self.target_height = 0.9

    def _get_torso_height(self, state: mjx.Data) -> jax.Array:
        """Get the height of the torso above the ground."""
        return state.site_xpos[self.torso_id, 2]

    def _get_torso_orientation(self, state: mjx.Data) -> jax.Array:
        """Get the rotation from the current torso orientation to upright."""
        sensor_adr = self.model.sensor_adr[self.orientation_sensor_id]
        quat = state.sensordata[sensor_adr : sensor_adr + 4]
        goal_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
        return mjx._src.math.quat_sub(quat, goal_quat)

    def _get_torso_velocity(self, state: mjx.Data) -> jax.Array:
        """Get the horizontal velocity of the torso."""
        sensor_adr = self.model.sensor_adr[self.velocity_sensor_id]
        return state.sensordata[sensor_adr]

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        orientation_cost = jnp.sum(
            jnp.square(self._get_torso_orientation(state))
        )
        velocity_cost = jnp.square(
            self._get_torso_velocity(state) - self.target_velocity
        )
        height_cost = jnp.square(
            self._get_torso_height(state) - self.target_height
        )
        control_cost = jnp.sum(jnp.square(control))
        return (
            1.0 * orientation_cost
            + 0.1 * velocity_cost
            + 1.0 * height_cost
            + 0.5 * control_cost
        )

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        orientation_cost = jnp.sum(
            jnp.square(self._get_torso_orientation(state))
        )
        height_cost = jnp.square(
            self._get_torso_height(state) - self.target_height
        )
        return orientation_cost + 10 * height_cost
