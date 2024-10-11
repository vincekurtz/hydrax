import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class CubeRotation(Task):
    """Cube rotation with the LEAP hand."""

    def __init__(self):
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/cube/scene.xml")

        super().__init__(
            mj_model,
            planning_horizon=5,
            sim_steps_per_control_step=4,
            u_max=1,
            trace_sites=[],
        )

        # Get sensor ids
        self.cube_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "cube_position"
        )
        self.cube_orientation_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "cube_orientation"
        )

    def _get_cube_position(self, state: mjx.Data) -> jax.Array:
        """Position of the cube relative to the target grasp position."""
        sensor_adr = self.model.sensor_adr[self.cube_position_sensor]
        return state.sensordata[sensor_adr : sensor_adr + 3]

    def _get_cube_orientation_err(self, state: mjx.Data) -> jax.Array:
        """Orientation of the cube relative to the target grasp orientation."""
        sensor_adr = self.model.sensor_adr[self.cube_orientation_sensor]
        cube_quat = state.sensordata[sensor_adr : sensor_adr + 4]
        # TODO: implement target cube
        goal_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
        return mjx._src.math.quat_sub(cube_quat, goal_quat)

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        position_err = self._get_cube_position(state)
        orientation_err = self._get_cube_orientation_err(state)
        return jnp.sum(jnp.square(position_err)) + jnp.sum(
            jnp.square(orientation_err)
        )

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        position_err = self._get_cube_position(state)
        orientation_err = self._get_cube_orientation_err(state)
        return jnp.sum(jnp.square(position_err)) + jnp.sum(
            jnp.square(orientation_err)
        )
