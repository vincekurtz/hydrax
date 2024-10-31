from typing import Dict

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
            planning_horizon=3,
            sim_steps_per_control_step=4,
            trace_sites=["cube_center", "if_tip", "mf_tip", "rf_tip", "th_tip"],
        )

        # Get sensor ids
        self.cube_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "cube_position"
        )
        self.cube_orientation_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "cube_orientation"
        )

        # Distance (m) beyond which we impose a high cube position cost
        self.delta = 0.015

    def _get_cube_position_err(self, state: mjx.Data) -> jax.Array:
        """Position of the cube relative to the target grasp position."""
        sensor_adr = self.model.sensor_adr[self.cube_position_sensor]
        return state.sensordata[sensor_adr : sensor_adr + 3]

    def _get_cube_orientation_err(self, state: mjx.Data) -> jax.Array:
        """Orientation of the cube relative to the target grasp orientation."""
        sensor_adr = self.model.sensor_adr[self.cube_orientation_sensor]
        cube_quat = state.sensordata[sensor_adr : sensor_adr + 4]

        # Quaternion subtraction gives us rotation relative to goal
        goal_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
        return mjx._src.math.quat_sub(cube_quat, goal_quat)

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        position_err = self._get_cube_position_err(state)
        squared_distance = jnp.sum(jnp.square(position_err[0:2]))  # ignore z
        position_cost = 0.1 * squared_distance + 100 * jnp.maximum(
            squared_distance - self.delta**2, 0.0
        )

        orientation_err = self._get_cube_orientation_err(state)
        orientation_cost = jnp.sum(jnp.square(orientation_err))

        grasp_cost = 0.001 * jnp.sum(jnp.square(control))

        return position_cost + orientation_cost + grasp_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        position_err = self._get_cube_position_err(state)
        return 100 * jnp.sum(jnp.square(position_err))

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomize the friction parameters."""
        n_geoms = self.model.geom_friction.shape[0]
        multiplier = jax.random.uniform(rng, (n_geoms,), minval=0.5, maxval=2.0)
        new_frictions = self.model.geom_friction.at[:, 0].set(
            self.model.geom_friction[:, 0] * multiplier
        )
        return {"geom_friction": new_frictions}

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        """Randomly shift the measured configurations."""
        shift = 0.005 * jax.random.normal(rng, (self.model.nq,))
        return {"qpos": data.qpos + shift}

    def get_obs(self, state: mjx.Data) -> jax.Array:
        """Observe the hand position, cube pose relative to target, and vels."""
        hand_qpos = state.qpos[7:]
        cube_position_err = self._get_cube_position_err(state)
        cube_orientation_err = self._get_cube_orientation_err(state)
        return jnp.concatenate(
            [cube_position_err, cube_orientation_err, hand_qpos, state.qvel]
        )
