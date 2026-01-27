from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class QuadrupedStanding(Task):
    """The Unitree Go2 quadruped maintains a standing position.

    This task encourages the robot to stand upright at a target height
    with minimal movement and control effort.
    """

    def __init__(self) -> None:
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/unitree_go2/mjx_scene_position_collision.xml"
        )
        super().__init__(
            mj_model,
            trace_sites=["imu", "FL_foot", "FR_foot", "RL_foot", "RR_foot"],
        )

        # Get sensor and site IDs for efficient access
        self.orientation_sensor_id = mj_model.sensor("orientation").id
        self.imu_site_id = mj_model.site("imu").id

        # Standing configuration from keyframe
        self.qstand = jnp.array(mj_model.keyframe("home").qpos)

        # Target height for the base (from keyframe)
        self.target_height = 0.27

        """
        print(self.u_min)
        print(self.u_max)
        """

    def _get_base_height(self, state: mjx.Data) -> jax.Array:
        """Get the height of the base above the ground."""
        return state.site_xpos[self.imu_site_id, 2]

    def _get_base_orientation(self, state: mjx.Data) -> jax.Array:
        """Get the rotation from the current base orientation to upright."""
        sensor_adr = self.model.sensor_adr[self.orientation_sensor_id]
        quat = state.sensordata[sensor_adr : sensor_adr + 4]
        upright = jnp.array([0.0, 0.0, 1.0])
        return mjx._src.math.rotate(upright, quat)

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        # Orientation cost: penalize deviation from upright
        orientation_cost = jnp.sum(
            jnp.square(self._get_base_orientation(state))
        )

        # Height cost: penalize deviation from target height
        height_cost = jnp.square(
            self._get_base_height(state) - self.target_height
        )

        # Nominal configuration cost: penalize joint deviations from standing pose
        nominal_cost = jnp.sum(jnp.square(state.qpos[7:] - self.qstand[7:]))

        # Control effort cost: penalize large control inputs
        control_cost = jnp.sum(jnp.square(control))

        return (
            1.0 * orientation_cost
            + 1.0 * height_cost
            + 1 * nominal_cost
            + 0.01 * control_cost
        )

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        return self.running_cost(state, jnp.zeros(self.model.nu))

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
        """Randomly perturb the measured base position and velocities."""
        rng, q_rng, v_rng = jax.random.split(rng, 3)
        q_err = 0.01 * jax.random.normal(q_rng, (7,))
        v_err = 0.01 * jax.random.normal(v_rng, (6,))

        qpos = data.qpos.at[0:7].set(data.qpos[0:7] + q_err)
        qvel = data.qvel.at[0:6].set(data.qvel[0:6] + v_err)

        return {"qpos": qpos, "qvel": qvel}
