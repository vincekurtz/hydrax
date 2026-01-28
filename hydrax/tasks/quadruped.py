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
            ROOT + "/models/unitree_go2/mjx_scene_position.xml"
        )
        super().__init__(
            mj_model,
            trace_sites=["imu", "FL_foot", "FR_foot", "RL_foot", "RR_foot"],
        )

        # Target standing configuration from keyframe
        # qpos: [x, y, z, qw, qx, qy, qz, joint_positions...]
        # Using the "home" keyframe as reference: 0 0 0.27 1 0 0 0 0 0.9 -1.8 (repeated for 4 legs)
        self.target_height = 0.27  # Target height for the base
        self.target_orientation = jnp.array([1.0, 0.0, 0.0, 0.0])  # Upright (qw, qx, qy, qz)

        # Target joint positions for standing (from the home keyframe)
        # FL, RL, FR, RR legs: hip, thigh, calf = [0, 0.9, -1.8] for each
        self.target_qpos = jnp.array([
            0.0, 0.9, -1.8,  # FL (Front Left)
            0.0, 0.9, -1.8,  # RL (Rear Left)
            0.0, 0.9, -1.8,  # FR (Front Right)
            0.0, 0.9, -1.8,  # RR (Rear Right)
        ])

        # Cost weights for different components
        self.height_weight = 10.0
        self.orientation_weight = 0.0
        self.joint_pos_weight = 1.0
        self.velocity_weight = 0.1
        self.control_weight = 0.01

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        # Height error: penalize deviation from target height
        current_height = state.qpos[2]
        height_cost = self.height_weight * jnp.square(current_height - self.target_height)

        # Orientation error: penalize deviation from upright orientation
        # Quaternion: qpos[3:7] = [qw, qx, qy, qz]
        current_orientation = state.qpos[3:7]
        # Compute quaternion difference (simple squared error)
        orientation_error = current_orientation - self.target_orientation
        orientation_cost = self.orientation_weight * jnp.sum(jnp.square(orientation_error))

        # Joint position error: penalize deviation from target joint positions
        # Joint positions start at qpos[7:]
        current_joint_pos = state.qpos[7:]
        joint_pos_error = current_joint_pos - self.target_qpos
        joint_pos_cost = self.joint_pos_weight * jnp.sum(jnp.square(joint_pos_error))

        # Velocity cost: penalize movement
        velocity_cost = self.velocity_weight * jnp.sum(jnp.square(state.qvel))

        # Control cost: penalize large control inputs
        control_error = control - self.target_qpos
        control_cost = self.control_weight * jnp.sum(jnp.square(control_error))

        return (
            height_cost
            + orientation_cost
            + joint_pos_cost
            + velocity_cost
            + control_cost
        )

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        # Height error
        current_height = state.qpos[2]
        height_cost = self.height_weight * jnp.square(current_height - self.target_height)

        # Orientation error
        current_orientation = state.qpos[3:7]
        orientation_error = current_orientation - self.target_orientation
        orientation_cost = self.orientation_weight * jnp.sum(jnp.square(orientation_error))

        # Joint position error
        current_joint_pos = state.qpos[7:]
        joint_pos_error = current_joint_pos - self.target_qpos
        joint_pos_cost = self.joint_pos_weight * jnp.sum(jnp.square(joint_pos_error))

        # Velocity cost
        velocity_cost = self.velocity_weight * jnp.sum(jnp.square(state.qvel))

        return self.dt * (
            height_cost
            + orientation_cost
            + joint_pos_cost
            + velocity_cost
        )

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
