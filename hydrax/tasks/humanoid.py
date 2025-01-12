from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class Humanoid(Task):
    """Standup task for the Unitree G1 humanoid."""

    def __init__(
        self, planning_horizon: int = 3, sim_steps_per_control_step: int = 10
    ):
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/g1/scene.xml")

        super().__init__(
            mj_model,
            planning_horizon=planning_horizon,
            sim_steps_per_control_step=sim_steps_per_control_step,
            trace_sites=["imu", "left_toe", "right_toe"],
        )

        # Get site ids
        self.torso_id = mj_model.site("imu").id

        # Get sensor addresses
        self.ori_adr = mj_model.sensor_adr[mj_model.sensor("imu-body-quat").id]
        self.com_adr = mj_model.sensor_adr[mj_model.sensor("com").id]
        self.com_vel_adr = mj_model.sensor_adr[mj_model.sensor("com_vel").id]
        self.left_toe_adr = mj_model.sensor_adr[mj_model.sensor("left_toe").id]
        self.right_toe_adr = mj_model.sensor_adr[
            mj_model.sensor("right_toe").id
        ]
        self.left_heel_adr = mj_model.sensor_adr[
            mj_model.sensor("left_heel").id
        ]
        self.right_heel_adr = mj_model.sensor_adr[
            mj_model.sensor("right_heel").id
        ]

        # Set the target height
        self.target_height = 0.9

        # Standing configuration
        self.qstand = jnp.array(mj_model.keyframe("stand").qpos)

    def _average_foot_position(self, state: mjx.Data) -> jax.Array:
        """Get the average foot position in the x/y plane."""
        # x, y positions of the toes and heels of the left and right feet
        lt = state.sensordata[self.left_toe_adr : self.left_toe_adr + 2]
        rt = state.sensordata[self.right_toe_adr : self.right_toe_adr + 2]
        lh = state.sensordata[self.left_heel_adr : self.left_heel_adr + 2]
        rh = state.sensordata[self.right_heel_adr : self.right_heel_adr + 2]

        # Average foot position in the world frame
        return 0.25 * (lt + rt + lh + rh)

    def _get_capture_point(self, state: mjx.Data) -> jax.Array:
        """Approximately where we should step to avoid falling."""
        # Position of the center-of-mass
        com_pos_xy = state.sensordata[self.com_adr : self.com_adr + 2]

        # Velocity of the center-of-mass
        com_vel_xy = state.sensordata[self.com_vel_adr : self.com_vel_adr + 2]

        # Capture point dynamics
        omega = jnp.sqrt(9.81 / self.target_height)
        return com_pos_xy + com_vel_xy / omega

    def _get_torso_height(self, state: mjx.Data) -> jax.Array:
        """Get the height of the torso above the feet."""
        torso_height = state.site_xpos[self.torso_id, 2]

        # Foot heights
        lt_height = state.sensordata[self.left_toe_adr + 2]
        rt_height = state.sensordata[self.right_toe_adr + 2]
        lh_height = state.sensordata[self.left_heel_adr + 2]
        rh_height = state.sensordata[self.right_heel_adr + 2]
        avg_foot_height = 0.25 * (lt_height + rt_height + lh_height + rh_height)

        return torso_height - avg_foot_height

    def _get_torso_orientation(self, state: mjx.Data) -> jax.Array:
        """Get the rotation from the current torso orientation to upright."""
        quat = state.sensordata[self.ori_adr : self.ori_adr + 4]
        upright = jnp.array([0.0, 0.0, 1.0])
        return mjx._src.math.rotate(upright, quat)

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        orientation_cost = jnp.sum(
            jnp.square(self._get_torso_orientation(state))
        )
        height_cost = jnp.square(
            self._get_torso_height(state) - self.target_height
        )
        balance_cost = jnp.sum(
            jnp.square(
                self._average_foot_position(state)
                - self._get_capture_point(state)
            )
        )
        control_cost = jnp.sum(jnp.square(control))
        nominal_cost = jnp.sum(jnp.square(state.qpos[7:] - self.qstand[7:]))
        velocity_cost = jnp.sum(jnp.square(state.qvel[0:6]))
        return (
            1.00 * orientation_cost
            + 100.00 * height_cost
            + 10.00 * balance_cost
            + 1.0 * nominal_cost
            + 0.10 * velocity_cost
            + 0.01 * control_cost
        )

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        return 10 * self.running_cost(state, jnp.zeros(self.model.nu))

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
