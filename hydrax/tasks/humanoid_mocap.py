from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from huggingface_hub import hf_hub_download
from mujoco import mjx
from mujoco.mjx._src.math import quat_sub

from hydrax import ROOT
from hydrax.task_base import Task


class HumanoidMocap(Task):
    """The Unitree G1 humanoid tracks a reference from motion capture.

    Retargeted motion capture data comes from the LocoMuJoCo dataset:
    https://huggingface.co/datasets/robfiras/loco-mujoco-datasets/tree/main.
    """

    def __init__(
        self,
        reference_filename: str = "Lafan1/mocap/UnitreeG1/walk1_subject1.npz",
        impl: str = "jax",
    ) -> None:
        """Load the MuJoCo model and set task parameters.

        The list of available reference files can be found at
        https://huggingface.co/datasets/robfiras/loco-mujoco-datasets/tree/main.
        """
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/g1/scene_23dof.xml"
        )
        super().__init__(
            mj_model,
            trace_sites=["imu_in_torso", "left_foot", "right_foot"],
            impl=impl,
        )

        # Get sensor IDs
        self.left_foot_pos_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "left_foot_position"
        )
        self.left_foot_quat_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "left_foot_orientation"
        )
        self.right_foot_pos_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "right_foot_position"
        )
        self.right_foot_quat_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "right_foot_orientation"
        )

        # Download and load reference data
        npz_file = np.load(
            hf_hub_download(
                repo_id="robfiras/loco-mujoco-datasets",
                filename=reference_filename,
                repo_type="dataset",
            )
        )

        reference = npz_file["qpos"]
        self.reference = jnp.array(reference)
        self.reference_fps = npz_file["frequency"]

        # Precompute reference foot positions and orientations
        mj_data = mujoco.MjData(mj_model)
        n_frames = len(reference)
        ref_left_pos = np.zeros((n_frames, 3))
        ref_left_quat = np.zeros((n_frames, 4))
        ref_right_pos = np.zeros((n_frames, 3))
        ref_right_quat = np.zeros((n_frames, 4))
        for i in range(n_frames):
            mj_data.qpos[:] = reference[i]
            mujoco.mj_forward(mj_model, mj_data)
            ref_left_pos[i] = mj_data.site_xpos[mj_model.site("left_foot").id]
            ref_right_pos[i] = mj_data.site_xpos[mj_model.site("right_foot").id]
            mujoco.mju_mat2Quat(
                ref_left_quat[i],
                mj_data.site_xmat[mj_model.site("left_foot").id].flatten(),
            )
            mujoco.mju_mat2Quat(
                ref_right_quat[i],
                mj_data.site_xmat[mj_model.site("right_foot").id].flatten(),
            )

        # Convert reference data to jax arrays
        self.ref_left_pos = jnp.array(ref_left_pos)
        self.ref_left_quat = jnp.array(ref_left_quat)
        self.ref_right_pos = jnp.array(ref_right_pos)
        self.ref_right_quat = jnp.array(ref_right_quat)

        # Cost weights
        cost_weights = np.ones(mj_model.nq)
        cost_weights[:7] = 5.0  # Base pose is more important
        self.cost_weights = jnp.array(cost_weights)

    def _get_reference_configuration(self, t: jax.Array) -> jax.Array:
        """Get the reference position (q) at time t."""
        i = jnp.int32(t * self.reference_fps)
        i = jnp.clip(i, 0, self.reference.shape[0] - 1)
        return self.reference[i, :]

    def _get_reference_foot_data(self, t: jax.Array) -> tuple[jax.Array, ...]:
        """Get the reference foot positions and orientations at time t."""
        i = jnp.int32(t * self.reference_fps)
        i = jnp.clip(i, 0, self.reference.shape[0] - 1)
        return (
            self.ref_left_pos[i],
            self.ref_left_quat[i],
            self.ref_right_pos[i],
            self.ref_right_quat[i],
        )

    def _get_foot_position_errors(
        self, state: mjx.Data
    ) -> tuple[jax.Array, jax.Array]:
        """Get position errors for both feet."""
        ref_left_pos, _, ref_right_pos, _ = self._get_reference_foot_data(
            state.time
        )

        left_pos_adr = self.model.sensor_adr[self.left_foot_pos_sensor]
        right_pos_adr = self.model.sensor_adr[self.right_foot_pos_sensor]

        left_err = (
            state.sensordata[left_pos_adr : left_pos_adr + 3] - ref_left_pos
        )
        right_err = (
            state.sensordata[right_pos_adr : right_pos_adr + 3] - ref_right_pos
        )

        return left_err, right_err

    def _get_foot_orientation_errors(
        self, state: mjx.Data
    ) -> tuple[jax.Array, jax.Array]:
        """Get orientation errors for both feet."""
        _, ref_left_quat, _, ref_right_quat = self._get_reference_foot_data(
            state.time
        )

        left_quat_adr = self.model.sensor_adr[self.left_foot_quat_sensor]
        right_quat_adr = self.model.sensor_adr[self.right_foot_quat_sensor]

        left_quat = state.sensordata[left_quat_adr : left_quat_adr + 4]
        right_quat = state.sensordata[right_quat_adr : right_quat_adr + 4]

        left_err = quat_sub(left_quat, ref_left_quat)
        right_err = quat_sub(right_quat, ref_right_quat)

        return left_err, right_err

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        # Configuration error weighs the base pose more heavily
        q_ref = self._get_reference_configuration(state.time)
        q = state.qpos
        q_err = self.cost_weights * (q - q_ref)
        configuration_cost = jnp.sum(jnp.square(q_err))

        # Foot tracking costs
        left_pos_err, right_pos_err = self._get_foot_position_errors(state)
        left_ori_err, right_ori_err = self._get_foot_orientation_errors(state)

        foot_position_cost = jnp.sum(jnp.square(left_pos_err)) + jnp.sum(
            jnp.square(right_pos_err)
        )
        foot_orientation_cost = jnp.sum(jnp.square(left_ori_err)) + jnp.sum(
            jnp.square(right_ori_err)
        )

        # Control penalty
        u_ref = q_ref[7:]
        control_cost = jnp.sum(jnp.square(control - u_ref))

        return (
            1.0 * configuration_cost
            + 5.0 * foot_position_cost
            + 0.1 * foot_orientation_cost
            + 1.0 * control_cost
        )

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        q_ref = self._get_reference_configuration(state.time)
        q = state.qpos
        q_err = self.cost_weights * (q - q_ref)
        configuration_cost = jnp.sum(jnp.square(q_err))

        # Add foot tracking costs to terminal cost
        left_pos_err, right_pos_err = self._get_foot_position_errors(state)
        left_ori_err, right_ori_err = self._get_foot_orientation_errors(state)

        foot_position_cost = jnp.sum(jnp.square(left_pos_err)) + jnp.sum(
            jnp.square(right_pos_err)
        )
        foot_orientation_cost = jnp.sum(jnp.square(left_ori_err)) + jnp.sum(
            jnp.square(right_ori_err)
        )

        return self.dt * (
            1.0 * configuration_cost
            + 1.0 * foot_position_cost
            + 0.1 * foot_orientation_cost
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

    def make_data(self) -> mjx.Data:
        """Create a new state object with extra constraints allocated."""
        return super().make_data(naconmax=20000, njmax=200)
