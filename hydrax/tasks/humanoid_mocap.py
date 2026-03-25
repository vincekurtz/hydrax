from dataclasses import dataclass
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from huggingface_hub import hf_hub_download
from mujoco import mjx
from mujoco.mjx._src.math import quat_inv, quat_mul, quat_sub, rotate

from hydrax import ROOT
from hydrax.task_base import Task


def _yaw_quat(q: jax.Array) -> jax.Array:
    """Extract the yaw (z-axis rotation) component of a quaternion.

    Args:
        q: Quaternion in (w, x, y, z) format. Shape is (4,).

    Returns:
        A unit quaternion representing only the yaw rotation. Shape is (4,).
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    yaw = jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half_yaw = yaw * 0.5
    return jnp.array([jnp.cos(half_yaw), 0.0, 0.0, jnp.sin(half_yaw)])


@dataclass
class HumanoidMocapOptions:
    """Configuration options for the HumanoidMocap task.

    Reward terms and weights match the BeyondMimic implementation in mjlab:
    https://github.com/mujocolab/mjlab/blob/main/src/mjlab/tasks/tracking/tracking_env_cfg.py

    The running cost is the negative of the total BeyondMimic reward.
    """

    # --- Anchor (root) body ---
    anchor_body_name: str = "pelvis"

    # --- Reward weights and standard deviations ---

    # Global anchor (root) position tracking
    anchor_pos_weight: float = 0.5
    anchor_pos_std: float = 0.3

    # Global anchor (root) orientation tracking
    anchor_ori_weight: float = 0.5
    anchor_ori_std: float = 0.4

    # Relative body position tracking (relative to anchor frame)
    body_pos_weight: float = 1.0
    body_pos_std: float = 0.3

    # Relative body orientation tracking (relative to anchor frame)
    body_ori_weight: float = 1.0
    body_ori_std: float = 0.4

    # Global body linear velocity tracking
    body_lin_vel_weight: float = 1.0
    body_lin_vel_std: float = 1.0

    # Global body angular velocity tracking
    body_ang_vel_weight: float = 1.0
    body_ang_vel_std: float = 3.14

    # List of body names to track.
    tracked_bodies: Tuple[str, ...] = (
        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    )

    # --- Domain randomization ranges ---

    # Contact friction: uniform range for geom_friction[:, 0]
    geom_friction_range: Tuple[float, float] = (0.3, 1.6)

    # Contact time constant (geom_solref[:, 0]); MuJoCo default is 0.02
    geom_solref_range: Tuple[float, float] = (0.01, 0.04)

    # Contact margin (geom_margin); MuJoCo default is 0.0
    geom_margin_range: Tuple[float, float] = (0.0, 0.005)

    # Body mass: multiplicative scale drawn from [1-scale, 1+scale]
    body_mass_scale: float = 0.2

    # Center-of-mass position: additive noise drawn from [-offset, +offset] (m)
    body_ipos_offset: float = 0.005

    # Joint damping for actuated DOFs: uniform range (N·m·s/rad)
    dof_damping_range: Tuple[float, float] = (0.0, 5.0)

    # Joint friction loss for actuated DOFs: uniform range (N·m)
    dof_frictionloss_range: Tuple[float, float] = (0.0, 1.0)

    # Actuator kP / kD gains: multiplicative scale drawn from [1-scale, 1+scale]
    actuator_gain_scale: float = 0.2


class HumanoidMocap(Task):
    """The Unitree G1 humanoid tracks a reference from motion capture.

    Retargeted motion capture data comes from the LocoMuJoCo dataset:
    https://huggingface.co/datasets/robfiras/loco-mujoco-datasets/tree/main.
    """

    def __init__(
        self,
        reference_filename: str = "Lafan1/mocap/UnitreeG1/walk1_subject1.npz",
        impl: str = "jax",
        options: HumanoidMocapOptions | None = None,
    ) -> None:
        """Load the MuJoCo model and set task parameters.

        The list of available reference files can be found at
        https://huggingface.co/datasets/robfiras/loco-mujoco-datasets/tree/main.

        Args:
            reference_filename: The name of the reference mocap file to track.
            impl: Backend to use for simulation rollouts ("jax" or "warp").
            options: Task options controlling cost weights and domain
                     randomization ranges.
        """
        if options is None:
            options = HumanoidMocapOptions()
        self.options = options

        mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/g1/scene.xml")
        super().__init__(
            mj_model,
            trace_sites=["imu_in_torso", "left_foot", "right_foot"],
            impl=impl,
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
        self.reference_qpos = reference
        self.reference_fps = npz_file["frequency"]

        # Precompute the pose and velocity of each body throughout the
        # reference trajectory.
        mj_data = mujoco.MjData(mj_model)
        n_frames = len(reference)
        reference_xpos = np.zeros((n_frames - 1, mj_model.nbody, 3))
        reference_xquat = np.zeros((n_frames - 1, mj_model.nbody, 4))
        reference_body_lin_vel = np.zeros((n_frames - 1, mj_model.nbody, 3))
        reference_body_ang_vel = np.zeros((n_frames - 1, mj_model.nbody, 3))
        for i in range(n_frames - 1):
            mj_data.qpos[:] = reference[i]

            # Set qvel = (reference[i+1] - reference[i]) / dt, converting
            # quaternions appropriately.
            dt = 1.0 / self.reference_fps
            mujoco.mj_differentiatePos(
                mj_model, mj_data.qvel, dt, reference[i], reference[i + 1]
            )

            # Forward kinematics to get body poses etc.
            mujoco.mj_forward(mj_model, mj_data)

            reference_xpos[i] = mj_data.xpos
            reference_xquat[i] = mj_data.xquat

            # cvel has shape (nbody, 6): [:3] angular, [3:] linear,
            # expressed in the world frame.
            reference_body_ang_vel[i] = mj_data.cvel[:, :3]
            reference_body_lin_vel[i] = mj_data.cvel[:, 3:]

        # Anchor (root) body index for global tracking
        self.anchor_body_index = mj_data.body(options.anchor_body_name).id

        # Tracked body indices for body-level tracking
        self.tracked_body_indices = jnp.array(
            [mj_data.body(body_name).id for body_name in options.tracked_bodies]
        )

        # Convert reference data to jax arrays, indexed by tracked bodies
        bodies = [mj_data.body(name).id for name in options.tracked_bodies]
        self.reference_xpos = jnp.array(reference_xpos[:, bodies])
        self.reference_xquat = jnp.array(reference_xquat[:, bodies])
        self.reference_body_lin_vel = jnp.array(
            reference_body_lin_vel[:, bodies]
        )
        self.reference_body_ang_vel = jnp.array(
            reference_body_ang_vel[:, bodies]
        )

        # Anchor body reference data (global frame)
        anchor_id = self.anchor_body_index
        self.reference_anchor_pos = jnp.array(reference_xpos[:, anchor_id])
        self.reference_anchor_quat = jnp.array(reference_xquat[:, anchor_id])

        # Store reward weights and std values
        self.anchor_pos_weight = options.anchor_pos_weight
        self.anchor_pos_std = options.anchor_pos_std
        self.anchor_ori_weight = options.anchor_ori_weight
        self.anchor_ori_std = options.anchor_ori_std
        self.body_pos_weight = options.body_pos_weight
        self.body_pos_std = options.body_pos_std
        self.body_ori_weight = options.body_ori_weight
        self.body_ori_std = options.body_ori_std
        self.body_lin_vel_weight = options.body_lin_vel_weight
        self.body_lin_vel_std = options.body_lin_vel_std
        self.body_ang_vel_weight = options.body_ang_vel_weight
        self.body_ang_vel_std = options.body_ang_vel_std

    def _get_time_index(self, t: jax.Array) -> jax.Array:
        """Get the clipped frame index for time t."""
        i = jnp.int32(t * self.reference_fps)
        return jnp.clip(i, 0, self.reference_anchor_pos.shape[0] - 1)

    def _get_reference_anchor_pose(
        self, t: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Get the reference anchor body position and orientation at time t."""
        i = self._get_time_index(t)
        return self.reference_anchor_pos[i], self.reference_anchor_quat[i]

    def _get_reference_body_poses(
        self, t: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Get reference tracked body positions and orientations at time t."""
        i = self._get_time_index(t)
        return self.reference_xpos[i], self.reference_xquat[i]

    def _get_reference_body_velocities(
        self, t: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Get reference body linear and angular velocities at time t."""
        i = self._get_time_index(t)
        return self.reference_body_lin_vel[i], self.reference_body_ang_vel[i]

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ).

        This is the negative of the BeyondMimic reward, consisting of:
          - Global anchor position and orientation tracking
          - Anchor-relative body position and orientation tracking
          - Global body linear and angular velocity tracking

        See https://github.com/mujocolab/mjlab/blob/main/src/mjlab/tasks/tracking/
        """
        t = state.time

        # --- Reference data ---
        ref_anchor_pos, ref_anchor_quat = self._get_reference_anchor_pose(t)
        ref_body_pos, ref_body_quat = self._get_reference_body_poses(t)
        ref_body_lin_vel, ref_body_ang_vel = (
            self._get_reference_body_velocities(t)
        )

        # --- Robot data ---
        anchor_idx = self.anchor_body_index
        robot_anchor_pos = state.xpos[anchor_idx]
        robot_anchor_quat = state.xquat[anchor_idx]

        bodies = self.tracked_body_indices
        robot_body_pos = state.xpos[bodies]
        robot_body_quat = state.xquat[bodies]
        # cvel: (nbody, 6) where [:3]=angular, [3:]=linear in world frame
        robot_body_lin_vel = state.cvel[bodies, 3:]
        robot_body_ang_vel = state.cvel[bodies, :3]

        # =================================================================
        # 1. Global anchor position reward:
        #    exp(-||ref_anchor_pos - robot_anchor_pos||^2 / std^2)
        # =================================================================
        anchor_pos_err = jnp.sum(jnp.square(ref_anchor_pos - robot_anchor_pos))
        r_anchor_pos = jnp.exp(-anchor_pos_err / self.anchor_pos_std**2)

        # =================================================================
        # 2. Global anchor orientation reward:
        #    exp(-angle(ref_anchor_quat, robot_anchor_quat)^2 / std^2)
        #    quat_sub returns axis*angle (3D), whose squared norm = angle^2
        # =================================================================
        anchor_ori_err = jnp.sum(
            jnp.square(quat_sub(ref_anchor_quat, robot_anchor_quat))
        )
        r_anchor_ori = jnp.exp(-anchor_ori_err / self.anchor_ori_std**2)

        # =================================================================
        # 3. Relative body position reward.
        #    Reference body positions are transformed so the reference anchor
        #    aligns with the robot's anchor (xy + yaw). Heights stay in the
        #    reference frame.
        #
        #    delta_pos = robot_anchor_pos with z replaced by ref_anchor z
        #    delta_ori = yaw_only(robot_anchor_quat * inv(ref_anchor_quat))
        #    body_pos_relative = delta_pos + rotate(body_pos - anchor_pos,
        #                                           delta_ori)
        # =================================================================
        delta_pos = robot_anchor_pos.at[2].set(ref_anchor_pos[2])
        delta_ori = _yaw_quat(
            quat_mul(robot_anchor_quat, quat_inv(ref_anchor_quat))
        )

        ref_body_pos_relative = delta_pos[None, :] + jax.vmap(
            lambda p: rotate(p - ref_anchor_pos, delta_ori)
        )(ref_body_pos)

        body_pos_err = jnp.mean(
            jnp.sum(jnp.square(ref_body_pos_relative - robot_body_pos), axis=-1)
        )
        r_body_pos = jnp.exp(-body_pos_err / self.body_pos_std**2)

        # =================================================================
        # 4. Relative body orientation reward.
        #    body_quat_relative = delta_ori * ref_body_quat
        #    error = mean over bodies of angle^2
        # =================================================================
        ref_body_quat_relative = jax.vmap(lambda q: quat_mul(delta_ori, q))(
            ref_body_quat
        )

        body_ori_err = jnp.mean(
            jax.vmap(lambda q1, q2: jnp.sum(jnp.square(quat_sub(q1, q2))))(
                ref_body_quat_relative, robot_body_quat
            )
        )
        r_body_ori = jnp.exp(-body_ori_err / self.body_ori_std**2)

        # =================================================================
        # 5. Global body linear velocity reward:
        #    exp(-mean_b(||ref_lin_vel_b - robot_lin_vel_b||^2) / std^2)
        # =================================================================
        body_lin_vel_err = jnp.mean(
            jnp.sum(jnp.square(ref_body_lin_vel - robot_body_lin_vel), axis=-1)
        )
        r_body_lin_vel = jnp.exp(-body_lin_vel_err / self.body_lin_vel_std**2)

        # =================================================================
        # 6. Global body angular velocity reward:
        #    exp(-mean_b(||ref_ang_vel_b - robot_ang_vel_b||^2) / std^2)
        # =================================================================
        body_ang_vel_err = jnp.mean(
            jnp.sum(jnp.square(ref_body_ang_vel - robot_body_ang_vel), axis=-1)
        )
        r_body_ang_vel = jnp.exp(-body_ang_vel_err / self.body_ang_vel_std**2)

        # =================================================================
        # Total reward and cost
        # =================================================================
        total_reward = (
            self.anchor_pos_weight * r_anchor_pos
            + self.anchor_ori_weight * r_anchor_ori
            + self.body_pos_weight * r_body_pos
            + self.body_ori_weight * r_body_ori
            + self.body_lin_vel_weight * r_body_lin_vel
            + self.body_ang_vel_weight * r_body_ang_vel
        )

        return -total_reward

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        return self.running_cost(state, jnp.zeros(self.model.nu)) * self.dt

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomize physical and contact modeling parameters."""
        opts = self.options
        rng, friction_rng, stiffness_rng, margin_rng = jax.random.split(rng, 4)
        rng, mass_rng, ipos_rng, damping_rng, fric_rng, kp_rng, kd_rng = (
            jax.random.split(rng, 7)
        )

        # Friction coefficients (via geom_friction)
        n_geoms = self.model.geom_friction.shape[0]
        geom_friction = self.model.geom_friction.at[:, 0].set(
            jax.random.uniform(
                friction_rng,
                (n_geoms,),
                minval=opts.geom_friction_range[0],
                maxval=opts.geom_friction_range[1],
            )
        )

        # Contact stiffness (via geom_solref). We'll modify the time constant
        # in particular (mujoco default is 0.02).
        n_geoms = self.model.geom_solref.shape[0]
        geom_solref = self.model.geom_solref.at[:, 0].set(
            jax.random.uniform(
                stiffness_rng,
                (n_geoms,),
                minval=opts.geom_solref_range[0],
                maxval=opts.geom_solref_range[1],
            )
        )

        # Contact margin (distance at which contact forces activate. Default is
        # zero.)
        n_geoms = self.model.geom_margin.shape[0]
        geom_margin = self.model.geom_margin.at[:].set(
            jax.random.uniform(
                margin_rng,
                (n_geoms,),
                minval=opts.geom_margin_range[0],
                maxval=opts.geom_margin_range[1],
            )
        )

        # Body masses: multiplicative noise ±body_mass_scale
        n_bodies = self.model.body_mass.shape[0]
        mass_scale = jax.random.uniform(
            mass_rng,
            (n_bodies,),
            minval=1.0 - opts.body_mass_scale,
            maxval=1.0 + opts.body_mass_scale,
        )
        body_mass = self.model.body_mass * mass_scale

        # Center of mass positions: additive noise ±body_ipos_offset per axis
        body_ipos = self.model.body_ipos + jax.random.uniform(
            ipos_rng,
            self.model.body_ipos.shape,
            minval=-opts.body_ipos_offset,
            maxval=opts.body_ipos_offset,
        )

        # Joint damping for actuated DOFs.
        # The first 6 DOFs belong to the free root joint and are left at 0.
        n_dof = self.model.dof_damping.shape[0]
        dof_damping = self.model.dof_damping.at[6:].set(
            jax.random.uniform(
                damping_rng,
                (n_dof - 6,),
                minval=opts.dof_damping_range[0],
                maxval=opts.dof_damping_range[1],
            )
        )

        # Joint friction loss for actuated DOFs.
        dof_frictionloss = self.model.dof_frictionloss.at[6:].set(
            jax.random.uniform(
                fric_rng,
                (n_dof - 6,),
                minval=opts.dof_frictionloss_range[0],
                maxval=opts.dof_frictionloss_range[1],
            )
        )

        # Actuator kP gains: multiplicative noise ±actuator_gain_scale.
        # gainprm[:, 0] = kP; biasprm[:, 1] = -kP (must stay consistent).
        n_act = self.model.actuator_gainprm.shape[0]
        kp_scale = jax.random.uniform(
            kp_rng,
            (n_act,),
            minval=1.0 - opts.actuator_gain_scale,
            maxval=1.0 + opts.actuator_gain_scale,
        )
        kp = self.model.actuator_gainprm[:, 0] * kp_scale
        actuator_gainprm = self.model.actuator_gainprm.at[:, 0].set(kp)
        actuator_biasprm = self.model.actuator_biasprm.at[:, 1].set(-kp)

        # Actuator kD gains: multiplicative noise ±actuator_gain_scale.
        kd_scale = jax.random.uniform(
            kd_rng,
            (n_act,),
            minval=1.0 - opts.actuator_gain_scale,
            maxval=1.0 + opts.actuator_gain_scale,
        )
        actuator_biasprm = actuator_biasprm.at[:, 2].set(
            self.model.actuator_biasprm[:, 2] * kd_scale
        )

        return {
            "geom_friction": geom_friction,
            "geom_solref": geom_solref,
            "geom_margin": geom_margin,
            "body_mass": body_mass,
            "body_ipos": body_ipos,
            "dof_damping": dof_damping,
            "dof_frictionloss": dof_frictionloss,
            "actuator_gainprm": actuator_gainprm,
            "actuator_biasprm": actuator_biasprm,
        }

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
        return super().make_data(naconmax=50000, njmax=200)
