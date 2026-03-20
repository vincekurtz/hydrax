from dataclasses import dataclass
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from huggingface_hub import hf_hub_download
from mujoco import mjx
from mujoco.mjx._src.math import quat_sub

from hydrax import ROOT
from hydrax.task_base import Task


@dataclass
class HumanoidMocapOptions:
    """Configuration options for the HumanoidMocap task."""

    # --- Cost weights ---

    # Generalized position (qpos) tracking
    configuration_cost_weight: float = 0.1

    # Generalized velocity (qvel) tracking
    generalized_velocity_cost_weight: float = 0.0

    # Body position tracking (xpos)
    body_position_cost_weight: float = 1.0

    # Body orientation tracking (xquat)
    body_orientation_cost_weight: float = 0.0

    # Body linear and angular velocity tracking (cvel)
    body_twist_cost_weight: float = 0.0

    # List of body names to track
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
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/g1/scene.xml"
        )
        super().__init__(
            mj_model,
            trace_sites=["imu_in_torso", "left_foot", "right_foot"],
            impl=impl,
        )

        # Download and load reference data
        reference = np.loadtxt(
            hf_hub_download(
                repo_id="lvhaidong/LAFAN1_Retargeting_Dataset",
                filename=reference_filename,
                repo_type="dataset",
            ),
            delimiter=",",
        )
        self.reference_fps = 30

        # Fix the reference to use mujoco format with wxyz quaternions.
        pos = reference[:, :3]
        quat_xyzw = reference[:, 3:7]
        quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
        reference = np.concatenate([pos, quat_wxyz, reference[:, 7:]], axis=1)

        # Precompute the pose of each body throughout the reference trajectory.
        mj_data = mujoco.MjData(mj_model)
        n_frames = len(reference)
        reference_xpos = np.zeros((n_frames - 1, mj_model.nbody, 3))
        reference_xquat = np.zeros((n_frames - 1, mj_model.nbody, 4))
        reference_qvel = np.zeros((n_frames - 1, mj_model.nv))
        reference_cvel = np.zeros((n_frames - 1, mj_model.nbody, 6))
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
            reference_qvel[i] = mj_data.qvel

            # N.B. this reference body velocity is expressed as a spatial
            # velocity in the frame of the root of the kinematic tree. This is
            # a bit funny, and we should consider using mj_objectVelocity
            # instead. But for now these quantities are readily available.
            reference_cvel[i] = mj_data.cvel

        # Precompute the indices of the tracked bodies for fast lookup during
        # cost computation.
        self.tracked_body_indices = jnp.array(
            [mj_data.body(body_name).id for body_name in options.tracked_bodies]
        )

        # Convert reference data to jax arrays
        self.reference_qpos = jnp.array(reference[0:-1])
        self.reference_qvel = jnp.array(reference_qvel)
        self.reference_xpos = jnp.array(reference_xpos)
        self.reference_xquat = jnp.array(reference_xquat)
        self.reference_cvel = jnp.array(reference_cvel)

        if options is None:
            options = HumanoidMocapOptions()
        self.options = options

        # Weigh different cost terms, then normalize so all cost terms add to 1.
        total_weights = (
            options.configuration_cost_weight
            + options.generalized_velocity_cost_weight
            + options.body_position_cost_weight
            + options.body_orientation_cost_weight
            + options.body_twist_cost_weight
        )
        self.configuration_cost_weight = (
            options.configuration_cost_weight / total_weights
        )
        self.generalized_velocity_cost_weight = (
            options.generalized_velocity_cost_weight / total_weights
        )
        self.body_position_cost_weight = (
            options.body_position_cost_weight / total_weights
        )
        self.body_orientation_cost_weight = (
            options.body_orientation_cost_weight / total_weights
        )
        self.body_twist_cost_weight = (
            options.body_twist_cost_weight / total_weights
        )

    def _get_reference_configuration(self, t: jax.Array) -> jax.Array:
        """Get the reference position (q) at time t."""
        i = jnp.int32(t * self.reference_fps)
        i = jnp.clip(i, 0, self.reference_qpos.shape[0] - 1)
        return self.reference_qpos[i, :]

    def _get_reference_velocity(self, t: jax.Array) -> jax.Array:
        """Get the reference velocity (v) at time t."""
        i = jnp.int32(t * self.reference_fps)
        i = jnp.clip(i, 0, self.reference_qvel.shape[0] - 1)
        return self.reference_qvel[i, :]

    def _get_reference_body_poses(
        self, t: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Get the reference body positions and orientations at time t."""
        i = jnp.int32(t * self.reference_fps)
        i = jnp.clip(i, 0, self.reference_xpos.shape[0] - 1)
        bodies = self.tracked_body_indices
        return self.reference_xpos[i, bodies], self.reference_xquat[i, bodies]

    def _get_reference_body_twists(self, t: jax.Array) -> jax.Array:
        """Get the reference body linear and angular velocities at time t."""
        i = jnp.int32(t * self.reference_fps)
        i = jnp.clip(i, 0, self.reference_cvel.shape[0] - 1)
        bodies = self.tracked_body_indices
        return self.reference_cvel[i, bodies]

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        # Joint angle tracking error
        q_ref = self._get_reference_configuration(state.time)
        q = state.qpos
        q_err = q - q_ref  # size (nq,)

        # Body pose tracking error
        ref_xpos, ref_xquat = self._get_reference_body_poses(state.time)
        xpos_err = state.xpos[self.tracked_body_indices] - ref_xpos
        xquat_err = jax.vmap(quat_sub)(
            state.xquat[self.tracked_body_indices], ref_xquat
        )

        # Generalized velocity tracking error
        v_ref = self._get_reference_velocity(state.time)
        v = state.qvel
        v_err = v - v_ref  # size (nv,)

        # Body twist tracking error
        ref_cvel = self._get_reference_body_twists(state.time)
        cvel_err = state.cvel[self.tracked_body_indices] - ref_cvel

        # Tracking costs J = 1 - exp(-|error|^2) for each error term. This puts
        # each error term between 0 and 1.
        q_squared_error = jnp.sum(jnp.square(q_err))
        q_cost = 1.0 - jnp.exp(-q_squared_error)

        v_squared_error = jnp.sum(jnp.square(v_err))
        v_cost = 1.0 - jnp.exp(-v_squared_error)

        xpos_squared_error = jnp.sum(jnp.square(xpos_err))
        xpos_cost = 1.0 - jnp.exp(-xpos_squared_error)

        xquat_squared_error = jnp.sum(jnp.square(xquat_err))
        xquat_cost = 1.0 - jnp.exp(-xquat_squared_error)

        cvel_squared_error = jnp.sum(jnp.square(cvel_err))
        cvel_cost = 1.0 - jnp.exp(-cvel_squared_error)

        # Weighted sum of the different error terms. Weights are normalized so
        # the total cost is between 0 and 1.
        total_cost = (
            self.configuration_cost_weight * q_cost
            + self.body_position_cost_weight * xpos_cost
            + self.body_orientation_cost_weight * xquat_cost
            + self.generalized_velocity_cost_weight * v_cost
            + self.body_twist_cost_weight * cvel_cost
        )

        return total_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        # We'll use the same cost as the running costs. Multiplied by
        # the time step dt ensures that the terminal cost is weighed equally
        # with the running costs. In other words, this terminal cost should not
        # be interpreted as an optimal cost-to-go.
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
        return super().make_data(naconmax=20000, njmax=200)
