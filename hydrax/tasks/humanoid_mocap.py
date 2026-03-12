from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from huggingface_hub import hf_hub_download
from mujoco import mjx

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

        # Track positions from all mocap points.
        self.tracked_sensor_sites = (
            ("left_toe_position", "left_toe"),
            ("right_toe_position", "right_toe"),
            ("left_heel_position", "left_heel"),
            ("right_heel_position", "right_heel"),
            ("left_knee_position", "left_knee"),
            ("right_knee_position", "right_knee"),
            ("pelvis_position", "pelvis_site"),
            ("torso_position", "torso_site"),
            ("head_position", "head_site"),
            ("left_shoulder_position", "left_shoulder"),
            ("right_shoulder_position", "right_shoulder"),
            ("left_elbow_position", "left_elbow"),
            ("right_elbow_position", "right_elbow"),
            ("left_thumb_position", "left_thumb"),
            ("right_thumb_position", "right_thumb"),
            ("left_pink_position", "left_pink"),
            ("right_pink_position", "right_pink"),
        )

        sensor_ids = [
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
            for sensor_name, _ in self.tracked_sensor_sites
        ]
        self.position_sensor_adrs = tuple(
            int(mj_model.sensor_adr[sensor_id]) for sensor_id in sensor_ids
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

        # Precompute reference site positions for all tracked sensors.
        mj_data = mujoco.MjData(mj_model)
        site_ids = [
            mj_model.site(site_name).id
            for _, site_name in self.tracked_sensor_sites
        ]
        n_frames = len(reference)
        ref_positions = np.zeros((n_frames, len(site_ids), 3))
        for i in range(n_frames):
            mj_data.qpos[:] = reference[i]
            mujoco.mj_forward(mj_model, mj_data)
            ref_positions[i] = mj_data.site_xpos[site_ids]

        self.ref_sensor_positions = jnp.array(ref_positions)

    def _get_reference_positions(self, t: jax.Array) -> jax.Array:
        """Get reference positions for all tracked sensors at time t."""
        i = jnp.int32(t * self.reference_fps)
        i = jnp.clip(i, 0, self.ref_sensor_positions.shape[0] - 1)
        return self.ref_sensor_positions[i]

    def _get_position_errors(self, state: mjx.Data) -> jax.Array:
        """Get position tracking errors for all tracked sensors."""
        measured_positions = jnp.stack(
            [
                state.sensordata[adr : adr + 3]
                for adr in self.position_sensor_adrs
            ]
        )
        return measured_positions - self._get_reference_positions(state.time)

    def _position_tracking_cost(self, state: mjx.Data) -> jax.Array:
        """Compute the sum of squared position errors across tracked sensors."""
        position_errors = self._get_position_errors(state)
        return jnp.sum(jnp.square(position_errors))

    def running_cost(self, state: mjx.Data, _control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        return self._position_tracking_cost(state)

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        return self.dt * self._position_tracking_cost(state)

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
