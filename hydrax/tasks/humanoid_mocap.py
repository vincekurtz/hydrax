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

    Retargeted motion capture data comes from the LAFAN1 dataset:
    https://huggingface.co/datasets/unitreerobotics/LAFAN1_Retargeting_Dataset/.
    """

    def __init__(
        self,
        planning_horizon: int = 4,
        sim_steps_per_control_step: int = 5,
        reference_filename: str = "walk1_subject1.csv",
    ):
        """Load the MuJoCo model and set task parameters.

        The list of available reference files can be found at
        https://huggingface.co/datasets/unitreerobotics/LAFAN1_Retargeting_Dataset/tree/main/g1.
        """
        mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/g1/scene.xml")

        super().__init__(
            mj_model,
            planning_horizon=planning_horizon,
            sim_steps_per_control_step=sim_steps_per_control_step,
            trace_sites=["imu_in_torso", "left_foot", "right_foot"],
        )

        # Download the retargeted mocap reference
        reference = np.loadtxt(
            hf_hub_download(
                repo_id="unitreerobotics/LAFAN1_Retargeting_Dataset",
                filename=reference_filename,
                subfolder="g1",
                repo_type="dataset",
            ),
            delimiter=",",
        )

        # Convert the dataset to mujoco format, with wxyz quaternion
        pos = reference[:, :3]
        xyzw = reference[:, 3:7]
        wxyz = np.concatenate([xyzw[:, 3:], xyzw[:, :3]], axis=1)
        reference = np.concatenate([pos, wxyz, reference[:, 7:]], axis=1)
        self.reference = jnp.array(reference)

        # Cost weights
        cost_weights = np.ones(mj_model.nq)
        cost_weights[:7] = 10.0  # Base pose is more important
        self.cost_weights = jnp.array(cost_weights)

    def _get_reference_configuration(self, t: jax.Array) -> jax.Array:
        """Get the reference position (q) at time t."""
        i = jnp.int32(t * 30.0)  # reference runs at 30 FPS
        i = jnp.clip(i, 0, self.reference.shape[0] - 1)
        return self.reference[i, :]

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        # Configuration error weighs the base pose more heavily
        q_ref = self._get_reference_configuration(state.time)
        q = state.qpos
        q_err = self.cost_weights * (q - q_ref)
        configuration_cost = jnp.sum(jnp.square(q_err))

        # Control penalty incentivizes driving toward the reference, since all
        # joints are position-controlled
        u_ref = q_ref[7:]
        control_cost = jnp.sum(jnp.square(control - u_ref))

        return 1.0 * configuration_cost + 1.0 * control_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        q_ref = self._get_reference_configuration(state.time)
        q = state.qpos
        q_err = self.cost_weights * (q - q_ref)
        return 0.0 * jnp.sum(jnp.square(q_err))

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
