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

    def __init__(self, reference_filename: str = "walk1_subject1.npz") -> None:
        """Load the MuJoCo model and set task parameters.

        The list of available reference files can be found at
        https://huggingface.co/datasets/robfiras/loco-mujoco-datasets/tree/main.
        """
        mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/g1/scene.xml")
        super().__init__(
            mj_model,
            trace_sites=["imu_in_torso", "left_foot", "right_foot"],
        )

        # Download the retargeted mocap reference
        npz_file = np.load(
            hf_hub_download(
                repo_id="robfiras/loco-mujoco-datasets",
                filename=reference_filename,
                subfolder="Lafan1/mocap/UnitreeG1",
                repo_type="dataset",
            )
        )

        reference = npz_file["qpos"]
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
        # N.B. we multiply by dt to ensure the terminal cost is comparable to
        # the running cost, since this isn't a proper cost-to-go.
        return self.dt * jnp.sum(jnp.square(q_err))

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
