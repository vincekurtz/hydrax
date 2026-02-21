from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class Crane(Task):
    """A luffing crane moves a payload to a target position."""

    def __init__(self, impl: str = "jax") -> None:
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/crane/scene.xml"
        )
        super().__init__(mj_model, trace_sites=["payload_end"], impl=impl)

        self.payload_pos_sensor_adr = mj_model.sensor_adr[
            mj_model.sensor("payload_pos").id
        ]
        self.payload_vel_sensor_adr = mj_model.sensor_adr[
            mj_model.sensor("payload_vel").id
        ]
        self.payload_idx = mj_model.body("payload").id

    def _get_payload_position(self, state: mjx.Data) -> jax.Array:
        """Get the position of the payload relative to the target."""
        return state.sensordata[
            self.payload_pos_sensor_adr : self.payload_pos_sensor_adr + 3
        ]

    def _get_payload_velocity(self, state: mjx.Data) -> jax.Array:
        """Get the velocity of the payload."""
        return state.sensordata[
            self.payload_vel_sensor_adr : self.payload_vel_sensor_adr + 3
        ]

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ) encourages payload tracking."""
        # Get the position and velocity of the payload relative to the target
        payload_pos = self._get_payload_position(state)
        payload_vel = self._get_payload_velocity(state)

        # Compute cost terms
        position_cost = jnp.sum(jnp.square(payload_pos))
        velocity_cost = jnp.sum(jnp.square(payload_vel))
        return position_cost + 0.1 * velocity_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """Terminal cost is the same as running cost."""
        return self.running_cost(state, jnp.zeros(self.model.nu))

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomize various model parameters."""
        rng, damping_rng, mass_rng, actuator_rng = jax.random.split(rng, 4)

        # Randomize joint damping
        damping_multiplier = jax.random.uniform(
            damping_rng, (1,), minval=0.05, maxval=20.0
        )
        new_damping = self.model.dof_damping * damping_multiplier

        # Randomize payload mass and inertia
        mass_multiplier = jax.random.uniform(
            mass_rng, (), minval=0.5, maxval=2.0
        )
        new_mass = self.model.body_mass.at[self.payload_idx].set(
            self.model.body_mass[self.payload_idx] * mass_multiplier
        )
        new_inertia = self.model.body_inertia.at[self.payload_idx].set(
            self.model.body_inertia[self.payload_idx] * mass_multiplier
        )

        # Randomize actuator gains
        # Adopted from the MJX tutorial:
        # https://github.com/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb
        new_gain = (
            jax.random.uniform(actuator_rng, (1,), minval=-5, maxval=5)
            + self.model.actuator_gainprm[:, 0]
        )
        new_actuator_gainprm = self.model.actuator_gainprm.at[:, 0].set(
            new_gain
        )
        new_actuator_biasprm = self.model.actuator_biasprm.at[:, 1].set(
            -new_gain
        )

        return {
            "body_mass": new_mass,
            "body_inertia": new_inertia,
            "dof_damping": new_damping,
            "actuator_gainprm": new_actuator_gainprm,
            "actuator_biasprm": new_actuator_biasprm,
        }

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        """Add noise to the state estimate."""
        rng, q_rng, v_rng = jax.random.split(rng, 3)
        q_err = 0.01 * jax.random.normal(q_rng, (self.model.nq,))
        v_err = 0.01 * jax.random.normal(v_rng, (self.model.nv,))

        return {"qpos": data.qpos + q_err, "qvel": data.qvel + v_err}
