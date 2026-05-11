"""Velocity-controlled point mass that must escape a U-shaped maze.

The task is identical in dynamics to ``hydrax.tasks.particle.Particle`` but
adds three inner walls forming a U around the start position. This creates
a local minimum on the straight-line path to the goal that defeats most
local samplers, providing a clean showcase for globally-exploratory
controllers such as MTP.
"""

from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class Navigation(Task):
    """A planar point mass that must navigate around U-shaped inner walls."""

    def __init__(self, impl: str = "jax") -> None:
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/particle_navigation/scene.xml"
        )
        super().__init__(mj_model, trace_sites=["pointmass"], impl=impl)

        self.pointmass_id = mj_model.site("pointmass").id

        # Start the particle behind the U-opening so the straight-line path
        # to the goal goes through a wall.
        self._initial_qpos = jnp.array([-0.2, 0.0])

        # Cache the planar geometry of the three inner walls (wall_ix,
        # wall_iy, wall_neg_iy) for the SDF-style cost term.
        self._wall_pos = jnp.array(
            [
                mj_model.geom("wall_ix").pos[:2],
                mj_model.geom("wall_iy").pos[:2],
                mj_model.geom("wall_neg_iy").pos[:2],
            ]
        )
        self._wall_size = jnp.array(
            [
                mj_model.geom("wall_ix").size[:2],
                mj_model.geom("wall_iy").size[:2],
                mj_model.geom("wall_neg_iy").size[:2],
            ]
        )

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ): wall SDF + tracking + control."""
        # Box-SDF distance to each inner wall (signed, axis-aligned).
        pos = state.site_xpos[self.pointmass_id][None, :2]
        wall_dist = jnp.abs(pos - self._wall_pos) - self._wall_size
        outside_dist = jnp.maximum(wall_dist, 0.0)
        inside_dist = jnp.minimum(jnp.max(wall_dist, axis=-1), 0.0)
        dist = (
            jnp.sqrt(jnp.sum(jnp.square(outside_dist), axis=-1) + 1e-12)
            + inside_dist
        ).min(axis=-1)
        wall_cost = 5.0 * jnp.exp(-50.0 * dist)
        control_cost = jnp.sum(jnp.square(control))
        return wall_cost + self.terminal_cost(state) + 0.1 * control_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T): position tracking + velocity reg."""
        position_cost = jnp.sum(
            jnp.square(state.site_xpos[self.pointmass_id] - state.mocap_pos[0])
        )
        velocity_cost = jnp.sum(jnp.square(state.qvel))
        return 5.0 * position_cost + 0.1 * velocity_cost

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomly perturb the actuator gains by ±10%."""
        multiplier = jax.random.uniform(
            rng,
            self.model.actuator_gainprm[:, 0].shape,
            minval=0.9,
            maxval=1.1,
        )
        new_gains = self.model.actuator_gainprm[:, 0] * multiplier
        new_gains = self.model.actuator_gainprm.at[:, 0].set(new_gains)
        return {"actuator_gainprm": new_gains}

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        """Randomly shift the measured particle position."""
        shift = jax.random.uniform(rng, (2,), minval=-0.01, maxval=0.01)
        return {"qpos": data.qpos + shift}
