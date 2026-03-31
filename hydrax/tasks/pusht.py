from dataclasses import dataclass
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


@dataclass
class PushTOptions:
    """Configuration options for the PushT task."""

    # --- Initial state ranges ---

    # Block spawn: center (x, y) and radius for uniform disk sampling, orientation range
    init_block_center: Tuple[float, float] = (0.1, 0.1)
    init_block_radius: float = 0.1
    init_block_ori_range: Tuple[float, float] = (-3.14, 3.14)

    # Pusher spawn: fixed position (x, y) in world frame
    init_pusher_pos: Tuple[float, float] = (0.0, 0.0)

    # --- Cost weights ---

    # Block position tracking weight
    position_weight: float = 2.0

    # Block orientation tracking weight
    orientation_weight: float = 1.0

    # Pusher proximity to block weight
    close_to_block_weight: float = 0.01

    # Pusher velocity regularization weight
    control_weight: float = 0.01

    # --- Domain randomization ranges ---

    # Contact friction: absolute range for geom_friction[:, 0]
    geom_friction_range: Tuple[float, float] = (0.5, 1.5)

    # Contact time constant (geom_solref[:, 0]); MuJoCo default is 0.02
    geom_solref_range: Tuple[float, float] = (0.01, 0.03)

    # Body mass: multiplicative scale range for all bodies
    body_mass_range: Tuple[float, float] = (0.8, 1.2)

    # Actuator kv gain: multiplicative scale range
    actuator_kv_range: Tuple[float, float] = (0.8, 1.2)


class PushT(Task):
    """Push a T-shaped block to a desired pose."""

    def __init__(
        self,
        impl: str = "jax",
        options: PushTOptions | None = None,
    ) -> None:
        """Load the MuJoCo model and set task parameters.

        Args:
            impl: Backend to use for simulation rollouts ("jax" or "warp").
            options: Task options controlling cost weights and domain
                     randomization ranges.
        """
        if options is None:
            options = PushTOptions()
        self.options = options

        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/pusht/scene.xml"
        )
        super().__init__(mj_model, trace_sites=["pusher"], impl=impl)

        # Get sensor ids
        self.block_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "position"
        )
        self.block_orientation_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "orientation"
        )

    def _get_position_err(self, state: mjx.Data) -> jax.Array:
        """Position of the block relative to the target position."""
        sensor_adr = self.model.sensor_adr[self.block_position_sensor]
        return state.sensordata[sensor_adr : sensor_adr + 3]

    def _get_orientation_err(self, state: mjx.Data) -> jax.Array:
        """Orientation of the block relative to the target orientation."""
        sensor_adr = self.model.sensor_adr[self.block_orientation_sensor]
        block_quat = state.sensordata[sensor_adr : sensor_adr + 4]
        goal_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
        return mjx._src.math.quat_sub(block_quat, goal_quat)

    def _close_to_block_err(self, state: mjx.Data) -> jax.Array:
        """Position of the pusher block relative to the block."""
        block_pos = state.qpos[:2]
        pusher_pos = state.qpos[3:] + jnp.array([0.0, 0.1])  # y bias
        return block_pos - pusher_pos

    def _get_pusher_velocity(self, state: mjx.Data) -> jax.Array:
        """Velocity of the pusher (root_x, root_y joints)."""
        return state.qvel[3:]

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        position_err = self._get_position_err(state)
        orientation_err = self._get_orientation_err(state)
        close_to_block_err = self._close_to_block_err(state)
        pusher_vel = self._get_pusher_velocity(state)

        position_cost = jnp.sum(jnp.square(position_err))
        orientation_cost = jnp.sum(jnp.square(orientation_err))
        close_to_block_cost = jnp.sum(jnp.square(close_to_block_err))
        control_cost = jnp.sum(jnp.square(pusher_vel))

        opts = self.options
        return (
              opts.position_weight * position_cost
            + opts.orientation_weight * orientation_cost
            + opts.close_to_block_weight * close_to_block_cost
            + opts.control_weight * control_cost
        )

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ℓ_T(x_T)."""
        return self.running_cost(state, jnp.zeros(self.model.nu))

    def compute_metrics(self, state: mjx.Data, control: jax.Array) -> dict:
        """Compute individual cost terms for logging.

        Returns a dictionary of scalar cost terms:
            position_cost, orientation_cost, close_to_block_cost, total_cost
        """
        position_err = self._get_position_err(state)
        orientation_err = self._get_orientation_err(state)
        close_to_block_err = self._close_to_block_err(state)
        pusher_vel = self._get_pusher_velocity(state)

        position_cost = jnp.sum(jnp.square(position_err))
        orientation_cost = jnp.sum(jnp.square(orientation_err))
        close_to_block_cost = jnp.sum(jnp.square(close_to_block_err))
        control_cost = jnp.sum(jnp.square(pusher_vel))
        opts = self.options
        total_cost = (
            opts.position_weight * position_cost
            + opts.orientation_weight * orientation_cost
            + opts.close_to_block_weight * close_to_block_cost
            + opts.control_weight * control_cost
        )

        return {
            "position_cost": position_cost,
            "orientation_cost": orientation_cost,
            "close_to_block_cost": close_to_block_cost,
            "control_cost": control_cost,
            "total_cost": total_cost,
        }

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomize friction, contact parameters, body masses, and kv."""
        opts = self.options
        rng, friction_rng, solref_rng, mass_rng, kv_rng = (
            jax.random.split(rng, 5)
        )

        # Friction coefficient (tangential)
        n_geoms = self.model.geom_friction.shape[0]
        geom_friction = self.model.geom_friction.at[:, 0].set(
            jax.random.uniform(
                friction_rng,
                (n_geoms,),
                minval=opts.geom_friction_range[0],
                maxval=opts.geom_friction_range[1],
            )
        )

        # Contact stiffness (via geom_solref time constant)
        geom_solref = self.model.geom_solref.at[:, 0].set(
            jax.random.uniform(
                solref_rng,
                (n_geoms,),
                minval=opts.geom_solref_range[0],
                maxval=opts.geom_solref_range[1],
            )
        )

        # Body mass: multiplicative noise (all bodies)
        n_bodies = self.model.body_mass.shape[0]
        mass_scales = jax.random.uniform(
            mass_rng,
            (n_bodies,),
            minval=opts.body_mass_range[0],
            maxval=opts.body_mass_range[1],
        )
        body_mass = self.model.body_mass * mass_scales

        # Actuator kv: multiplicative noise
        # gainprm[:, 0] = kv; biasprm[:, 2] = -kv
        n_act = self.model.actuator_gainprm.shape[0]
        kv_scales = jax.random.uniform(
            kv_rng,
            (n_act,),
            minval=opts.actuator_kv_range[0],
            maxval=opts.actuator_kv_range[1],
        )
        kv = self.model.actuator_gainprm[:, 0] * kv_scales
        actuator_gainprm = self.model.actuator_gainprm.at[:, 0].set(kv)
        actuator_biasprm = self.model.actuator_biasprm.at[:, 2].set(-kv)

        return {
            "geom_friction": geom_friction,
            "geom_solref": geom_solref,
            "body_mass": body_mass,
            "actuator_gainprm": actuator_gainprm,
            "actuator_biasprm": actuator_biasprm,
        }

    def sample_initial_position(self, rng: jax.Array) -> jax.Array:
        """Sample a random initial qpos from the configured ranges.

        Block position is sampled uniformly within a disk. Pusher is fixed.
        qpos layout: [block_x, block_y, block_theta, pusher_x, pusher_y]
        """
        opts = self.options
        rng, angle_rng, radius_rng, theta_rng = jax.random.split(rng, 4)

        # Block: uniform sample in a disk
        angle = jax.random.uniform(
            angle_rng, (), minval=0.0, maxval=2.0 * jnp.pi
        )
        r = opts.init_block_radius * jnp.sqrt(
            jax.random.uniform(radius_rng, (), minval=0.0, maxval=1.0)
        )
        center = jnp.array(opts.init_block_center)
        block_xy = center + r * jnp.array([jnp.cos(angle), jnp.sin(angle)])

        block_theta = jax.random.uniform(
            theta_rng, (),
            minval=opts.init_block_ori_range[0],
            maxval=opts.init_block_ori_range[1],
        )

        # Pusher: fixed position (qpos is offset from body origin at (0, 0.1))
        pusher_world = jnp.array(opts.init_pusher_pos)
        pusher_xy = pusher_world - jnp.array([0.0, 0.1])

        return jnp.concatenate([block_xy, block_theta[None], pusher_xy])

    def make_data(self) -> mjx.Data:
        """Create a new state object with extra constraints allocated."""
        return super().make_data(nconmax=200_000, naconmax=600_000, njmax=2_000)
