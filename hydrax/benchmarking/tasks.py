"""Registry of tasks and per-task random initial-condition samplers.

Each entry specifies the task factory, control horizon settings, episode
length for closed-loop runs, and a function that, given a JAX PRNG key,
returns numpy arrays for (qpos, qvel, mocap_pos). Following the user's
spec, the per-trial seed is the *only* source of randomness: it determines
both the initial condition and the algorithm RNG, so seed → IC is a
one-to-one mapping.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from hydrax.task_base import Task
from hydrax.tasks.cart_pole import CartPole
from hydrax.tasks.cube import CubeRotation
from hydrax.tasks.double_cart_pole import DoubleCartPole
from hydrax.tasks.humanoid_standup import HumanoidStandup
from hydrax.tasks.particle import Particle
from hydrax.tasks.pendulum import Pendulum
from hydrax.tasks.pusht import PushT
from hydrax.tasks.walker import Walker

InitialState = Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]
ICSampler = Callable[[jax.Array], InitialState]


@dataclass
class TaskSpec:
    """Per-task benchmark configuration."""

    factory: Callable[[], Task]
    ic_sampler: ICSampler
    plan_horizon: float
    num_knots: int
    spline_type: str
    control_frequency: float  # Hz; controller replans at this rate
    episode_length: float  # seconds, for closed-loop benchmark


# --- IC samplers ---------------------------------------------------------


def _pendulum_ic(rng: jax.Array) -> InitialState:
    """Near the downward equilibrium (hard swing-up case).

    qpos=0 is hanging straight down; the goal is qpos=pi (upright).
    Small uniform perturbations keep seeds distinct while keeping all
    trials in the hard region.
    """
    rng_q, rng_v = jax.random.split(rng)
    qpos = jax.random.uniform(rng_q, (1,), minval=-0.1, maxval=0.1)
    qvel = jax.random.uniform(rng_v, (1,), minval=-0.1, maxval=0.1)
    return np.array(qpos), np.array(qvel), None


def _cart_pole_ic(rng: jax.Array) -> InitialState:
    """Random cart position and pole angle."""
    rng_q, rng_v = jax.random.split(rng)
    qpos = jax.random.uniform(
        rng_q,
        (2,),
        minval=jnp.array([-0.5, -jnp.pi]),
        maxval=jnp.array([0.5, jnp.pi]),
    )
    qvel = jax.random.uniform(rng_v, (2,), minval=-0.2, maxval=0.2)
    return np.array(qpos), np.array(qvel), None


def _double_cart_pole_ic(rng: jax.Array) -> InitialState:
    """Random cart position and two pole angles."""
    rng_q, rng_v = jax.random.split(rng)
    qpos = jax.random.uniform(
        rng_q,
        (3,),
        minval=jnp.array([-0.3, -jnp.pi, -jnp.pi]),
        maxval=jnp.array([0.3, jnp.pi, jnp.pi]),
    )
    qvel = jax.random.uniform(rng_v, (3,), minval=-0.2, maxval=0.2)
    return np.array(qpos), np.array(qvel), None


def _particle_ic(rng: jax.Array) -> InitialState:
    """Particle starts near origin; target (mocap) placed at a random offset."""
    rng_q, rng_v, rng_m = jax.random.split(rng, 3)
    qpos = jax.random.uniform(rng_q, (2,), minval=-0.1, maxval=0.1)
    qvel = jnp.zeros(2)
    # Target in a meaningful range for the particle scene
    target_xy = jax.random.uniform(rng_m, (2,), minval=-0.3, maxval=0.3)
    mocap_pos = jnp.concatenate([target_xy, jnp.array([0.0])])[None, :]
    return np.array(qpos), np.array(qvel), np.array(mocap_pos)


def _pusht_ic(rng: jax.Array) -> InitialState:
    """Random block pose; pusher starts away from the block."""
    rng_b, rng_p = jax.random.split(rng)
    block_xy = jax.random.uniform(rng_b, (2,), minval=-0.3, maxval=0.3)
    block_theta = jax.random.uniform(rng_b, (1,), minval=-jnp.pi, maxval=jnp.pi)
    pusher_xy = jax.random.uniform(rng_p, (2,), minval=-0.3, maxval=0.3)
    qpos = jnp.concatenate([block_xy, block_theta, pusher_xy])
    qvel = jnp.zeros(5)
    return np.array(qpos), np.array(qvel), None


def _walker_ic(rng: jax.Array) -> InitialState:
    """Nominal walker pose with small joint perturbation."""
    # qpos is planar walker: [base_x, base_z, base_pitch, hip_l, knee_l,
    # ankle_l, hip_r, knee_r, ankle_r] (or similar). Use small noise.
    rng_q, rng_v = jax.random.split(rng)
    qpos0 = jnp.zeros(9)
    qpos0 = qpos0.at[1].set(1.25)  # nominal torso height
    qpos = qpos0 + 0.05 * jax.random.normal(rng_q, (9,))
    qvel = 0.05 * jax.random.normal(rng_v, (9,))
    return np.array(qpos), np.array(qvel), None


def _cube_ic(rng: jax.Array) -> InitialState:
    """Nominal LEAP-hand grasp with small joint perturbation."""
    # Use the model's default qpos0 (from inspection above) and perturb joints.
    qpos0 = jnp.array(
        [
            -0.8,
            0.0,
            -0.8,
            -0.8,
            -0.8,
            0.0,
            -0.8,
            -0.8,
            -0.8,
            0.0,
            -0.8,
            -0.8,
            -0.8,
            -0.8,
            -0.8,
            0.0,
            0.11,
            0.0,
            0.1,  # cube position
            1.0,
            0.0,
            0.0,
            0.0,  # cube quaternion
        ]
    )
    rng_q, rng_v = jax.random.split(rng)
    noise = jnp.concatenate(
        [
            0.05 * jax.random.normal(rng_q, (16,)),  # hand joints
            jnp.zeros(3),  # cube position fixed
            jnp.zeros(4),  # cube quat fixed
        ]
    )
    qpos = qpos0 + noise
    qvel = jnp.zeros(22)
    # Leave mocap at its default (target is identity quat per cost function)
    return np.array(qpos), np.array(qvel), None


def _humanoid_standup_ic(rng: jax.Array) -> InitialState:
    """Knocked-over standing keyframe with random orientation perturbation."""
    # Build the IC from the 'stand' keyframe with a flipped base orientation,
    # matching examples/humanoid_standup.py.
    mj_model = HumanoidStandup().mj_model
    qpos = np.array(mj_model.keyframe("stand").qpos)
    qpos[3:7] = np.array([0.7, 0.0, -0.7, 0.0])  # knocked over

    # Small joint and orientation perturbations driven by the seed
    rng_q, rng_v = jax.random.split(rng)
    joint_noise = 0.05 * np.array(jax.random.normal(rng_q, (mj_model.nq - 7,)))
    quat_noise = 0.05 * np.array(jax.random.normal(rng_q, (4,)))
    qpos[7:] = qpos[7:] + joint_noise
    qpos[3:7] = qpos[3:7] + quat_noise
    qpos[3:7] = qpos[3:7] / np.linalg.norm(qpos[3:7])

    qvel = 0.05 * np.array(jax.random.normal(rng_v, (mj_model.nv,)))
    return qpos, qvel, None


# --- Registry ------------------------------------------------------------


TASKS: Dict[str, TaskSpec] = {
    "pendulum": TaskSpec(
        factory=Pendulum,
        ic_sampler=_pendulum_ic,
        plan_horizon=1.0,
        num_knots=11,
        spline_type="zero",
        control_frequency=50.0,
        episode_length=3.0,
    ),
    "cart_pole": TaskSpec(
        factory=CartPole,
        ic_sampler=_cart_pole_ic,
        plan_horizon=1.0,
        num_knots=4,
        spline_type="cubic",
        control_frequency=50.0,
        episode_length=4.0,
    ),
    "double_cart_pole": TaskSpec(
        factory=DoubleCartPole,
        ic_sampler=_double_cart_pole_ic,
        plan_horizon=1.0,
        num_knots=4,
        spline_type="cubic",
        control_frequency=50.0,
        episode_length=5.0,
    ),
    "particle": TaskSpec(
        factory=Particle,
        ic_sampler=_particle_ic,
        plan_horizon=0.25,
        num_knots=11,
        spline_type="zero",
        control_frequency=50.0,
        episode_length=2.0,
    ),
    "pusht": TaskSpec(
        factory=PushT,
        ic_sampler=_pusht_ic,
        plan_horizon=0.5,
        num_knots=6,
        spline_type="zero",
        control_frequency=50.0,
        episode_length=4.0,
    ),
    "walker": TaskSpec(
        factory=Walker,
        ic_sampler=_walker_ic,
        plan_horizon=0.6,
        num_knots=5,
        spline_type="zero",
        control_frequency=50.0,
        episode_length=3.0,
    ),
    "cube": TaskSpec(
        factory=CubeRotation,
        ic_sampler=_cube_ic,
        plan_horizon=0.25,
        num_knots=4,
        spline_type="zero",
        control_frequency=25.0,
        episode_length=3.0,
    ),
    "humanoid_standup": TaskSpec(
        factory=HumanoidStandup,
        ic_sampler=_humanoid_standup_ic,
        plan_horizon=0.6,
        num_knots=4,
        spline_type="zero",
        control_frequency=50.0,
        episode_length=3.0,
    ),
}
