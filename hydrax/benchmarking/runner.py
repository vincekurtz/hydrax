"""Headless runners for closed-loop MPC and open-loop trajectory opt."""

import time
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from hydrax.alg_base import SamplingBasedController
from hydrax.task_base import Task


def _make_initial_mjx_data(
    task: Task,
    qpos: np.ndarray,
    qvel: np.ndarray,
    mocap_pos: Optional[np.ndarray],
) -> mjx.Data:
    """Build an mjx.Data initialized to the given state."""
    data = task.make_data()
    fields = {"qpos": jnp.asarray(qpos), "qvel": jnp.asarray(qvel)}
    if mocap_pos is not None:
        fields["mocap_pos"] = jnp.asarray(mocap_pos)
    return data.replace(**fields)


def run_closed_loop(
    task: Task,
    controller: SamplingBasedController,
    qpos: np.ndarray,
    qvel: np.ndarray,
    mocap_pos: Optional[np.ndarray],
    control_frequency: float,
    episode_length: float,
    seed: int = 0,
) -> Tuple[float, float]:
    """Run a headless MPC simulation and return (total_cost, wall_time).

    Wall time excludes JIT compilation: one warmup call runs before timing
    begins. Returns cumulative running cost (with dt weighting) across the
    full episode.
    """
    dt = task.dt
    sim_per_ctrl = max(int(round(1.0 / control_frequency / dt)), 1)
    num_ctrl_steps = max(int(round(episode_length / dt / sim_per_ctrl)), 1)

    mjx_data = _make_initial_mjx_data(task, qpos, qvel, mocap_pos)
    policy_params = controller.init_params(seed=seed)

    jit_optimize = jax.jit(controller.optimize)
    interp_func = controller.interp_func

    @jax.jit
    def step_segment(
        state: mjx.Data, controls: jax.Array
    ) -> Tuple[mjx.Data, jax.Array]:
        """Apply controls one at a time; return updated state + cost sum."""

        def body(carry: mjx.Data, u: jax.Array) -> Tuple[mjx.Data, jax.Array]:
            carry = carry.replace(ctrl=u)
            carry = mjx.step(task.model, carry)
            return carry, task.running_cost(carry, u) * dt

        state, costs = jax.lax.scan(body, state, controls)
        return state, jnp.sum(costs)

    @jax.jit
    def interp_controls(
        tk: jax.Array, mean: jax.Array, t_curr: jax.Array
    ) -> jax.Array:
        """Sample the control spline at this segment's sim times."""
        tq = jnp.arange(sim_per_ctrl) * dt + t_curr
        return interp_func(tq, tk, mean[None, ...])[0]

    # Warmup JIT (results discarded; state reset below)
    policy_params, _ = jit_optimize(mjx_data, policy_params)
    controls = interp_controls(
        policy_params.tk, policy_params.mean, mjx_data.time
    )
    _, _ = step_segment(mjx_data, controls)

    # Real run from a fresh initial state
    mjx_data = _make_initial_mjx_data(task, qpos, qvel, mocap_pos)
    policy_params = controller.init_params(seed=seed)

    total_cost = jnp.float32(0.0)
    start = time.time()
    for _ in range(num_ctrl_steps):
        policy_params, _ = jit_optimize(mjx_data, policy_params)
        controls = interp_controls(
            policy_params.tk, policy_params.mean, mjx_data.time
        )
        mjx_data, segment_cost = step_segment(mjx_data, controls)
        total_cost = total_cost + segment_cost
    # Block on result to get accurate wall time
    total_cost_f = float(np.asarray(total_cost))
    wall_time = time.time() - start
    return total_cost_f, wall_time


def run_open_loop(
    task: Task,
    controller: SamplingBasedController,
    qpos: np.ndarray,
    qvel: np.ndarray,
    mocap_pos: Optional[np.ndarray],
    num_iterations: int,
    seed: int = 0,
) -> Tuple[float, float]:
    """Run open-loop trajectory optimization from a fixed initial state.

    Returns (final_mean_cost, wall_time). The reported cost is the total
    cost of evaluating the final mean trajectory deterministically (no
    noise), so different algorithms are compared on the policy they would
    actually deploy. Wall time excludes JIT compilation.
    """
    mjx_data = _make_initial_mjx_data(task, qpos, qvel, mocap_pos)
    policy_params = controller.init_params(seed=seed)

    jit_optimize = jax.jit(controller.optimize)

    @jax.jit
    def evaluate_mean(state: mjx.Data, params: Any) -> jax.Array:
        """Roll out the current mean trajectory and return total cost."""
        tq = jnp.linspace(0.0, controller.plan_horizon, controller.ctrl_steps)
        controls = controller.interp_func(tq, params.tk, params.mean[None, ...])
        knots = params.mean[None, ...]
        _, traj = controller.eval_rollouts(
            controller.model, state, controls, knots
        )
        return jnp.sum(traj.costs[0])

    # Warmup
    warm_params, _ = jit_optimize(mjx_data, policy_params)
    _ = evaluate_mean(mjx_data, warm_params)

    # Real run
    policy_params = controller.init_params(seed=seed)
    start = time.time()
    for _ in range(num_iterations):
        policy_params, _ = jit_optimize(mjx_data, policy_params)
    final_cost = float(np.asarray(evaluate_mean(mjx_data, policy_params)))
    wall_time = time.time() - start
    return final_cost, wall_time
