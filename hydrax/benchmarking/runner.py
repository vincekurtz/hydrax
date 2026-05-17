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

    interp_func = controller.interp_func

    @jax.jit
    def mpc_step(
        state: mjx.Data, params: Any
    ) -> Tuple[mjx.Data, Any, jax.Array]:
        """Plan, interpolate, and simulate one control segment in one JIT.

        Merging all three operations avoids two extra kernel-dispatch round-
        trips per control step and lets XLA optimize across the boundaries.
        Rollouts from optimize() are not returned, so XLA need not allocate
        or copy those output buffers to the host.
        """
        # Plan: update policy params (rollouts discarded as XLA sees no
        # downstream use for their output buffers).
        params, _ = controller.optimize(state, params)

        # Interpolate the spline at this segment's sim-step times.
        tq = jnp.arange(sim_per_ctrl) * dt + state.time
        controls = interp_func(tq, params.tk, params.mean[None, ...])[0]

        # Simulate and accumulate running cost.
        def body(carry: mjx.Data, u: jax.Array) -> Tuple[mjx.Data, jax.Array]:
            carry = carry.replace(ctrl=u)
            carry = mjx.step(task.model, carry)
            return carry, task.running_cost(carry, u) * dt

        state, costs = jax.lax.scan(body, state, controls)
        return state, params, jnp.sum(costs)

    # Warmup: compile and fully execute mpc_step before the timed region.
    # block_until_ready prevents async-dispatch compilation from bleeding
    # into the timing region (JAX dispatches compilation asynchronously, so
    # without this the first timed call would still be waiting for it).
    _, _, warmup_cost = mpc_step(mjx_data, policy_params)
    jax.block_until_ready(warmup_cost)

    # Real run from a fresh initial state.
    mjx_data = _make_initial_mjx_data(task, qpos, qvel, mocap_pos)
    policy_params = controller.init_params(seed=seed)

    total_cost = jnp.float32(0.0)
    start = time.time()
    for _ in range(num_ctrl_steps):
        mjx_data, policy_params, segment_cost = mpc_step(
            mjx_data, policy_params
        )
        total_cost = total_cost + segment_cost
    # Block on device to get accurate wall time.
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

    @jax.jit
    def optimize_params(state: mjx.Data, params: Any) -> Any:
        """Run one optimize step; return only the updated params.

        Discarding rollouts from the return value lets XLA skip allocating
        and copying the trajectory output buffers (num_samples × H × nu)
        to the host on every call.
        """
        params, _ = controller.optimize(state, params)
        return params

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

    # Warmup both compiled functions; block to drain async compilation.
    warm_params = optimize_params(mjx_data, policy_params)
    warmup_cost = evaluate_mean(mjx_data, warm_params)
    jax.block_until_ready(warmup_cost)

    # Real run.
    policy_params = controller.init_params(seed=seed)
    start = time.time()
    for _ in range(num_iterations):
        policy_params = optimize_params(mjx_data, policy_params)
    final_cost = float(np.asarray(evaluate_mean(mjx_data, policy_params)))
    wall_time = time.time() - start
    return final_cost, wall_time
