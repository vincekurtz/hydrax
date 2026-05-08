"""GPU-only closed-loop MPC benchmarking utilities.

The benchmark repeatedly replans with a sampling-based controller, applies the
resulting policy in receding-horizon fashion, and advances the system state
with MJX dynamics. All compute runs inside a single JIT-compiled
``jax.lax.scan``.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from mujoco import mjx

from hydrax.alg_base import SamplingBasedController, SamplingParams


class ClosedLoopBenchmarkResult(NamedTuple):
    """Result of a closed-loop benchmark run.

    Attributes:
        costs: Cumulative realized running cost after each replan step,
            shape ``(num_replans,)``.
        times: Simulation times corresponding to ``costs``,
            shape ``(num_replans,)``.
        final_state: Final MJX state after the last simulated step.
        final_params: Policy parameters after the last optimization step.
    """

    costs: jax.Array
    times: jax.Array
    final_state: mjx.Data
    final_params: Any


def run_closed_loop_benchmark(
    ctrl: SamplingBasedController,
    initial_state: mjx.Data,
    total_time: float,
    replan_frequency: float,
    seed: int = 0,
    params: SamplingParams | None = None,
) -> ClosedLoopBenchmarkResult:
    """Run a closed-loop MPC benchmark over a fixed simulation duration.

    At each replan step, this function:

    1. Calls ``ctrl.optimize`` from the current state.
    2. Simulates forward for one replan period using controls queried from the
       optimized control spline.
    3. Accumulates realized running cost along the simulated trajectory.

    The benchmark discretizes time using the task time step and the requested
    replanning frequency. If the requested period is not an integer multiple
    of ``task.dt``, the nearest integer number of simulator steps is used.

    Args:
        ctrl: The sampling-based controller to benchmark.
        initial_state: Initial state ``x0`` (MJX data on device).
        total_time: Requested benchmark duration in seconds.
        replan_frequency: Requested replanning rate in Hz.
        seed: Random seed used when ``params`` is not provided.
        params: Optional pre-initialized policy parameters. If provided,
            ``seed`` is ignored.

    Returns:
        A :class:`ClosedLoopBenchmarkResult` containing cumulative realized
        running cost, corresponding simulation times, final state, and final
        policy parameters.
    """
    if total_time <= 0.0:
        raise ValueError("total_time must be > 0")
    if replan_frequency <= 0.0:
        raise ValueError("replan_frequency must be > 0")

    if params is None:
        params = ctrl.init_params(seed=seed)

    sim_dt = float(ctrl.task.dt)
    requested_replan_period = 1.0 / replan_frequency
    sim_steps_per_replan = max(int(round(requested_replan_period / sim_dt)), 1)
    replan_period = sim_steps_per_replan * sim_dt
    num_replans = max(int(jnp.ceil(total_time / replan_period)), 1)

    def _simulate_one_replan_period(
        state: mjx.Data,
        params: Any,
    ) -> tuple[mjx.Data, Any, jax.Array]:
        params, _ = ctrl.optimize(state, params)

        tq = state.time + jnp.arange(sim_steps_per_replan) * sim_dt
        knots = jnp.clip(params.mean, ctrl.task.u_min, ctrl.task.u_max)[None]
        controls = ctrl.interp_func(tq, params.tk, knots)[0]

        def _sim_step(
            x: mjx.Data,
            u: jax.Array,
        ) -> tuple[mjx.Data, jax.Array]:
            x = x.replace(ctrl=u)
            x = mjx.step(ctrl.task.model, x)
            running_cost = sim_dt * ctrl.task.running_cost(x, u)
            return x, running_cost

        state, running_costs = jax.lax.scan(_sim_step, state, controls)
        period_cost = jnp.sum(running_costs)

        return state, params, period_cost

    def _body(
        carry: tuple[mjx.Data, Any, jax.Array],
        _: Any,
    ) -> tuple[tuple[mjx.Data, Any, jax.Array], jax.Array]:
        state, policy_params, cumulative_cost = carry
        state, policy_params, period_cost = _simulate_one_replan_period(
            state, policy_params
        )
        cumulative_cost = cumulative_cost + period_cost
        return (state, policy_params, cumulative_cost), cumulative_cost

    @jax.jit
    def _run(initial_carry: tuple[mjx.Data, Any, jax.Array]):
        final_carry, costs = jax.lax.scan(
            _body,
            initial_carry,
            None,
            length=num_replans,
        )
        return final_carry, costs

    (final_state, final_params, _), costs = _run(
        (initial_state, params, jnp.array(0.0))
    )
    times = initial_state.time + jnp.arange(1, num_replans + 1) * replan_period

    return ClosedLoopBenchmarkResult(
        costs=costs,
        times=times,
        final_state=final_state,
        final_params=final_params,
    )