"""GPU-only open-loop optimization benchmarking utilities.

All computation runs inside a single JIT-compiled JAX function, using
``jax.lax.scan`` to execute multiple optimization iterations in one compiled
graph. No MuJoCo CPU stepping or host/device transfers occur during the
optimization loop.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from mujoco import mjx

from hydrax.alg_base import SamplingBasedController, SamplingParams


class BenchmarkResult(NamedTuple):
    """Result of an open-loop benchmark run.

    Attributes:
        mean_costs: Mean rollout cost at each optimization iteration,
            shape ``(iterations,)``.
        best_costs: Minimum rollout cost at each optimization iteration,
            shape ``(iterations,)``.
        final_params: Policy parameters after the last optimization iteration.
    """

    mean_costs: jax.Array
    best_costs: jax.Array
    final_params: Any


def run_open_loop_benchmark(
    ctrl: SamplingBasedController,
    initial_state: mjx.Data,
    iterations: int,
    seed: int = 0,
    params: SamplingParams | None = None,
) -> BenchmarkResult:
    """Run an open-loop optimization benchmark, tracking cost convergence.

    Executes *iterations* calls to ``ctrl.optimize`` from the fixed
    *initial_state*, recording the mean and best rollout cost after each call.
    The entire compute path runs inside a single ``jax.jit``-compiled
    ``jax.lax.scan``: no MuJoCo CPU stepping, no NumPy conversions, and no
    host/device transfers occur during the optimization loop.

    Args:
        ctrl: The sampling-based controller to benchmark.
        initial_state: The fixed initial state ``x₀``.  Must be an
            ``mjx.Data`` already on device (e.g. from ``task.make_data()`` or
            ``mjx.put_data()``).
        iterations: Number of optimization iterations to run.
        seed: Random seed used to initialize policy parameters when *params* is
            not provided.
        params: Optional pre-initialized policy parameters.  When supplied,
            *seed* is ignored.

    Returns:
        A :class:`BenchmarkResult` with:

        * ``mean_costs`` – mean rollout cost per iteration,
          shape ``(iterations,)``.
        * ``best_costs`` – minimum rollout cost per iteration,
          shape ``(iterations,)``.
        * ``final_params`` – policy parameters after the final iteration.
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    if params is None:
        params = ctrl.init_params(seed=seed)

    def _body(carry: Any, _: Any):
        new_carry, rollouts = ctrl.optimize(initial_state, carry)
        # rollouts.costs: (num_rollouts, H+1); sum over horizon axis
        rollout_costs = jnp.sum(rollouts.costs, axis=-1)  # (num_rollouts,)
        return new_carry, (jnp.mean(rollout_costs), jnp.min(rollout_costs))

    @jax.jit
    def _run(carry: Any):
        final_carry, (mean_costs, best_costs) = jax.lax.scan(
            _body, carry, None, length=iterations
        )
        return final_carry, mean_costs, best_costs

    final_params, mean_costs, best_costs = _run(params)
    return BenchmarkResult(
        mean_costs=mean_costs,
        best_costs=best_costs,
        final_params=final_params,
    )
