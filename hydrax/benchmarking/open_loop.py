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
        costs: Cost of explicitly rolling out the mean control tape
            (``params.mean``) at each optimization iteration,
            shape ``(iterations,)``.
        final_params: Policy parameters after the last optimization iteration.
    """

    costs: jax.Array
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
    *initial_state*, explicitly rolling out the updated mean control tape
    (``params.mean``) after each call and recording that trajectory cost.
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

                * ``costs`` – explicit rollout cost of ``params.mean`` per
                    iteration,
          shape ``(iterations,)``.
        * ``final_params`` – policy parameters after the final iteration.
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    if params is None:
        params = ctrl.init_params(seed=seed)

    def _body(carry: Any, _: Any):
        new_carry, _ = ctrl.optimize(initial_state, carry)

        # Evaluate exactly one trajectory: the control tape defined by
        # params.mean, rather than statistics over sampled rollouts.
        mean_knots = jnp.clip(new_carry.mean, ctrl.task.u_min, ctrl.task.u_max)
        mean_rollout = ctrl.rollout_with_randomizations(
            initial_state,
            new_carry.tk,
            mean_knots[None, ...],
            new_carry.rng,
        )
        mean_cost = jnp.sum(mean_rollout.costs[0], axis=-1)
        return new_carry, mean_cost

    @jax.jit
    def _run(carry: Any):
        final_carry, costs = jax.lax.scan(
            _body, carry, None, length=iterations
        )
        return final_carry, costs

    final_params, costs = _run(params)
    return BenchmarkResult(
        costs=costs,
        final_params=final_params,
    )
