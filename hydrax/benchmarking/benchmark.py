"""Benchmark a sampling-based controller on a task using MJX.

Runs a closed-loop simulation entirely on-device as a single JIT-compiled
lax.scan, recording the per-step running cost and the total cost of each
sampled rollout at every MPC step.
"""

from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from hydrax.alg_base import SamplingBasedController
from hydrax.task_base import Task


def benchmark(
    task: Task,
    controller: SamplingBasedController,
    initial_state: mjx.Data,
    total_time: float,
    initial_params: Optional[Any] = None,
    save_path: Optional[str] = None,
) -> Tuple[jax.Array, jax.Array]:
    """Run a closed-loop MJX benchmark of a sampling-based controller.

    At each step the controller replans (one call to `controller.optimize`),
    the first action of the resulting policy is applied, and the system is
    advanced by one MJX simulation step. The entire rollout is jitted as a
    single `lax.scan`, so all work happens on the GPU.

    Args:
        task: The task defining the simulation dynamics and cost.
        controller: The sampling-based controller to benchmark. May plan with
                    a different (e.g. domain-randomized) model than `task`.
        initial_state: The MJX state to start the rollout from.
        total_time: Total simulated time in seconds. The number of
                    closed-loop steps is `round(total_time / task.dt)`.
        initial_params: Optional initial policy parameters. Defaults to
                        `controller.init_params()`.
        save_path: If provided, save the results to this path as an `.npz`
                   file via `save_results`.

    Returns:
        running_costs: Shape `(num_steps,)`. The realized running cost
                       `ℓ(x_{t+1}, u_t)` at each simulation step.
        rollout_costs: Shape `(num_steps, num_samples)`. The total cost
                       (summed over the planning horizon, after the
                       controller's risk-strategy aggregation) of each
                       sampled rollout at each MPC step.
    """
    if total_time <= 0:
        raise ValueError(f"total_time must be positive, got {total_time}")

    num_steps = int(round(total_time / task.dt))
    if num_steps < 1:
        raise ValueError(
            f"total_time={total_time}s is shorter than one sim step "
            f"(task.dt={task.dt}s)"
        )

    if initial_params is None:
        initial_params = controller.init_params()

    # Assume the controller's model matches the true model perfectly.
    sim_model = task.model

    def _step(
        carry: Tuple[mjx.Data, Any], _: None
    ) -> Tuple[Tuple[mjx.Data, Any], Tuple[jax.Array, jax.Array]]:
        state, params = carry
        params, rollouts = controller.optimize(state, params)
        u = controller.get_action(params, state.time)
        state = state.replace(ctrl=u)
        state = mjx.step(sim_model, state)
        running_cost = task.running_cost(state, u)
        # rollouts.costs: (num_samples, H+1) after risk-strategy combination
        rollout_total_costs = jnp.sum(rollouts.costs, axis=-1)
        return (state, params), (running_cost, rollout_total_costs)

    @jax.jit
    def _run(state: mjx.Data, params: Any) -> Tuple[jax.Array, jax.Array]:
        _, (running_costs, rollout_costs) = jax.lax.scan(
            _step, (state, params), xs=None, length=num_steps
        )
        return running_costs, rollout_costs

    running_costs, rollout_costs = _run(initial_state, initial_params)

    # Force the computation to finish so timing and save_path are meaningful.
    running_costs.block_until_ready()
    rollout_costs.block_until_ready()

    if save_path is not None:
        save_results(save_path, running_costs, rollout_costs)

    return running_costs, rollout_costs


def save_results(
    path: str,
    running_costs: jax.Array,
    rollout_costs: jax.Array,
) -> None:
    """Save benchmark results to an `.npz` file for later processing.

    Args:
        path: Output file path.
        running_costs: Per-step realized running costs, shape `(num_steps,)`.
        rollout_costs: Per-step rollout totals, shape
                       `(num_steps, num_samples)`.
    """
    np.savez(
        path,
        running_costs=np.asarray(running_costs),
        rollout_costs=np.asarray(rollout_costs),
    )
