import time
from copy import deepcopy
from datetime import datetime

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
from mujoco import mjx

from hydrax.alg_base import SamplingBasedController


def trajectory_optimization(
    ctrl: SamplingBasedController,
    initial_state: mjx.Data,
    iterations: int,
) -> mjx.Data:
    """Perform open-loop sampling-based trajectory optimization.

    Args:
        ctrl: The sampling-based controller to use for optimization. Defines the
              task, planning horizon, dynamics, etc.
        initial_state: The initial state of the system.
        iterations: The number of optimization iterations to perform.

    Returns:
        The optimized state trajectory, packed in an mjx.Data object. This data
        object will have fields of shape (T, ...) corresponding to the
        optimized trajectory. For example, the configuration at the i-th
        timestep will be in data.qpos[i].
    """
    assert iterations > 0, "Number of iterations must be positive"

    params = ctrl.init_params()
    jit_optimizer_step = jax.jit(ctrl.optimize)

    print("Starting open-loop trajectory optimization...")
    start_time = datetime.now()
    for i in range(iterations):
        # Perform a single optimization step
        params, rollouts = jit_optimizer_step(initial_state, params)

        # Report average and best cost
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over timesteps
        avg_cost = jnp.mean(costs)
        std_cost = jnp.std(costs)
        best_cost = jnp.min(costs)

        # Progress printout
        elapsed = datetime.now() - start_time
        print(
            f"  Iteration {i + 1}/{iterations} | Cost: {best_cost:.3f}, "
            + f"Avg: {avg_cost:.3f}, Std: {std_cost:.3f}, Time: {elapsed}"
        )

    # Select the minimum-cost trajectory
    best_idx = jnp.argmin(costs)

    # Get the state trajectory corresponding to the best trajectory
    print("Retrieving best trajectory...")
    states, _ = jax.jit(ctrl.eval_rollouts)(
        ctrl.model,
        initial_state,
        rollouts.controls[best_idx, None],  # get the proper vmap shape
        rollouts.knots[best_idx, None],
    )

    # Un-vmap the trajectory to get arrays of shape (T, ...)
    states = jax.tree.map(lambda x: x[0], states)

    return states


def playback(trajectory: mjx.Data, ctrl: SamplingBasedController) -> None:
    """Play back an open-loop trajectory on the mujoco visualizer.

    The trajectory will be looped indefinitely until the user closes the
    visualizer. No interactive forces are applied: this is purely for
    visualization.

    Args:
        trajectory: An optimized trajectory, as returned by
                    trajectory_optimization. This should have fields of shape
                    (T, ...) corresponding to the optimized trajectory.
        ctrl: The sampling-based controller used to generate the trajectory.
              This is needed to access the mujoco model and the timestep.
    """
    mj_model = deepcopy(ctrl.task.mj_model)
    mj_data = mujoco.MjData(mj_model)

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        i = 0
        while viewer.is_running():
            start_time = time.time()

            # Set the state to the current point in the trajectory
            mj_data.qpos[:] = trajectory.qpos[i]
            mj_data.qvel[:] = trajectory.qvel[i]
            mj_data.time += ctrl.dt
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()

            # Run in roughly real time
            elapsed = time.time() - start_time
            if elapsed < ctrl.dt:
                time.sleep(ctrl.dt - elapsed)

            # Loop the trajectory when we reach the end
            i += 1
            if i >= trajectory.qpos.shape[0]:
                time.sleep(1.0)  # pause for a moment
                i = 0
                mj_data.time = 0.0
