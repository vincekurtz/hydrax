import argparse
import time
from copy import deepcopy

from evosax.algorithms.distribution_based import CMA_ES
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
from mujoco import mjx

from hydrax.algs import CEM, MPPI, PredictiveSampling, Evosax
from hydrax.tasks.cart_pole import CartPole

"""
Perform open-loop trajectory optimization for the cart-pole swingup task.
"""

# Define the task (cost and dynamics)
task = CartPole()

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Perform open-loop trajectory optimization for the cart-pole."
)
parser.add_argument(
    "--iterations",
    type=int,
    default=10,
    help="Number of optimization iterations to perform.",
)
subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)"
)
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser("cem", help="Cross-Entropy Method")
subparsers.add_parser("cmaes", help="CMA-ES")
args = parser.parse_args()

# Set up the controller
if args.algorithm == "ps" or args.algorithm is None:
    print("Running predictive sampling")
    ctrl = PredictiveSampling(
        task,
        num_samples=128,
        noise_level=0.3,
        spline_type="cubic",
        plan_horizon=2.0,
        num_knots=4,
    )
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task,
        num_samples=128,
        noise_level=0.3,
        temperature=0.1,
        spline_type="cubic",
        plan_horizon=2.0,
        num_knots=4,
    )
elif args.algorithm == "cem":
    print("Running CEM")
    ctrl = CEM(
        task,
        num_samples=128,
        num_elites=3,
        sigma_start=0.5,
        sigma_min=0.1,
        spline_type="cubic",
        plan_horizon=2.0,
        num_knots=4,
    )
elif args.algorithm == "cmaes":
    print("Running CMA-ES")
    ctrl = Evosax(
        task,
        CMA_ES,
        num_samples=128,
        plan_horizon=2.0,
        spline_type="zero",
        num_knots=4,
    )
else:
    parser.error("Other algorithms not implemented for this example!")

# Set the initial state
mjx_data = mjx.make_data(task.mj_model)  # TODO: use task.make_data()

# Run the optimization loop
params = ctrl.init_params()
jit_optimizer_step = jax.jit(ctrl.optimize)

for i in range(args.iterations):
    print(f"Iteration {i + 1}/{args.iterations}:", end="")
    params, rollouts = jit_optimizer_step(mjx_data, params)

    # Report average and best cost
    costs = jnp.sum(rollouts.costs, axis=1)  # sum over timesteps
    avg_cost = jnp.mean(costs)
    std_cost = jnp.std(costs)
    best_cost = jnp.min(costs)

    print(f" Best: {best_cost:.3f}, Avg: {avg_cost:.3f}, Std: {std_cost:.3f}")

# Select the minimum-cost trajectory
best_idx = jnp.argmin(costs)

# Get the state trajectory corresponding to the best trajectory
print("Retrieving best trajectory...")
states, _ = jax.jit(ctrl.eval_rollouts)(
    task.model,
    mjx_data,
    rollouts.controls[best_idx, None],  # get the proper
    rollouts.knots[best_idx, None],
)

# Play back on the mujoco visualizer
print("Starting playback...")
mj_model = deepcopy(task.mj_model)
mj_data = mujoco.MjData(mj_model)

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    i = 0
    while viewer.is_running():
        start_time = time.time()

        # Set the state to the current point in the trajectory
        mj_data.qpos[:] = states.qpos[0, i]
        mj_data.qvel[:] = states.qvel[0, i]
        mj_data.time += ctrl.dt
        mujoco.mj_forward(mj_model, mj_data)
        viewer.sync()

        # Run in roughly real time
        elapsed = time.time() - start_time
        if elapsed < ctrl.dt:
            time.sleep(ctrl.dt - elapsed)

        # Loop the trajectory when we reach the end
        i += 1
        if i >= states.qpos.shape[1]:
            time.sleep(1.0)  # pause for a moment
            i = 0
            mj_data.time = 0.0
