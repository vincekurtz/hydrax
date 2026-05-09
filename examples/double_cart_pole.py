import argparse

import mujoco

from hydrax.algs import ErCma, PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.double_cart_pole import DoubleCartPole

"""
Run an interactive simulation of a double pendulum on a cart. Only the cart
is actuated, and the goal is to swing up the pendulum and balance it upright.
"""

parser = argparse.ArgumentParser(
    description="Run an interactive simulation of a double pendulum on a cart."
)
parser.add_argument(
    "--warp",
    action="store_true",
    help="Whether to use the (experimental) MjWarp backend. (default: False)",
)
subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)"
)
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("er_cma", help="Entropy-Regularized CMA")
args = parser.parse_args()

# Define the task (cost and dynamics)
task = DoubleCartPole(impl="warp" if args.warp else "jax")

# Set up the controller
if args.algorithm == "ps" or args.algorithm is None:
    print("Running predictive sampling")
    ctrl = PredictiveSampling(
        task,
        num_samples=1024,
        noise_level=0.3,
        plan_horizon=1.0,
        spline_type="cubic",
        num_knots=4,
    )
elif args.algorithm == "er_cma":
    print("Running ER-CMA")
    ctrl = ErCma(
        task,
        num_samples=1024,
        initial_noise_level=0.3,
        minimum_noise_level=1e-3,
        maximum_noise_level=1e3,
        initial_entropy_bonus=0.3,
        final_entropy_bonus=0.5,
        covariance_adaptation_rate=0.1,
        temperature=0.1,
        plan_horizon=1.0,
        spline_type="cubic",
        num_knots=4,
    )
else:
    parser.error("Invalid algorithm")

# Define the model used for simulation
mj_model = task.mj_model
mj_data = mujoco.MjData(mj_model)

# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=50,
    fixed_camera_id=0,
    show_traces=False,
    max_traces=1,
)
