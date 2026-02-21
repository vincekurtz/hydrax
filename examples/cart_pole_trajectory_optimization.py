import argparse

from evosax.algorithms.distribution_based import CMA_ES

from hydrax.algs import CEM, MPPI, Evosax, PredictiveSampling
from hydrax.open_loop import playback, trajectory_optimization
from hydrax.tasks.cart_pole import CartPole

"""
Perform open-loop trajectory optimization for the cart-pole swingup task.
"""

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
parser.add_argument(
    "--warp",
    action="store_true",
    help="Whether to use the (experimental) MjWarp backend. (default: False)",
    required=False,
)
subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)"
)
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser("cem", help="Cross-Entropy Method")
subparsers.add_parser("cmaes", help="CMA-ES")
args = parser.parse_args()

# Define the task (cost and dynamics)
task = CartPole(impl="warp" if args.warp else "jax")

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
        spline_type="cubic",
        num_knots=4,
    )
else:
    parser.error("Other algorithms not implemented for this example!")

# Run trajectory optimization
mjx_data = task.make_data()  # initial state
optimal_trajectory = trajectory_optimization(ctrl, mjx_data, args.iterations)

# Play back on the mujoco visualizer
print("Starting playback...")
playback(optimal_trajectory, ctrl)
