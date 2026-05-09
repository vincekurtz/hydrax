import argparse
from copy import deepcopy

import mujoco

from hydrax.algs import ErCma, PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.pusht import PushT

"""
Run an interactive simulation of the push-T task with predictive sampling.
"""

parser = argparse.ArgumentParser(
    description="Run an interactive simulation of the push-T task."
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
task = PushT(impl="warp" if args.warp else "jax")

# Set up the controller
if args.algorithm == "ps" or args.algorithm is None:
    print("Running predictive sampling")
    ctrl = PredictiveSampling(
        task,
        num_samples=128,
        noise_level=0.4,
        num_randomizations=4,
        plan_horizon=0.5,
        spline_type="zero",
        num_knots=6,
    )
elif args.algorithm == "er_cma":
    print("Running ER-CMA")
    ctrl = ErCma(
        task,
        num_samples=512,
        initial_noise_level=0.4,
        minimum_noise_level=1e-3,
        maximum_noise_level=1e3,
        initial_entropy_bonus=0.3,
        final_entropy_bonus=0.5,
        covariance_adaptation_rate=0.1,
        temperature=0.001,
        num_randomizations=1,
        plan_horizon=0.5,
        spline_type="zero",
        num_knots=6,
    )
else:
    parser.error("Invalid algorithm")

# Define the model used for simulation
mj_model = deepcopy(task.mj_model)
mj_model.opt.timestep = 0.001
mj_model.opt.iterations = 100
mj_model.opt.ls_iterations = 50
mj_data = mujoco.MjData(mj_model)
mj_data.qpos = [0.1, 0.1, 1.3, 0.0, 0.0]

# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=50,
    show_traces=False,
)
