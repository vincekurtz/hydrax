import argparse
from copy import deepcopy

import mujoco

from hydrax.algs import ErCma, PredictiveSampling
from hydrax.risk import ConditionalValueAtRisk
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.crane import Crane

"""
Run an interactive simulation of crane payload tracking
"""

parser = argparse.ArgumentParser(
    description="Run an interactive simulation of crane payload tracking"
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
task = Crane(impl="warp" if args.warp else "jax")

# Set up the controller
if args.algorithm == "ps" or args.algorithm is None:
    print("Running predictive sampling")
    ctrl = PredictiveSampling(
        task,
        num_samples=8,
        noise_level=0.05,
        num_randomizations=32,
        risk_strategy=ConditionalValueAtRisk(0.1),
        plan_horizon=0.8,
        spline_type="zero",
        num_knots=3,
    )
elif args.algorithm == "er_cma":
    print("Running ER-CMA")
    ctrl = ErCma(
        task,
        num_samples=32,
        initial_noise_level=0.05,
        minimum_noise_level=1e-3,
        maximum_noise_level=1e3,
        initial_entropy_bonus=0.3,
        final_entropy_bonus=0.5,
        covariance_adaptation_rate=0.1,
        temperature=0.1,
        plan_horizon=0.8,
        spline_type="zero",
        num_knots=3,
    )
else:
    parser.error("Invalid algorithm")

# Define the model used for simulation
mj_model = deepcopy(task.mj_model)
mj_data = mujoco.MjData(mj_model)

# Introduce some modeling error
mj_model.dof_damping *= 0.1
body_idx = mj_model.body("payload").id
mj_model.body_mass[body_idx] *= 1.5
mj_model.body_inertia[body_idx] *= 1.5

# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=30,
    show_traces=False,
)
