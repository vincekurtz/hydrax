import argparse

import evosax
import mujoco

from hydrax.algs import CEM, MPPI, Evosax, PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.cube import CubeRotation

"""
Run an interactive simulation of the cube rotation task.

Double click on the floating target cube, then change the goal orientation with
[ctrl + left click].
"""

# Define the task (cost and dynamics)
task = CubeRotation()

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an interactive simulation of the cube rotation task."
)
subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)"
)
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser("cem", help="Cross-Entropy Method")
subparsers.add_parser("cmaes", help="CMA-ES")
args = parser.parse_args()

# Set the controller based on command-line arguments
if args.algorithm == "ps" or args.algorithm is None:
    print("Running predictive sampling")
    ctrl = PredictiveSampling(
        task,
        num_samples=32,
        noise_level=0.2,
        num_randomizations=32,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=4,
    )
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task,
        num_samples=128,
        noise_level=0.2,
        temperature=0.001,
        num_randomizations=8,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=4,
    )
elif args.algorithm == "cem":
    print("Running CEM")
    ctrl = CEM(
        task,
        num_samples=128,
        num_elites=5,
        sigma_start=0.5,
        sigma_min=0.5,
        num_randomizations=8,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=4,
    )
elif args.algorithm == "cmaes":
    print("Running CMA-ES")
    ctrl = Evosax(
        task,
        evosax.Sep_CMA_ES,
        num_samples=128,
        elite_ratio=0.5,
        num_randomizations=8,
        plan_horizon=0.25,
        spline_type="zero",
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
    frequency=20,
    fixed_camera_id=None,
    show_traces=True,
    max_traces=1,
    trace_color=[1.0, 1.0, 1.0, 1.0],
)
