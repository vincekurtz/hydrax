import argparse

import mujoco
import numpy as np

from hydrax.algs import CEM, DIAL, MPPI, PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.quadruped import QuadrupedStanding

"""
Run an interactive simulation of the quadruped standing task.
"""

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an interactive simulation of quadruped (Unitree Go2) standing."
)
subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)"
)
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser("dial", help="Diffusion-Inspired Annealing for Legged MPC")
subparsers.add_parser("cem", help="Cross-Entropy Method")
args = parser.parse_args()

# Define the task (cost and dynamics)
task = QuadrupedStanding()

# Set the controller based on command-line arguments
if args.algorithm == "ps" or args.algorithm is None:
    print("Running Predictive Sampling")
    ctrl = PredictiveSampling(
        task,
        num_samples=128,
        noise_level=0.3,
        plan_horizon=0.6,
        spline_type="zero",
        num_knots=4,
    )
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task,
        num_samples=128,
        noise_level=0.3,
        temperature=0.1,
        num_randomizations=4,
        plan_horizon=0.6,
        spline_type="zero",
        num_knots=4,
    )
elif args.algorithm == "dial":
    print("Running DIAL-MPC")
    ctrl = DIAL(
        task,
        num_samples=128,
        noise_level=0.3,
        beta_opt_iter=1.0,
        beta_horizon=1.0,
        temperature=0.1,
        num_randomizations=4,
        plan_horizon=0.6,
        spline_type="zero",
        num_knots=4,
        iterations=2,
    )
elif args.algorithm == "cem":
    print("Running CEM")
    ctrl = CEM(
        task,
        num_samples=256,
        num_elites=20,
        sigma_start=0.3,
        sigma_min=0.05,
        explore_fraction=0.5,
        plan_horizon=0.6,
        spline_type="zero",
        num_knots=4,
        iterations=1,
    )
else:
    parser.error("Invalid algorithm")

# Define the model used for simulation
mj_model = task.mj_model
mj_model.opt.timestep = 0.01
mj_model.opt.o_solimp = [0.9, 0.95, 0.001, 0.5, 2]
mj_model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_OVERRIDE

# Set the initial state to the home keyframe (standing position)
mj_data = mujoco.MjData(mj_model)
mj_data.qpos[:] = mj_model.keyframe("home").qpos

# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=50,
    show_traces=False,
)
