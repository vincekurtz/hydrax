import argparse

import mujoco
import numpy as np

from hydrax.algs import MPPI, PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.pendulum import Pendulum

"""
Run an interactive simulation of the pendulum swingup task.
"""

# Define the task (cost and dynamics)
task = Pendulum(planning_horizon=10, sim_steps_per_control_step=5)

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an interactive simulation of the pendulum swingup task."
)
subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)"
)
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
args = parser.parse_args()

# Set the controller based on command-line arguments
if args.algorithm == "ps" or args.algorithm is None:
    print("Running predictive sampling")
    ctrl = PredictiveSampling(task, num_samples=32, noise_level=0.1)
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(task, num_samples=32, noise_level=0.1, temperature=0.1)
else:
    parser.error("Invalid algorithm")

# Define the model used for simulation
mj_model = task.mj_model

# Set the initial state
mj_data = mujoco.MjData(mj_model)
mj_data.qpos[:] = np.array([0.0])
mj_data.qvel[:] = np.array([0.0])

# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=50,
    fixed_camera_id=0,
    show_traces=True,
    max_traces=1,
)
