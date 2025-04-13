import argparse

import mujoco
import numpy as np

from hydrax.algs import MPPI, PredictiveSampling, CEM, Evosax
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.double_pendulum import DoublePendulum  # Import the new task
import evosax

"""
Run an interactive simulation of the double pendulum swingup task.
"""

# Define the task (cost and dynamics)
# Note: May need tuning of horizon, num_samples, noise_level, temperature
task = DoublePendulum(planning_horizon=20, sim_steps_per_control_step=5)

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an interactive simulation of the double pendulum swingup task."
)
subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)"
)
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser("cem", help="Cross-Entropy Method")
subparsers.add_parser("samr", help="SAMR")
args = parser.parse_args()

# Set the controller based on command-line arguments
# Parameters might need tuning for the harder double pendulum task
if args.algorithm == "ps" or args.algorithm is None:
    print("Running predictive sampling")
    # Increased samples might be needed
    ctrl = PredictiveSampling(task, num_samples=64, noise_level=0.2)
elif args.algorithm == "mppi":
    print("Running MPPI")
    # Increased samples and adjusted noise/temp might be needed
    ctrl = MPPI(task, num_samples=1024, noise_level=0.3, temperature=0.1)
elif args.algorithm == "cem":
    print("Running CEM")
    ctrl = CEM(
        task,
        num_samples=500,
        num_elites=20,
        sigma_start=0.4,
        sigma_min=0.05,
        explore_fraction=0.1,
    )
elif args.algorithm == "samr":
    print("Running SAMR")
    ctrl = Evosax(task, evosax.SAMR_GA, num_samples=1024)
else:
    parser.error("Invalid algorithm")

# Define the model used for simulation
mj_model = task.mj_model

# Set the initial state (two joints)
mj_data = mujoco.MjData(mj_model)
mj_data.qpos[:] = np.array([0.0, 0.0])  # Initial angles for both joints
mj_data.qvel[:] = np.array([0.0, 0.0])  # Initial velocities for both joints

# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=50,
    fixed_camera_id=None,
    show_traces=True,
    max_traces=2,  # Show traces for both tips
)
