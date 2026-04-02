import argparse
from copy import deepcopy

import mujoco

from hydrax.algs import PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.pusht import PushT

"""
Run an interactive simulation of the push-T task with predictive sampling.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--sim_seed", type=int, default=0,
                    help="Random seed for initial condition sampling (default: 0).")
args = parser.parse_args()

# Define the task (cost and dynamics)
task = PushT()

# Set up the controller
ctrl = PredictiveSampling(
    task,
    num_samples=128,
    noise_level=0.4,
    num_randomizations=0,
    plan_horizon=0.5,
    spline_type="zero",
    num_knots=6,
)

# Sample initial condition from seed
q0 = task.sample_initial_position(args.sim_seed)
print(f"sim_seed={args.sim_seed}, q0={q0}")

# Define the model used for simulation
mj_model = deepcopy(task.mj_model)
mj_model.opt.timestep = 0.001
mj_model.opt.iterations = 100
mj_model.opt.ls_iterations = 50
mj_data = mujoco.MjData(mj_model)
mj_data.qpos = q0

# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=50,
    show_traces=False,
)
