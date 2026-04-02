import argparse
from copy import deepcopy

import mujoco

from hydrax.algs import PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.pusht import PushT

"""
Run an interactive simulation of the push-T task with predictive sampling.

q0_T = [0.0    0.1   0.0    0.0   0.1]  # translate
q0_T = [0.0    0.0   3.14  -0.1  -0.1]  # rotate
q0_T = [0.1    0.1   1.57   0.0   0.0]  # translate and rotate 1
q0_T = [-0.1  -0.1   3.14   0.0  -0.1]  # translate and rotate 2

"""

parser = argparse.ArgumentParser()
parser.add_argument("--initial_pos", type=float, nargs=5, default=[0.1, 0.1, 1.3, 0.0, 0.0],
                    help="Initial positions: tx ty ttheta px py")
args = parser.parse_args()
q0 = list(args.initial_pos)

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
