from copy import deepcopy

import mujoco

from hydrax.algs import PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.pusht import PushT

"""
Run an interactive simulation of the push-T task with predictive sampling.
"""

# Define the task (cost and dynamics)
task = PushT()

# Set up the controller
ctrl = PredictiveSampling(
    task,
    num_samples=128,
    noise_level=0.4,
    num_randomizations=4,
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
mj_data.qpos = [0.1, 0.1, 1.3, 0.0, 0.0]

# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=50,
    show_traces=True,
)
