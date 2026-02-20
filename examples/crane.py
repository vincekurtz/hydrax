from copy import deepcopy

import mujoco

from hydrax.algs import PredictiveSampling
from hydrax.risk import ConditionalValueAtRisk
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.crane import Crane

"""
Run an interactive simulation of crane payload tracking
"""

# Define the task (cost and dynamics)
task = Crane()

# Set up the controller
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
