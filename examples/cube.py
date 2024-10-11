import sys

import mujoco
import numpy as np

from hydrax import ROOT
from hydrax.algs import MPPI, PredictiveSampling
from hydrax.mpc import run_interactive
from hydrax.tasks.cube import CubeRotation

"""
Run an interactive simulation of the walker task.
"""

# Define the task (cost and dynamics)
task = CubeRotation()

# Set the controller based on command-line arguments
if len(sys.argv) == 1 or sys.argv[1] == "ps":
    print("Running predictive sampling")
    ctrl = PredictiveSampling(task, num_samples=128, noise_level=0.1)
elif sys.argv[1] == "mppi":
    print("Running MPPI")
    ctrl = MPPI(task, num_samples=128, noise_level=0.1, temperature=0.1)
else:
    print("Usage: python leap_hand.py [ps|mppi]")
    sys.exit(1)

# Define the model used for simulation
mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/cube/scene.xml")
start_state = np.zeros(mj_model.nq + mj_model.nv)

# Run the interactive simulation
run_interactive(
    mj_model,
    ctrl,
    start_state,
    frequency=10,
    fixed_camera_id=None,
    show_traces=False,
    max_traces=1,
)
