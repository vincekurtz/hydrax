import mujoco
import numpy as np

from hydrax import ROOT
from hydrax.algs import PredictiveSampling
from hydrax.mpc import run_interactive
from hydrax.tasks.humanoid import Humanoid

"""
Run an interactive simulation of the humanoid task.
"""

# Define the task (cost and dynamics)
task = Humanoid()

# Set up the controller
ctrl = PredictiveSampling(task, num_samples=1024, noise_level=0.2)

# Define the model used for simulation
mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/g1/scene.xml")
start_state = np.concatenate(
    [mj_model.keyframe("stand").qpos, np.zeros(mj_model.nv)]
)

# Run the interactive simulation
run_interactive(
    mj_model,
    ctrl,
    start_state,
    frequency=50,
    show_traces=True,
    max_traces=1,
)
