import mujoco

from hydrax import ROOT
from hydrax.algs import PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.humanoid import Humanoid

"""
Run an interactive simulation of the humanoid task.
"""

# Define the task (cost and dynamics)
task = Humanoid()

# Set up the controller
ctrl = PredictiveSampling(task, num_samples=128, noise_level=0.2)

# Define the model used for simulation
mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/g1/scene.xml")

# Set the initial state
mj_data = mujoco.MjData(mj_model)
mj_data.qpos[:] = mj_model.keyframe("stand").qpos

# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=30,
    show_traces=True,
    max_traces=1,
)
