import mujoco

from hydrax.algs import PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.crane import Crane

"""
Run an interactive simulation of crane payload tracking
"""

# Define the task (cost and dynamics)
task = Crane()

# Set up the controller
ctrl = PredictiveSampling(task, num_samples=128, noise_level=0.05)

# Define the model used for simulation
mj_model = task.mj_model
mj_data = mujoco.MjData(mj_model)

# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=30,
    show_traces=False,
)
