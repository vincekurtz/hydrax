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
ctrl = PredictiveSampling(task, num_samples=1024, noise_level=0.5)

# Define the model used for simulation
mj_model = task.mj_model
mj_data = mujoco.MjData(mj_model)

# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=100,
    show_traces=True,
)
