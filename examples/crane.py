import mujoco

from hydrax.algs import MPPI
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.crane import Crane

"""
Run an interactive simulation of crane payload tracking
"""

# Define the task (cost and dynamics)
task = Crane()

# Set up the controller
ctrl = MPPI(task, num_samples=128, noise_level=0.1, temperature=0.0001)
# ctrl = PredictiveSampling(task, num_samples=1024, noise_level=0.01)

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
    max_traces=5,
)
