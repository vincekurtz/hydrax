import mujoco

from hydrax.algs import MPPI
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.humanoid import Humanoid

"""
Run an interactive simulation of the humanoid task.
"""

# Define the task (cost and dynamics)
task = Humanoid()

# Set up the controller
ctrl = MPPI(
    task,
    num_samples=128,
    noise_level=1.0,
    temperature=0.1,
    num_randomizations=4,
)

# Define the model used for simulation
mj_model = task.mj_model
mj_model.opt.timestep = 0.005

# Set the initial state so the robot falls and needs to stand back up
mj_data = mujoco.MjData(mj_model)
mj_data.qpos[:] = mj_model.keyframe("stand").qpos
mj_data.qpos[3:7] = [0.7, 0.0, -0.7, 0.0]

# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=50,
    show_traces=True,
    max_traces=1,
)
