import mujoco

from hydrax.algs import Evosax
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.double_pendulum import DoublePendulum
import evosax

"""
Run an interactive simulation of the double pendulum swingup task.
"""

# Define the task (cost and dynamics)
task = DoublePendulum(planning_horizon=20, sim_steps_per_control_step=5)

ctrl = Evosax(task, evosax.SAMR_GA, num_samples=2048)

# Define the model used for simulation
mj_model = task.mj_model

# Set the initial state (two joints)
mj_data = mujoco.MjData(mj_model)

# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=50,
    fixed_camera_id=None,
    show_traces=True,
    max_traces=2,
)
