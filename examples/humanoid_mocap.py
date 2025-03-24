import mujoco

from hydrax.algs import CEM
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.humanoid_mocap import HumanoidMocap

"""
Run an interactive simulation of the humanoid motion capture tracking task.
"""


# Define the task (cost and dynamics)
task = HumanoidMocap(reference_filename="run1_subject2.csv")

# Set up the controller
ctrl = CEM(
    task,
    num_samples=512,
    num_elites=10,
    sigma_start=0.1,
    sigma_min=0.1,
)

# Define the model used for simulation
mj_model = task.mj_model
mj_model.opt.timestep = 0.01
mj_model.opt.iterations = 10
mj_model.opt.ls_iterations = 50
mj_model.opt.o_solimp = [0.9, 0.95, 0.001, 0.5, 2]
mj_model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_OVERRIDE

# Set the initial state
mj_data = mujoco.MjData(mj_model)
mj_data.qpos[:] = task.reference[0]

run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=100,
    show_traces=False,
)
