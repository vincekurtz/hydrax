import argparse

import mujoco

from hydrax.algs import CEM
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.humanoid_mocap import HumanoidMocap

"""
Run an interactive simulation of the humanoid motion capture tracking task.
"""

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an interactive simulation of mocap tracking with the G1."
)
parser.add_argument(
    "--reference_filename",
    type=str,
    default="walk1_subject1.csv",
    help="Reference mocap file name, from https://huggingface.co/datasets/unitreerobotics/LAFAN1_Retargeting_Dataset/tree/main/g1.",
)
parser.add_argument(
    "--show_reference",
    action="store_true",
    help="Show the reference trajectory as a 'ghost' in the simulation.",
)
args = parser.parse_args()

# Define the task (cost and dynamics)
task = HumanoidMocap(reference_filename=args.reference_filename)

# Set up the controller
ctrl = CEM(
    task,
    num_samples=512,
    num_elites=20,
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

if args.show_reference:
    reference = task.reference
else:
    reference = None

run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=100,
    show_traces=False,
    reference=reference,
)
