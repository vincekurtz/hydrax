import argparse
from copy import deepcopy

import mujoco

from hydrax.algs import CEM
from hydrax.risk import AverageCost
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.humanoid_mocap import HumanoidMocap, HumanoidMocapOptions

"""
Run an interactive simulation of the humanoid motion capture tracking task.
"""

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an interactive simulation of mocap tracking with the G1."
)
parser.add_argument(
    "--warp",
    action="store_true",
    help="Whether to use the (experimental) MjWarp backend. (default: False)",
    required=False,
)
parser.add_argument(
    "--reference_filename",
    type=str,
    default="Lafan1/mocap/UnitreeG1/walk1_subject1.npz",
    help="Reference mocap file name, from https://huggingface.co/datasets/robfiras/loco-mujoco-datasets/tree/main.",
)
parser.add_argument(
    "--show_reference",
    action="store_true",
    help="Show the reference trajectory as a 'ghost' in the simulation.",
)
parser.add_argument(
    "--iterations",
    type=int,
    default=1,
    help="Number of CEM iterations.",
)

args = parser.parse_args()

# Define the task (cost and dynamics)
task = HumanoidMocap(
    reference_filename=args.reference_filename,
    impl="warp" if args.warp else "jax",
    options=HumanoidMocapOptions(),
)

# Set up the controller
ctrl = CEM(
    task,
    num_samples=512,
    num_elites=20,
    sigma_start=0.2,
    sigma_min=0.01,
    explore_fraction=0.5,
    plan_horizon=0.6,
    num_randomizations=0,
    risk_strategy=AverageCost(),
    spline_type="zero",
    num_knots=3,
    iterations=args.iterations,
)

# Define the model used for simulation
mj_model = deepcopy(task.mj_model)

# Set the initial state
mj_data = mujoco.MjData(mj_model)
mj_data.qpos[:] = task.reference_qpos[0]

if args.show_reference:
    reference = task.reference_qpos
else:
    reference = None

run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=100,
    show_traces=False,
    reference=reference,
    reference_fps=task.reference_fps,
)
