import argparse

from mujoco import mjx

from hydrax.algs import CEM
from hydrax.risk import AverageCost
from hydrax.simulation.deterministic import run_headless
from hydrax.tasks.humanoid_mocap import HumanoidMocap, HumanoidMocapOptions

"""
Run a headless simulation of the humanoid motion capture tracking task.
"""

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run a headless simulation of mocap tracking with the G1."
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
    default="g1/walk1_subject1.csv",
    help="Reference mocap file name, from https://huggingface.co/datasets/robfiras/loco-mujoco-datasets/tree/main.",
)
parser.add_argument(
    "--duration",
    type=float,
    required=True,
    help="Duration to run the simulation (seconds).",
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
    iterations=1,
)

# Create the mjx model/data for simulation
mjx_model_sim = mjx.put_model(task.mj_model)
mjx_data_sim = mjx.make_data(task.mj_model)
mjx_data_sim = mjx_data_sim.replace(qpos=task.reference_qpos[0])

# Run the headless simulation
result = run_headless(
    ctrl,
    mjx_model_sim,
    mjx_data_sim,
    frequency=100,
    duration=args.duration,
)
