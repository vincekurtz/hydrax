import argparse

from mujoco import mjx

from hydrax.algs import CEM
from hydrax.risk import AverageCost, WorstCase, BestCase
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
    default="Lafan1/mocap/UnitreeG1/walk1_subject1.npz",
    help="Reference mocap file name, from https://huggingface.co/datasets/robfiras/loco-mujoco-datasets/tree/main.",
)
parser.add_argument(
    "--duration",
    type=float,
    required=True,
    help="Duration to run the simulation (seconds).",
)
parser.add_argument(
    "--num_randomizations",
    type=int,
    default=0,
    help="Number of randomizations to perform for risk-sensitive strategies (default: 0).",
)
parser.add_argument(
    "--risk_strategy",
    type=str,
    choices=["average", "worst", "best"],
    default="average",
    help="Risk strategy to use (choices: [average, worst, best], default: average).",
)
# TODO: arg for domain randomization level

args = parser.parse_args()

# Define the task (cost and dynamics)
task = HumanoidMocap(
    reference_filename=args.reference_filename,
    impl="warp" if args.warp else "jax",
    options=HumanoidMocapOptions(),
)

# check duration value
assert args.duration > 0, "duration must be a positive number."

# check domain randomization value
assert args.num_randomizations >= 0, "num_randomizations must be non-negative integer."

# select the risk strategy
if args.risk_strategy == "average":
    risk_strategy_ = AverageCost()
elif args.risk_strategy == "worst":
    risk_strategy_ = WorstCase()
elif args.risk_strategy == "best":
    risk_strategy_ = BestCase()
else:
    raise ValueError(f"Invalid risk strategy: {args.risk_strategy}")

# Set up the controller
ctrl = CEM(
    task,
    num_samples=1024,
    num_elites=10,
    sigma_start=0.3,
    sigma_min=0.1,
    explore_fraction=0.2,
    plan_horizon=0.8,
    num_randomizations=args.num_randomizations,
    risk_strategy=risk_strategy_,
    spline_type="cubic",
    num_knots=4,
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
