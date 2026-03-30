import argparse

import os
import h5py
import numpy as np

from mujoco import mjx

from hydrax.algs import CEM
from hydrax.risk import AverageCost, WorstCase, BestCase, ValueAtRisk, ConditionalValueAtRisk
from hydrax.simulation.deterministic import run_headless
from hydrax.tasks.humanoid_mocap import HumanoidMocap, HumanoidMocapOptions

"""
Run a headless simulation of the humanoid motion capture tracking task.
"""

##################################################################
# PARSE ARGUMENTS
##################################################################

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run a headless simulation of mocap tracking with the G1."
)
parser.add_argument(
    "--run_id",
    type=str,
    default=None,
    help="Experiment run ID (e.g. '001'). Used for naming the output h5 file.",
)
parser.add_argument(
    "--warp",
    action="store_true",
    help="Whether to use the (experimental) MjWarp backend. (default: False)",
)
parser.add_argument(
    "--duration",
    type=float,
    required=True,
    help="Duration to run the simulation (seconds).",
)
parser.add_argument(
    "--reference_filename",
    type=str,
    default="Lafan1/mocap/UnitreeG1/walk1_subject1.npz",
    help="Reference mocap file name, from https://huggingface.co/datasets/robfiras/loco-mujoco-datasets/tree/main.",
)
parser.add_argument(
    "--num_randomizations",
    type=int,
    default=0,
    help="Number of randomizations to perform for risk-sensitive strategies (default: 0).",
)
parser.add_argument(
    "--level_randomization",
    type=float,
    default=1.0,
    help="Level of domain randomization, value is in (0.0, 1.0]. (default: 0.0)",
)
parser.add_argument(
    "--risk_strategy",
    type=str,
    choices=["average", "worst", "best", "var", "cvar"],
    default="average",
    help="Risk strategy to use, default: average).",
)

args = parser.parse_args()

# check duration value
assert args.duration > 0.0, "duration must be a positive number."

# check domain randomization values
# num_randomizations <= 1 means no domain randomization (alg_base clamps to 1
# and only randomizes when > 1), so level_randomization is only meaningful then.
assert args.num_randomizations >= 0, "num_randomizations must be non-negative integer."
if args.num_randomizations > 1:
    assert 0.0 < args.level_randomization <= 1.0, "level_randomization must be in (0.0, 1.0]."

# select the risk strategy
risk_strategies = {"average": AverageCost, "worst": WorstCase, "best": BestCase,
                   "var": ValueAtRisk, "cvar": ConditionalValueAtRisk}
risk_strategy_ = risk_strategies[args.risk_strategy]()


##################################################################
# SIMULATION
##################################################################

# Define the task (cost and dynamics)
task = HumanoidMocap(
    reference_filename=args.reference_filename,
    impl="warp" if args.warp else "jax",
    options=HumanoidMocapOptions(level_randomization=args.level_randomization),
)

# CEM options (saved as a dict so we can reuse for logging)
cem_options = {
    "num_samples": 1024,
    "num_elites": 10,
    "sigma_start": 0.3,
    "sigma_min": 0.1,
    "explore_fraction": 0.2,
    "plan_horizon": 0.8,
    "num_randomizations": args.num_randomizations,
    "risk_strategy": risk_strategy_,
    "spline_type": "cubic",
    "num_knots": 4,
    "iterations": 1,
}

# Set up the controller
ctrl = CEM(task, **cem_options)

# Replace risk_strategy object with string for saving dictionary 
cem_options["risk_strategy"] = args.risk_strategy

# Create the mjx model/data for simulation
mjx_model_sim = mjx.put_model(task.mj_model)
mjx_data_sim = mjx.make_data(task.mj_model)
mjx_data_sim = mjx_data_sim.replace(qpos=task.reference_qpos[0])

# Run the headless simulation
results = run_headless(
    ctrl,
    mjx_model_sim,
    mjx_data_sim,
    frequency=100,
    duration=args.duration,
)


##################################################################
# SAVE RESULTS
##################################################################

# command line arguments for saving
experiment_args = {
    "warp": args.warp,
    "duration": args.duration,
    "reference_filename": args.reference_filename,
    "num_randomizations": args.num_randomizations,
    "level_randomization": args.level_randomization,
    "risk_strategy": args.risk_strategy,
}

# save directory
save_dir = "experiments/data"
os.makedirs(save_dir, exist_ok=True)
filename = f"run{args.run_id}.h5" if args.run_id else "results.h5"
save_path = os.path.join(save_dir, filename)

# Save results to HDF5 file
with h5py.File(save_path, "w") as f:
    # experiment arguments
    args_grp = f.create_group("experiment_args")
    for k, v in experiment_args.items():
        args_grp.attrs[k] = v

    # CEM options
    cem_grp = f.create_group("cem_options")
    for k, v in cem_options.items():
        cem_grp.attrs[k] = v

    # metrics (arrays as datasets, scalars as attrs)
    met_grp = f.create_group("metrics")
    for k, v in results["metrics"].items():
        if isinstance(v, np.ndarray):
            met_grp.create_dataset(k, data=v, compression="gzip")
        else:
            met_grp.attrs[k] = v

    # trajectory data
    traj_grp = f.create_group("trajectory")
    traj_grp.create_dataset("qpos", data=results["trajectory"]["qpos"], compression="gzip")
    traj_grp.create_dataset("qvel", data=results["trajectory"]["qvel"], compression="gzip")
    traj_grp.create_dataset("ctrl", data=results["trajectory"]["ctrl"], compression="gzip")
    traj_grp.attrs["sim_dt"] = results["trajectory"]["sim_dt"]
    traj_grp.attrs["ctrl_dt"] = results["trajectory"]["ctrl_dt"]

print("Results saved to results.h5")
