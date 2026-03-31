import argparse

import os
import h5py
import numpy as np

from mujoco import mjx

from hydrax.algs import PredictiveSampling
from hydrax.risk import AverageCost, WorstCase, BestCase
from hydrax.simulation.deterministic import run_headless_pusht
from hydrax.tasks.pusht import PushT

"""
Run a headless simulation of the push-T task.
"""

##################################################################
# PARSE ARGUMENTS
##################################################################

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run a headless simulation of the push-T task."
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
    help="Risk strategy to use, default: average).",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Random seed for domain randomization (default: 0).",
)

args = parser.parse_args()

# check duration value
assert args.duration > 0.0, "duration must be a positive number."

# check num_randomizations value
assert args.num_randomizations >= 0, "num_randomizations must be non-negative integer."

# check seed value
assert type(args.seed) == int and args.seed >= 0, "seed must be a non-negative integer."

# select the risk strategy
risk_strategies = {"average": AverageCost, "worst": WorstCase, "best": BestCase}
risk_strategy_ = risk_strategies[args.risk_strategy]()


##################################################################
# SIMULATION
##################################################################

# Define the task (cost and dynamics)
task = PushT(impl="warp" if args.warp else "jax")

# PredictiveSampling options (saved as a dict so we can reuse for logging)
ctrl_options = {
    "num_samples": 128,
    "noise_level": 0.4,
    "num_randomizations": args.num_randomizations,
    "seed": args.seed,
    "risk_strategy": risk_strategy_,
    "plan_horizon": 0.5,
    "spline_type": "zero",
    "num_knots": 6,
}

# Set up the controller
ctrl = PredictiveSampling(task, **ctrl_options)

# Replace risk_strategy object with string for saving dictionary
ctrl_options["risk_strategy"] = args.risk_strategy

# Create the mjx model/data for simulation
mj_model_sim = task.mj_model
mj_model_sim.opt.timestep = 0.001
mj_model_sim.opt.iterations = 100
mj_model_sim.opt.ls_iterations = 50
mjx_model_sim = mjx.put_model(mj_model_sim)
mjx_data_sim = mjx.make_data(mj_model_sim)
mjx_data_sim = mjx_data_sim.replace(
    qpos=np.array([0.1, 0.1, 1.3, 0.0, 0.0]),
)

# Run the headless simulation
results = run_headless_pusht(
    ctrl,
    mjx_model_sim,
    mjx_data_sim,
    frequency=50,
    duration=args.duration,
)


##################################################################
# SAVE RESULTS
##################################################################

# command line arguments for saving
experiment_args = {
    "warp": args.warp,
    "duration": args.duration,
    "num_randomizations": args.num_randomizations,
    "risk_strategy": args.risk_strategy,
    "seed": args.seed,
}

# save directory
save_dir = "experiments/pusht/data"
os.makedirs(save_dir, exist_ok=True)
filename = f"run_{int(args.run_id):03d}.h5" if args.run_id else "results.h5"
save_path = os.path.join(save_dir, filename)

# Save results to HDF5 file
with h5py.File(save_path, "w") as f:
    # experiment arguments
    args_grp = f.create_group("experiment_args")
    for k, v in experiment_args.items():
        args_grp.attrs[k] = v

    # PredictiveSampling options
    ps_grp = f.create_group("ctrl_options")
    for k, v in ctrl_options.items():
        ps_grp.attrs[k] = v

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

print("Results saved to:", save_path)
