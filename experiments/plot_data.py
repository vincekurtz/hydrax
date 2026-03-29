##
#
# Plot data from the humanoid mocap headless example.
#
##

import numpy as np
import h5py
import matplotlib.pyplot as plt

# Load results from HDF5 file
file_path = "./experiments/humanoid_mocap_headless/results.h5"

with h5py.File(file_path, "r") as f:
    # Load experiment arguments
    experiment_args = dict(f["experiment_args"].attrs)

    # Load CEM options
    cem_options = dict(f["cem_options"].attrs)

    # Load trajectory data (read datasets into numpy, attrs into scalars)
    traj_grp = f["trajectory"]
    trajectory = {k: v[:] for k, v in traj_grp.items()}
    trajectory.update(dict(traj_grp.attrs))

    # Load metrics
    metrics = dict(f["metrics"].attrs)


# print every dict
print("Experiment Arguments:")
for k, v in experiment_args.items():
    print(f"  {k}: {v}")

print("\nCEM Options:")
for k, v in cem_options.items():
    print(f"  {k}: {v}")

print("\nMetrics:")
for k, v in metrics.items():
    print(f"  {k}: {v}")

print("\nTrajectory Data:")
print(f"  sim_dt: {trajectory['sim_dt']}")
print(f"  ctrl_dt: {trajectory['ctrl_dt']}")
print(f"  qpos shape: {trajectory['qpos'].shape}")
print(f"  qvel shape: {trajectory['qvel'].shape}")
print(f"  ctrl shape: {trajectory['ctrl'].shape}")