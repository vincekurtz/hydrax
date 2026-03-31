##
#
# Plot data from the push-T headless example.
#
##

import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt

import mujoco
import mujoco.viewer
import time

##################################################################
# FILE PATH
##################################################################

parser = argparse.ArgumentParser(
    description="Evaluate a headless push-T simulation run."
)
parser.add_argument(
    "--file",
    required=True,
    help="The name of the file to analyze."
)
parser.add_argument(
    "--viz",
    action="store_true",
    help="Whether to visualize the trajectory. (default: False)",
    required=False,
)
parser.add_argument(
    "--plot",
    action="store_true",
    help="Whether to plot the trajectory data. (default: False)",
    required=False,
)
args = parser.parse_args()

file_path = "./experiments/pusht/data/" + args.file

##################################################################
# LOAD DATA
##################################################################

with h5py.File(file_path, "r") as f:
    # Load experiment arguments
    experiment_args = dict(f["experiment_args"].attrs)

    # Load PredictiveSampling options
    ctrl_options = dict(f["ctrl_options"].attrs)

    # Load trajectory data (read datasets into numpy, attrs into scalars)
    traj_grp = f["trajectory"]
    trajectory = {k: v[:] for k, v in traj_grp.items()}
    trajectory.update(dict(traj_grp.attrs))

    # Load metrics (datasets into numpy, attrs into scalars)
    met_grp = f["metrics"]
    metrics = {k: v[:] for k, v in met_grp.items()}
    metrics.update(dict(met_grp.attrs))

print(f"\nLoaded data from [{file_path}].")

# print every dict
print("\nExperiment Arguments:")
for k, v in experiment_args.items():
    print(f"  {k}: {v}")

print("\nPredictiveSampling Options:")
for k, v in ctrl_options.items():
    print(f"  {k}: {v}")

print("\nMetrics:")
total_wall_time = metrics["total_wall_time"]
position_cost = metrics["position_cost"]
orientation_cost = metrics["orientation_cost"]
close_to_block_cost = metrics["close_to_block_cost"]
total_cost = metrics["total_cost"]
print(f"  total_wall_time: {total_wall_time:.3f} seconds")
print(f"  position_cost, shape={position_cost.shape} array, mean={position_cost.mean():.3f}")
print(f"  orientation_cost, shape={orientation_cost.shape} array, mean={orientation_cost.mean():.3f}")
print(f"  close_to_block_cost, shape={close_to_block_cost.shape} array, mean={close_to_block_cost.mean():.3f}")
print(f"  total_cost, shape={total_cost.shape} array, mean={total_cost.mean():.3f}, std={total_cost.std():.3f}")
costs = {
    "position_cost": position_cost,
    "orientation_cost": orientation_cost,
    "close_to_block_cost": close_to_block_cost,
    "total_cost": total_cost,
}

print("\nTrajectory Data:")
sim_dt = trajectory["sim_dt"]
ctrl_dt = trajectory["ctrl_dt"]
qpos = trajectory["qpos"]
qvel = trajectory["qvel"]
ctrl = trajectory["ctrl"]
print(f"  sim_dt: {sim_dt}")
print(f"  ctrl_dt: {ctrl_dt}")
print(f"  qpos shape: {qpos.shape}")
print(f"  qvel shape: {qvel.shape}")
print(f"  ctrl shape: {ctrl.shape}")

##################################################################
# PLAYBACK and PLOTTING
##################################################################

if args.viz:

    # load push-T model and create data
    xml_path = "./hydrax/models/pusht/scene.xml"
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    # trajectory data
    qpos = trajectory["qpos"]
    sim_dt = trajectory["sim_dt"]
    n_frames = qpos.shape[0]
    times = np.arange(n_frames) * sim_dt

    # launch the viewer
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        t0 = time.time()
        while viewer.is_running():
            elapsed = time.time() - t0
            i = np.searchsorted(times, elapsed)
            i = min(i, n_frames - 1)

            # set qpos and forward kinematics
            mj_data.qpos[:] = qpos[i]
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()

            # loop back to start when done
            if elapsed > times[-1]:
                time.sleep(1.0)
                t0 = time.time()


if args.plot:

    # create time array for costs
    sim_time = np.arange(costs["total_cost"].shape[0]) * sim_dt

    # plot the costs
    plt.figure(figsize=(12, 8))
    for label, cost in costs.items():
        plt.plot(sim_time, cost, label=label)
    plt.xlabel("Time (s)")
    plt.ylabel("Cost")
    plt.title("Costs over time")
    plt.legend()
    plt.grid()
    plt.show()
