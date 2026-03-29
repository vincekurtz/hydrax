##
#
# Plot data from the humanoid mocap headless example.
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
    description="Run a headless simulation of mocap tracking with the G1."
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

file_path = "./experiments/data/" + args.file

##################################################################
# LOAD DATA
##################################################################

with h5py.File(file_path, "r") as f:
    # Load experiment arguments
    experiment_args = dict(f["experiment_args"].attrs)

    # Load CEM options
    cem_options = dict(f["cem_options"].attrs)

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

print("\nCEM Options:")
for k, v in cem_options.items():
    print(f"  {k}: {v}")

print("\nMetrics:")
total_wall_time = metrics["total_wall_time"]
r_anchor_pos = metrics["r_anchor_pos"]
r_anchor_ori = metrics["r_anchor_ori"]
r_body_pos = metrics["r_body_pos"]
r_body_ori = metrics["r_body_ori"]
r_body_lin_vel = metrics["r_body_lin_vel"]
r_body_ang_vel = metrics["r_body_ang_vel"]
total_reward = metrics["total_reward"]
print(f"  total_wall_time: {total_wall_time:.3f} seconds")
print(f"  r_anchor_pos, shape={r_anchor_pos.shape} array, mean={r_anchor_pos.mean():.3f}, std={r_anchor_pos.std():.3f}")
print(f"  r_anchor_ori, shape={r_anchor_ori.shape} array, mean={r_anchor_ori.mean():.3f}, std={r_anchor_ori.std():.3f}")
print(f"  r_body_pos, shape={r_body_pos.shape} array, mean={r_body_pos.mean():.3f}, std={r_body_pos.std():.3f}")
print(f"  r_body_ori, shape={r_body_ori.shape} array, mean={r_body_ori.mean():.3f}, std={r_body_ori.std():.3f}")
print(f"  r_body_lin_vel, shape={r_body_lin_vel.shape} array, mean={r_body_lin_vel.mean():.3f}, std={r_body_lin_vel.std():.3f}")
print(f"  r_body_ang_vel, shape={r_body_ang_vel.shape} array, mean={r_body_ang_vel.mean():.3f}, std={r_body_ang_vel.std():.3f}")
print(f"  total_reward, shape={total_reward.shape} array, mean={total_reward.mean():.3f}, std={total_reward.std():.3f}")
rewards = {
    "r_anchor_pos": r_anchor_pos,
    "r_anchor_ori": r_anchor_ori,
    "r_body_pos": r_body_pos,
    "r_body_ori": r_body_ori,
    "r_body_lin_vel": r_body_lin_vel,
    "r_body_ang_vel": r_body_ang_vel,
    "total_reward": total_reward,
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

    # load G1 model and create data
    xml_path = "./hydrax/models/g1/scene.xml"
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

    # create time array for rewards
    sim_time =np.arange(rewards["total_reward"].shape[0]) * sim_dt

    # plot the rewards
    plt.figure(figsize=(12, 8))
    for label, reward in rewards.items():
        plt.plot(sim_time, reward, label=label)
    plt.xlabel("Time (s)")
    plt.ylabel("Reward")
    plt.title("Rewards over time")
    plt.legend()
    plt.grid()
    plt.show()
