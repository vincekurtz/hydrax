#!/usr/bin/env python

import time

import mujoco
import mujoco.viewer
import numpy as np
from huggingface_hub import hf_hub_download

from hydrax import ROOT

##
#
# Visualize mocap-based targets from teh LAFAN1 dataset, retargeted to the
# Unitree G1 model.
#
##

# Load the sequence of configurations (30 FPS) from huggingface
filename = "walk1_subject1.csv"
dataset = np.loadtxt(
    hf_hub_download(
        repo_id="unitreerobotics/LAFAN1_Retargeting_Dataset",
        filename=filename,
        subfolder="g1",
        repo_type="dataset",
    ),
    delimiter=",",
)

# Convert the dataset to mujoco format, with wxyz quaternion
pos = dataset[:, :3]
xyzw = dataset[:, 3:7]
wxyz = np.concatenate([xyzw[:, 3:], xyzw[:, :3]], axis=1)
dataset = np.concatenate([pos, wxyz, dataset[:, 7:]], axis=1)


def get_configuration(t: float) -> np.ndarray:
    """Get the configuration at time t."""
    i = int(t * 30.0)  # The dataset runs at 30 FPS
    i = min(max(0, i), dataset.shape[0] - 1)
    return dataset[i, :]


# Set up a mujoco model
mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/g1/scene.xml")
mj_data = mujoco.MjData(mj_model)

# Start the visualizer, and step through the dataset
dt = 0.01
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    t = 0.0
    while viewer.is_running():
        mj_data.qpos[:] = get_configuration(t)
        mj_data.time = t
        mujoco.mj_forward(mj_model, mj_data)
        viewer.sync()

        time.sleep(dt)
        t += dt
