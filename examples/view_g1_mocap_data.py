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
filename = "dance1_subject2.csv"
dataset = np.loadtxt(
    hf_hub_download(
        repo_id="unitreerobotics/LAFAN1_Retargeting_Dataset",
        filename=filename,
        subfolder="g1",
        repo_type="dataset",
    ),
    delimiter=",",
)


def to_mujoco(q: np.ndarray) -> np.ndarray:
    """Convert a configuration from the LAFAN1 dataset to the G1 model."""
    pos = q[:3]
    xyzw = q[3:7]
    wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
    qpos = np.concatenate([pos, wxyz, q[7:]])
    return qpos


# Set up a mujoco model
mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/g1/scene.xml")
mj_data = mujoco.MjData(mj_model)

# Set the initial state
mj_data.qpos[:] = to_mujoco(dataset[0, :])

# Start the visualizer, and step through the dataset
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    i = 0.0
    while viewer.is_running():
        mj_data.qpos[:] = to_mujoco(dataset[int(i), :])
        mj_data.time = i / 30.0
        mujoco.mj_forward(mj_model, mj_data)
        viewer.sync()

        time.sleep(1.0 / 30.0)

        i += 1
        if i >= dataset.shape[0]:
            i = 0.0
