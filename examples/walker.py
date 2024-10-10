import sys

import mujoco
import numpy as np

from hydrax import ROOT
from hydrax.algs import MPPI, PredictiveSampling
from hydrax.mpc import run_interactive
from hydrax.tasks.walker import Walker

"""
Run an interactive simulation of the walker task.
"""

if __name__ == "__main__":
    task = Walker()

    # Set the controller based on command-line arguments
    if len(sys.argv) == 1 or sys.argv[1] == "ps":
        print("Running predictive sampling")
        ctrl = PredictiveSampling(task, num_samples=128, noise_level=0.5)
    elif sys.argv[1] == "mppi":
        print("Running MPPI")
        ctrl = MPPI(task, num_samples=128, noise_level=0.5, temperature=0.1)
    else:
        print("Usage: python walker.py [ps|mppi]")
        sys.exit(1)

    # Define the model used for simulation
    mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/walker/scene.xml")
    mj_model.opt.timestep = 0.005
    mj_model.opt.iterations = 50
    start_state = np.zeros(mj_model.nq + mj_model.nv)

    # Run the interactive simulation
    run_interactive(
        mj_model,
        ctrl,
        start_state,
        frequency=50,
        fixed_camera_id=0,
        show_traces=True,
        max_traces=1,
    )
