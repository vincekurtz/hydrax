import sys

import mujoco
import numpy as np

from hydrax import ROOT
from hydrax.algs import MPPI, PredictiveSampling
from hydrax.mpc import run_interactive
from hydrax.tasks.particle import Particle

"""
Run an interactive simulation of the particle tracking task.

Double click on the green target, then drag it around with [ctrl + right-click].
"""

if __name__ == "__main__":
    task = Particle()

    # Set the controller based on command-line arguments
    if len(sys.argv) == 1 or sys.argv[1] == "ps":
        print("Running predictive sampling")
        ctrl = PredictiveSampling(task, num_samples=16, noise_level=0.1)
    elif sys.argv[1] == "mppi":
        print("Running MPPI")
        ctrl = MPPI(task, num_samples=16, noise_level=0.3, temperature=0.01)
    else:
        print("Usage: python particle.py [ps|mppi]")
        sys.exit(1)

    # Define the model used for simulation
    mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/particle/scene.xml")
    start_state = np.array([0.0, 0.0, 0.0, 0.0])

    # Run the interactive simulation
    run_interactive(
        mj_model,
        ctrl,
        start_state,
        frequency=50,
        show_traces=True,
        max_traces=5,
    )
