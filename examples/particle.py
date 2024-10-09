import mujoco
import numpy as np

from hydra import ROOT
from hydra.algs.predictive_sampling import PredictiveSampling
from hydra.mpc import run_interactive
from hydra.tasks.particle import Particle

"""
Run an interactive simulation of the particle tracking task.

Double click on the green target, then drag it around with [ctrl + right-click].
"""

if __name__ == "__main__":
    # Define the controller
    task = Particle()
    ctrl = PredictiveSampling(task, num_samples=16, noise_level=0.05)

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
