import mujoco
import numpy as np

from hydra import ROOT
from hydra.algs.predictive_sampling import PredictiveSampling
from hydra.mpc import run_interactive
from hydra.tasks.walker import Walker

"""
Run an interactive simulation of the walker task.
"""

if __name__ == "__main__":
    # Define the controller
    task = Walker()
    ctrl = PredictiveSampling(task, num_samples=128, noise_level=0.2)

    # Define the model used for simulation
    mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/walker/scene.xml")
    mj_model.opt.timestep = 0.005
    start_state = np.zeros(mj_model.nq + mj_model.nv)

    # Run the interactive simulation
    run_interactive(
        mj_model,
        ctrl,
        start_state,
        frequency=50,
        fixed_camera_id=None,
        show_traces=True,
        max_traces=1,
    )
