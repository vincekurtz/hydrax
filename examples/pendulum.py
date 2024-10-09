import mujoco
import numpy as np

from hydrax import ROOT
from hydrax.algs.predictive_sampling import PredictiveSampling
from hydrax.mpc import run_interactive
from hydrax.tasks.pendulum import Pendulum

"""
Run an interactive simulation of the pendulum swingup task.
"""

if __name__ == "__main__":
    # Define the controller
    task = Pendulum()
    ctrl = PredictiveSampling(task, num_samples=16, noise_level=0.1)

    # Define the model used for simulation
    mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/pendulum/scene.xml")
    start_state = np.array([0.0, 0.0])

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
