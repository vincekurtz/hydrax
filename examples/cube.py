import sys

import evosax
import mujoco

from hydrax import ROOT
from hydrax.algs import CEM, MPPI, Evosax, PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.cube import CubeRotation

"""
Run an interactive simulation of the cube rotation task.

Double click on the floating target cube, then change the goal orientation with
[ctrl + left click].
"""

# Define the task (cost and dynamics)
task = CubeRotation()

# Set the controller based on command-line arguments
if len(sys.argv) == 1 or sys.argv[1] == "ps":
    print("Running predictive sampling")
    ctrl = PredictiveSampling(
        task, num_samples=32, noise_level=0.2, num_randomizations=32
    )
elif sys.argv[1] == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task,
        num_samples=128,
        noise_level=0.2,
        temperature=0.001,
        num_randomizations=8,
    )
elif sys.argv[1] == "cem":
    print("Running CEM")
    ctrl = CEM(
        task,
        num_samples=128,
        num_elites=5,
        sigma_start=0.5,
        sigma_min=0.5,
        num_randomizations=8,
    )
elif sys.argv[1] == "cmaes":
    print("Running CMA-ES")
    ctrl = Evosax(
        task,
        evosax.Sep_CMA_ES,
        num_samples=128,
        elite_ratio=0.5,
        num_randomizations=8,
    )
else:
    print("Usage: python cube.py [ps|mppi|cem|cmaes]")
    sys.exit(1)

# Define the model used for simulation
mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/cube/scene.xml")
mj_data = mujoco.MjData(mj_model)

# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=25,
    fixed_camera_id=None,
    show_traces=True,
    max_traces=1,
    trace_color=[1.0, 1.0, 1.0, 1.0],
)
