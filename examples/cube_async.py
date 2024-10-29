import mujoco

from hydrax import ROOT
from hydrax.algs import PredictiveSampling
from hydrax.simulation.asynchronous import run_interactive
from hydrax.tasks.cube import CubeRotation

"""
Run an asynchronous interactive simulation of the cube rotation task.

Asynchronous simulation is more realistic than deterministic simulation, but
offers limited features (e.g., no trace visualization).
"""


# For asynchronous simulation, we need to define the task and controller inside
# a function. This ensures that all jax code is isolated in one process.
def make_controller() -> PredictiveSampling:
    """Set up the controller."""
    task = CubeRotation()
    return PredictiveSampling(
        task, num_samples=32, noise_level=0.2, num_randomizations=32
    )


# Define the model used for simulation (with more realistic parameters)
mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/cube/scene.xml")
mj_model.opt.timestep = 0.002
mj_model.opt.iterations = 100
mj_model.opt.ls_iterations = 50
mj_model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
mj_data = mujoco.MjData(mj_model)

# Run the interactive simulation
run_interactive(
    make_controller,
    mj_model,
    mj_data,
)