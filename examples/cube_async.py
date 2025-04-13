import mujoco

from hydrax.algs import PredictiveSampling
from hydrax.simulation.asynchronous import run_interactive
from hydrax.tasks.cube import CubeRotation

"""
Run an asynchronous interactive simulation of the cube rotation task.

Asynchronous simulation is more realistic than deterministic simulation, but
offers limited features (e.g., no trace visualization).
"""

# Asynchronous simulations must be wrapped in a __main__ block, see
# https://docs.python.org/3/library/multiprocessing.html
if __name__ == "__main__":
    # Define the task and controller
    task = CubeRotation()
    ctrl = PredictiveSampling(
        task,
        num_samples=32,
        noise_level=0.2,
        num_randomizations=32,
        plan_horizon=0.12,
        spline_type="zero",
        num_knots=4,
    )

    # Define the model used for simulation (with more realistic parameters)
    mj_model = task.mj_model
    mj_model.opt.timestep = 0.005
    mj_model.opt.iterations = 100
    mj_model.opt.ls_iterations = 50
    mj_model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    mj_data = mujoco.MjData(mj_model)

    # Run the interactive simulation
    run_interactive(
        ctrl,
        mj_model,
        mj_data,
    )
