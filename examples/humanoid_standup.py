import argparse
from copy import deepcopy

import mujoco

from hydrax.algs import MPPI
from hydrax.simulation.asynchronous import run_interactive as run_async
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.humanoid_standup import HumanoidStandup

"""
Run an interactive simulation of the humanoid standup task.
"""

# Need to be wrapped in main loop for async simulation
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run an interactive simulation of humanoid (G1) standup."
    )
    parser.add_argument(
        "-a",
        "--asynchronous",
        action="store_true",
        help="Use asynchronous simulation",
        default=False,
    )
    parser.add_argument(
        "--warp",
        action="store_true",
        help="Whether to use the (experimental) MjWarp backend.",
        required=False,
    )
    args = parser.parse_args()

    # Define the task (cost and dynamics)
    task = HumanoidStandup(impl="warp" if args.warp else "jax")

    # Set up the controller
    ctrl = MPPI(
        task,
        num_samples=128,
        noise_level=0.3,
        temperature=0.1,
        num_randomizations=4,
        plan_horizon=0.6,
        spline_type="zero",
        num_knots=4,
    )

    # Define the model used for simulation (stiffer contact parameters)
    mj_model = deepcopy(task.mj_model)
    mj_model.opt.timestep = 0.01
    mj_model.opt.o_solimp = [0.9, 0.95, 0.001, 0.5, 2]
    mj_model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_OVERRIDE

    # Set the initial state so the robot falls and needs to stand back up
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = mj_model.keyframe("stand").qpos
    mj_data.qpos[3:7] = [0.7, 0.0, -0.7, 0.0]

    # Run the interactive simulation
    if args.asynchronous:
        print("Running asynchronous simulation")

        # Tighten up the simulator parameters, since it's running on CPU and
        # therefore won't slow down the planner
        mj_model.opt.timestep = 0.005
        mj_model.opt.iterations = 100
        mj_model.opt.ls_iterations = 50
        mj_model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC

        run_async(
            ctrl,
            mj_model,
            mj_data,
        )
    else:
        print("Running deterministic simulation")
        run_interactive(
            ctrl,
            mj_model,
            mj_data,
            frequency=50,
            show_traces=False,
        )
