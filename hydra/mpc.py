import time

import mujoco
import mujoco.viewer
import numpy as np

from hydra.base import SamplingBasedController


def run_interactive(
    mj_model: mujoco.MjModel,
    controller: SamplingBasedController,
    start_state: np.ndarray,
    frequency: float,
    fixed_camera_id: int = None,
) -> None:
    """Run an interactive simulation with the MPC controller.

    Args:
        mj_model: The MuJoCo model for the system to use for simulation. Could
                  be slightly different from the model used by the controller.
        controller: The controller instance, which includes the task
                    (e.g., model, cost) definition.
        start_state: The initial state x₀ = [q₀, v₀] of the system.
        frequency: The requested control frequency (Hz) for replanning.
        fixed_camera_id: The camera ID to use for the fixed camera view.

    Note: the actual control frequency may be slightly different than what is
    requested, because the control period must be an integer multiple of the
    simulation time step.
    """
    # Set the initial state
    assert len(start_state) == mj_model.nq + mj_model.nv
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = start_state[: mj_model.nq]
    mj_data.qvel[:] = start_state[mj_model.nq :]

    # Figure out how many sim steps to run before replanning
    replan_period = 1.0 / frequency
    sim_steps_per_replan = int(replan_period / mj_model.opt.timestep)
    sim_steps_per_replan = max(sim_steps_per_replan, 1)
    step_dt = sim_steps_per_replan * mj_model.opt.timestep
    actual_frequency = 1.0 / step_dt
    print(
        f"Planning at {actual_frequency} Hz, "
        f"simulating at {1.0/mj_model.opt.timestep} Hz"
    )

    # Start the simulation
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        if fixed_camera_id is not None:
            # Set the custom camera
            viewer.cam.fixedcamid = fixed_camera_id
            viewer.cam.type = 2

        while viewer.is_running():
            start_time = time.time()

            # TODO: optimize and get a control action
            u = np.zeros((mj_model.nu,))

            # Apply the action and step the simulation
            mj_data.ctrl[:] = np.array(u)
            for _ in range(sim_steps_per_replan):
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()

            # Try to run in roughly realtime
            elapsed = time.time() - start_time
            if elapsed < step_dt:
                time.sleep(step_dt)
