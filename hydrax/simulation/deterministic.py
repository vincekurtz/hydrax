import os
import time
from typing import Sequence

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import mjx

from hydrax import ROOT
from hydrax.alg_base import SamplingBasedController
from hydrax.utils.video import VideoRecorder

"""
Tools for deterministic (synchronous) simulation, with the simulator and
controller running one after the other in the same thread.
"""


def run_interactive(  # noqa: PLR0912, PLR0915
    controller: SamplingBasedController,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    frequency: float,
    initial_knots: jax.Array = None,
    fixed_camera_id: int = None,
    show_traces: bool = True,
    max_traces: int = 5,
    trace_width: float = 5.0,
    trace_color: Sequence = [1.0, 1.0, 1.0, 0.1],
    reference: np.ndarray = None,
    reference_fps: float = 30.0,
    record_video: bool = False,
) -> None:
    """Run an interactive simulation with the MPC controller.

    This is a deterministic simulation, with the controller and simulation
    running in the same thread. This is useful for repeatability, but is less
    realistic than asynchronous simulation.

    Note: the actual control frequency may be slightly different than what is
    requested, because the control period must be an integer multiple of the
    simulation time step.

    Args:
        controller: The controller instance, which includes the task
                    (e.g., model, cost) definition.
        mj_model: The MuJoCo model for the system to use for simulation. Could
                  be slightly different from the model used by the controller.
        mj_data: A MuJoCo data object containing the initial system state.
        frequency: The requested control frequency (Hz) for replanning.
        initial_knots: The initial knot points for the control spline at t=0
        fixed_camera_id: The camera ID to use for the fixed camera view.
        show_traces: Whether to show traces for the site positions.
        max_traces: The maximum number of traces to show at once.
        trace_width: The width of the trace lines (in pixels).
        trace_color: The RGBA color of the trace lines.
        reference: The reference trajectory (qs) to visualize.
        reference_fps: The frame rate of the reference trajectory.
        record_video: Whether to record a video of the simulation.
    """
    # Report the planning horizon in seconds for debugging
    print(
        f"Planning with {controller.ctrl_steps} steps "
        f"over a {controller.plan_horizon} second horizon "
        f"with {controller.num_knots} knots."
    )

    # Figure out how many sim steps to run before replanning
    replan_period = 1.0 / frequency
    sim_steps_per_replan = int(replan_period / mj_model.opt.timestep)
    sim_steps_per_replan = max(sim_steps_per_replan, 1)
    step_dt = sim_steps_per_replan * mj_model.opt.timestep
    actual_frequency = 1.0 / step_dt
    print(
        f"Planning at {actual_frequency} Hz, "
        f"simulating at {1.0 / mj_model.opt.timestep} Hz"
    )

    # Create a data structure for the controller to run rollouts from.
    mjx_data = controller.task.make_data()
    mjx_data = mjx_data.replace(
        qpos=mj_data.qpos,
        qvel=mj_data.qvel,
        mocap_pos=mj_data.mocap_pos,
        mocap_quat=mj_data.mocap_quat,
    )

    # Initialize the controller
    policy_params = controller.init_params(initial_knots=initial_knots)
    jit_optimize = jax.jit(controller.optimize)
    jit_interp_func = jax.jit(controller.interp_func)

    # Warm-up the controller
    print("Jitting the controller...")
    st = time.time()
    policy_params, rollouts = jit_optimize(mjx_data, policy_params)
    policy_params, rollouts = jit_optimize(mjx_data, policy_params)

    tq = jnp.arange(0, sim_steps_per_replan) * mj_model.opt.timestep
    tk = policy_params.tk
    knots = policy_params.mean[None, ...]
    _ = jit_interp_func(tq, tk, knots)
    _ = jit_interp_func(tq, tk, knots)
    print(f"Time to jit: {time.time() - st:.3f} seconds")
    num_traces = min(rollouts.controls.shape[1], max_traces)

    # Ghost reference setup
    if reference is not None:
        ref_data = mujoco.MjData(mj_model)
        assert reference.shape[1] == mj_model.nq
        ref_data.qpos[:] = reference[0, :]
        mujoco.mj_forward(mj_model, ref_data)

        vopt = mujoco.MjvOption()
        vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True  # Transparent.
        pert = mujoco.MjvPerturb()
        catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC  # only show dynamic bodies

    # Initialize video recording if enabled
    recorder = None
    if record_video:
        # Video dimensions
        width, height = 720, 480
        # Create the video recorder
        recorder = VideoRecorder(
            output_dir=os.path.join(ROOT, "recordings"),
            width=width,
            height=height,
            fps=actual_frequency,
        )
        # Ensure model visual offscreen buffer is compatible with video
        # recording
        mj_model.vis.global_.offwidth = width
        mj_model.vis.global_.offheight = height
        if not recorder.start():
            record_video = False
        renderer = mujoco.Renderer(mj_model, height=height, width=width)

    # Start the simulation
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        if fixed_camera_id is not None:
            # Set the custom camera
            viewer.cam.fixedcamid = fixed_camera_id
            viewer.cam.type = 2

        # Set up rollout traces
        if show_traces:
            num_trace_sites = len(controller.task.trace_site_ids)
            for i in range(
                num_trace_sites * num_traces * controller.ctrl_steps
            ):
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[i],
                    type=mujoco.mjtGeom.mjGEOM_LINE,
                    size=np.zeros(3),
                    pos=np.zeros(3),
                    mat=np.eye(3).flatten(),
                    rgba=np.array(trace_color),
                )
                viewer.user_scn.ngeom += 1

        # Add geometry for the ghost reference
        if reference is not None:
            mujoco.mjv_addGeoms(
                mj_model, ref_data, vopt, pert, catmask, viewer.user_scn
            )

        while viewer.is_running():
            start_time = time.time()

            # Set the start state for the controller
            mjx_data = mjx_data.replace(
                qpos=jnp.array(mj_data.qpos),
                qvel=jnp.array(mj_data.qvel),
                mocap_pos=jnp.array(mj_data.mocap_pos),
                mocap_quat=jnp.array(mj_data.mocap_quat),
                time=mj_data.time,
            )

            # Do a replanning step
            plan_start = time.time()
            policy_params, rollouts = jit_optimize(mjx_data, policy_params)
            plan_time = time.time() - plan_start

            # Visualize the rollouts
            if show_traces:
                ii = 0
                for k in range(num_trace_sites):
                    for i in range(num_traces):
                        for j in range(controller.ctrl_steps):
                            mujoco.mjv_connector(
                                viewer.user_scn.geoms[ii],
                                mujoco.mjtGeom.mjGEOM_LINE,
                                trace_width,
                                rollouts.trace_sites[i, j, k],
                                rollouts.trace_sites[i, j + 1, k],
                            )
                            ii += 1

            # Update the ghost reference
            if reference is not None:
                t_ref = mj_data.time * reference_fps
                i_ref = int(t_ref)
                i_ref = min(i_ref, reference.shape[0] - 1)
                ref_data.qpos[:] = reference[i_ref]
                mujoco.mj_forward(mj_model, ref_data)
                mujoco.mjv_updateScene(
                    mj_model,
                    ref_data,
                    vopt,
                    pert,
                    viewer.cam,
                    catmask,
                    viewer.user_scn,
                )

            # query the control spline at the sim frequency
            # (we assume the sim freq is the same as the low-level ctrl freq)
            sim_dt = mj_model.opt.timestep
            t_curr = mj_data.time

            tq = jnp.arange(0, sim_steps_per_replan) * sim_dt + t_curr
            tk = policy_params.tk
            knots = policy_params.mean[None, ...]
            us = np.asarray(jit_interp_func(tq, tk, knots))[0]  # (ss, nu)

            # simulate the system between spline replanning steps
            for i in range(sim_steps_per_replan):
                mj_data.ctrl[:] = np.array(us[i])
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()

                # Capture frame if recording
                if record_video and recorder.is_recording:
                    renderer.update_scene(mj_data, viewer.cam)
                    frame = renderer.render()
                    recorder.add_frame(frame.tobytes())

            # Try to run in roughly realtime
            elapsed = time.time() - start_time
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)

            # Print some timing information
            rtr = step_dt / (time.time() - start_time)
            print(
                f"Realtime rate: {rtr:.2f}, plan time: {plan_time:.4f}s",
                end="\r",
            )

    # Preserve the last printout
    print("")

    # Close the video recorder if recording was enabled
    if record_video and recorder is not None:
        recorder.stop()


###################################################################################
# HUMANOID MOCAP
###################################################################################

def run_headless_humanoid_mocap(  # noqa: PLR0912, PLR0915
    controller: SamplingBasedController,
    mjx_model_sim: mjx.Model,
    mjx_data_sim: mjx.Data,
    frequency: float,
    duration: float,
    initial_knots: jax.Array = None,
) -> dict:
    """Run a headless simulation with the MPC controller.

    This is a deterministic simulation without visualization, suitable for
    headless/remote systems or batch experiments. The controller and simulation
    run in the same thread. Everything runs on device (no CPU-GPU transfers).

    Note: the actual control frequency may be slightly different than what is
    requested, because the control period must be an integer multiple of the
    simulation time step.

    Args:
        controller: The controller instance, which includes the task
                    (e.g., model, cost) definition.
        mjx_model_sim: The MJX model for simulation. Could be slightly
                       different from the model used by the controller.
        mjx_data_sim: An MJX data object containing the initial system state.
        frequency: The requested control frequency (Hz) for replanning.
        duration: How long to run the simulation (seconds).
        initial_knots: The initial knot points for the control spline at t=0

    Returns:
        A results dictionary with two sub-dictionaries:
            - "trajectory"
            - "metrics"
    """
    # Report the planning horizon in seconds for debugging
    print(
        f"Planning with {controller.ctrl_steps} steps "
        f"over a {controller.plan_horizon} second horizon "
        f"with {controller.num_knots} knots."
    )

    # Figure out how many sim steps to run before replanning
    sim_dt = mjx_model_sim.opt.timestep
    replan_period = 1.0 / frequency
    sim_steps_per_replan = round(replan_period / sim_dt)
    sim_steps_per_replan = max(sim_steps_per_replan, 1)
    step_dt = sim_steps_per_replan * sim_dt
    actual_frequency = 1.0 / step_dt
    print(
        f"Planning at {actual_frequency} Hz, "
        f"simulating at {1.0 / sim_dt} Hz"
    )

    # Compute total number of sim steps
    num_replan_steps = round(duration / step_dt)
    total_sim_steps = num_replan_steps * sim_steps_per_replan

    # preallocate metric arrays
    nq = mjx_data_sim.qpos.shape[0]
    nv = mjx_data_sim.qvel.shape[0]
    nu = mjx_data_sim.ctrl.shape[0]
    qpos_hist = np.full((total_sim_steps, nq), np.nan)
    qvel_hist = np.full((total_sim_steps, nv), np.nan)
    ctrl_hist = np.full((total_sim_steps, nu), np.nan)

    # Create a data structure for the controller to run rollouts from
    mjx_data = controller.task.make_data()
    mjx_data = mjx_data.replace(
        qpos=mjx_data_sim.qpos,
        qvel=mjx_data_sim.qvel,
        mocap_pos=mjx_data_sim.mocap_pos,
        mocap_quat=mjx_data_sim.mocap_quat,
    )

    # Initialize the controller
    policy_params = controller.init_params(initial_knots=initial_knots)
    jit_optimize = jax.jit(controller.optimize)
    jit_interp_func = jax.jit(controller.interp_func)

    # JIT-compile the simulation stepping function.
    # Uses lax.scan to step mjx_data_sim forward sim_steps_per_replan times
    # and applies the interpolated controls at each sub-step.
    @jax.jit
    def sim_steps(mjx_data_sim, us):
        def step_fn(data, u):
            data = data.replace(ctrl=u)
            data = mjx.step(mjx_model_sim, data)
            metrics = controller.task.compute_metrics(data, u)
            return data, (data.qpos, data.qvel, metrics)
        mjx_data_sim, (qpos_seg, qvel_seg, metrics_seg) = jax.lax.scan(
            step_fn, mjx_data_sim, us
        )
        return mjx_data_sim, qpos_seg, qvel_seg, metrics_seg

    # Warm-up the controller
    print("Jitting...")
    st = time.time()
    policy_params, _ = jit_optimize(mjx_data, policy_params)
    policy_params, _ = jit_optimize(mjx_data, policy_params)

    tq = jnp.arange(0, sim_steps_per_replan) * sim_dt
    tk = policy_params.tk
    knots = policy_params.mean[None, ...]
    _ = jit_interp_func(tq, tk, knots)
    _ = jit_interp_func(tq, tk, knots)

    # Warm-up the simulation stepping function
    us_warmup = jit_interp_func(tq, tk, knots)[0]
    warmup_result = sim_steps(mjx_data_sim, us_warmup)
    _ = sim_steps(mjx_data_sim, us_warmup)

    # Preallocate metrics arrays from warmup result keys
    metrics_seg_warmup = warmup_result[3]
    metrics_hist = {
        k: np.full((total_sim_steps,), np.nan)
        for k in metrics_seg_warmup.keys()
    }
    print(f"Time to jit: {time.time() - st:.3f} seconds")

    # Fall detection parameters
    fall_threshold = 0.3  # base z-position below this means the robot fell
    base_z_index = 2      # index of z-position in qpos (standard free joint)
    failure = 0
    termination_time = duration  # default: survived the whole trajectory
    actual_sim_steps = total_sim_steps

    # Run the simulation (all on device, no CPU-GPU transfers in the loop)
    sim_wall_start = time.time()
    for step_idx in range(num_replan_steps):
        step_start_time = time.time()

        # Set the start state for the controller from the sim state
        mjx_data = mjx_data.replace(
            qpos=mjx_data_sim.qpos,
            qvel=mjx_data_sim.qvel,
            mocap_pos=mjx_data_sim.mocap_pos,
            mocap_quat=mjx_data_sim.mocap_quat,
            time=mjx_data_sim.time,
        )

        # Do a replanning step
        plan_start = time.time()
        policy_params, _ = jit_optimize(mjx_data, policy_params)
        plan_time = time.time() - plan_start

        # query the control spline at the sim frequency
        # (we assume the sim freq is the same as the low-level ctrl freq)
        tq = jnp.arange(0, sim_steps_per_replan) * sim_dt + mjx_data_sim.time
        tk = policy_params.tk
        knots = policy_params.mean[None, ...]
        us = jit_interp_func(tq, tk, knots)[0]  # (ss, nu) — stays on device

        # simulate the system between spline replanning steps (all on device)
        mjx_data_sim, qpos_seg, qvel_seg, metrics_seg = sim_steps(mjx_data_sim, us)

        # Record trajectory segment
        seg_start = step_idx * sim_steps_per_replan
        seg_end = seg_start + sim_steps_per_replan
        qpos_seg_np = np.asarray(qpos_seg)
        qvel_seg_np = np.asarray(qvel_seg)
        qpos_hist[seg_start:seg_end] = qpos_seg_np
        qvel_hist[seg_start:seg_end] = qvel_seg_np
        ctrl_hist[seg_start:seg_end] = np.asarray(us)

        # Record metrics segment
        for k, v in metrics_seg.items():
            metrics_hist[k][seg_start:seg_end] = np.asarray(v)

        # Check for fall: find first sub-step where base z < threshold
        base_z = qpos_seg_np[:, base_z_index]
        fell_mask = base_z < fall_threshold
        if np.any(fell_mask):
            fell_substep = np.argmax(fell_mask)  # first True index
            actual_sim_steps = seg_start + int(fell_substep)
            termination_time = actual_sim_steps * float(sim_dt)
            failure = 1
            print(f"\nFall detected at t={termination_time:.3f}s "
                  f"(base z={base_z[fell_substep]:.3f})")
            break

        # Print some timing information
        sim_time = float(mjx_data_sim.time)
        elapsed = time.time() - step_start_time
        rtr = step_dt / elapsed
        wall_time = time.time() - sim_wall_start
        print(
            f"Realtime rate: {rtr:.2f}, plan time: {plan_time:.4f}s"
            f", Sim time: {sim_time:.3f}s"
            f", Wall time: {wall_time:.3f}s",
            end="\r",
        )

    sim_wall_time = time.time() - sim_wall_start

    # Truncate histories to actual valid data
    qpos_hist = qpos_hist[:actual_sim_steps]
    qvel_hist = qvel_hist[:actual_sim_steps]
    ctrl_hist = ctrl_hist[:actual_sim_steps]
    for k in metrics_hist:
        metrics_hist[k] = metrics_hist[k][:actual_sim_steps]

    # total sim and wall time
    print("")
    print(f"Simulation finished.")
    print(f"Total sim time: {termination_time:.3f}s")
    print(f"Total wall time: {sim_wall_time:.3f}s")

    # make the trajectory and metrics dictionaries for return
    trajectory = {
        "sim_dt": round(float(sim_dt), 6),
        "ctrl_dt": round(float(step_dt), 6),
        "qpos": qpos_hist,
        "qvel": qvel_hist,
        "ctrl": ctrl_hist,
    }
    metrics = {
        "total_wall_time": sim_wall_time,
        "termination_time": termination_time,
        "failure": failure,
        **metrics_hist,
    }

    # results dictionary
    results = {
        "trajectory": trajectory,
        "metrics": metrics,
    }

    return results


###################################################################################
# PUSH-T
###################################################################################

def run_headless_pusht(  # noqa: PLR0912, PLR0915
    controller: SamplingBasedController,
    mjx_model_sim: mjx.Model,
    mjx_data_sim: mjx.Data,
    frequency: float,
    duration: float,
    sim_seed: int = 0,
    initial_knots: jax.Array = None,
) -> dict:
    """Run a headless simulation of the push-T task.

    This is a deterministic simulation without visualization, suitable for
    headless/remote systems or batch experiments. The controller and simulation
    run in the same thread. Everything runs on device (no CPU-GPU transfers).

    Note: the actual control frequency may be slightly different than what is
    requested, because the control period must be an integer multiple of the
    simulation time step.

    Args:
        controller: The controller instance, which includes the task
                    (e.g., model, cost) definition.
        mjx_model_sim: The MJX model for simulation. Could be slightly
                       different from the model used by the controller.
        mjx_data_sim: An MJX data object containing the initial system state.
        frequency: The requested control frequency (Hz) for replanning.
        duration: How long to run the simulation (seconds).
        sim_seed: Random seed for domain-randomizing the simulation model.
        initial_knots: The initial knot points for the control spline at t=0

    Returns:
        A results dictionary with two sub-dictionaries:
            - "trajectory"
            - "metrics"
    """
    # Sample initial condition from sim_seed via rejection sampling
    q0 = controller.task.sample_initial_position(sim_seed)
    mjx_data_sim = mjx_data_sim.replace(qpos=q0)

    # Domain-randomize the simulation model
    sim_rng = jax.random.key(sim_seed)
    dr_params = controller.task.domain_randomize_model(sim_rng)
    mjx_model_sim = mjx_model_sim.tree_replace(dr_params)
    
    # Report the planning horizon in seconds for debugging
    print(
        f"Planning with {controller.ctrl_steps} steps "
        f"over a {controller.plan_horizon} second horizon "
        f"with {controller.num_knots} knots."
    )

    # Figure out how many sim steps to run before replanning
    sim_dt = mjx_model_sim.opt.timestep
    replan_period = 1.0 / frequency
    sim_steps_per_replan = round(replan_period / sim_dt)
    sim_steps_per_replan = max(sim_steps_per_replan, 1)
    step_dt = sim_steps_per_replan * sim_dt
    actual_frequency = 1.0 / step_dt
    print(
        f"Planning at {actual_frequency} Hz, "
        f"simulating at {1.0 / sim_dt} Hz"
    )

    # Compute total number of sim steps
    num_replan_steps = round(duration / step_dt)
    total_sim_steps = num_replan_steps * sim_steps_per_replan

    # preallocate metric arrays
    nq = mjx_data_sim.qpos.shape[0]
    nv = mjx_data_sim.qvel.shape[0]
    nu = mjx_data_sim.ctrl.shape[0]
    qpos_hist = np.full((total_sim_steps, nq), np.nan)
    qvel_hist = np.full((total_sim_steps, nv), np.nan)
    ctrl_hist = np.full((total_sim_steps, nu), np.nan)

    # Create a data structure for the controller to run rollouts from
    mjx_data = controller.task.make_data()
    mjx_data = mjx_data.replace(
        qpos=mjx_data_sim.qpos,
        qvel=mjx_data_sim.qvel,
        mocap_pos=mjx_data_sim.mocap_pos,
        mocap_quat=mjx_data_sim.mocap_quat,
    )

    # Initialize the controller
    policy_params = controller.init_params(initial_knots=initial_knots)
    jit_optimize = jax.jit(controller.optimize)
    jit_interp_func = jax.jit(controller.interp_func)

    # JIT-compile the simulation stepping function.
    # Uses lax.scan to step mjx_data_sim forward sim_steps_per_replan times
    # and applies the interpolated controls at each sub-step.
    @jax.jit
    def sim_steps(mjx_data_sim, us):
        def step_fn(data, u):
            data = data.replace(ctrl=u)
            data = mjx.step(mjx_model_sim, data)
            metrics = controller.task.compute_metrics(data, u)
            return data, (data.qpos, data.qvel, metrics)
        mjx_data_sim, (qpos_seg, qvel_seg, metrics_seg) = jax.lax.scan(
            step_fn, mjx_data_sim, us
        )
        return mjx_data_sim, qpos_seg, qvel_seg, metrics_seg

    # Warm-up the controller
    print("Jitting...")
    st = time.time()
    policy_params, _ = jit_optimize(mjx_data, policy_params)
    policy_params, _ = jit_optimize(mjx_data, policy_params)

    tq = jnp.arange(0, sim_steps_per_replan) * sim_dt
    tk = policy_params.tk
    knots = policy_params.mean[None, ...]
    _ = jit_interp_func(tq, tk, knots)
    _ = jit_interp_func(tq, tk, knots)

    # Warm-up the simulation stepping function
    us_warmup = jit_interp_func(tq, tk, knots)[0]
    warmup_result = sim_steps(mjx_data_sim, us_warmup)
    _ = sim_steps(mjx_data_sim, us_warmup)

    # Preallocate metrics arrays from warmup result keys
    metrics_seg_warmup = warmup_result[3]
    metrics_hist = {
        k: np.full((total_sim_steps,), np.nan)
        for k in metrics_seg_warmup.keys()
    }
    print(f"Time to jit: {time.time() - st:.3f} seconds")

    # Run the simulation (all on device, no CPU-GPU transfers in the loop)
    sim_wall_start = time.time()
    for step_idx in range(num_replan_steps):
        step_start_time = time.time()

        # Set the start state for the controller from the sim state
        mjx_data = mjx_data.replace(
            qpos=mjx_data_sim.qpos,
            qvel=mjx_data_sim.qvel,
            mocap_pos=mjx_data_sim.mocap_pos,
            mocap_quat=mjx_data_sim.mocap_quat,
            time=mjx_data_sim.time,
        )

        # Do a replanning step
        plan_start = time.time()
        policy_params, _ = jit_optimize(mjx_data, policy_params)
        plan_time = time.time() - plan_start

        # query the control spline at the sim frequency
        tq = jnp.arange(0, sim_steps_per_replan) * sim_dt + mjx_data_sim.time
        tk = policy_params.tk
        knots = policy_params.mean[None, ...]
        us = jit_interp_func(tq, tk, knots)[0]  # (ss, nu) — stays on device

        # simulate the system between spline replanning steps (all on device)
        mjx_data_sim, qpos_seg, qvel_seg, metrics_seg = sim_steps(mjx_data_sim, us)

        # Record trajectory segment
        seg_start = step_idx * sim_steps_per_replan
        seg_end = seg_start + sim_steps_per_replan
        qpos_hist[seg_start:seg_end] = np.asarray(qpos_seg)
        qvel_hist[seg_start:seg_end] = np.asarray(qvel_seg)
        ctrl_hist[seg_start:seg_end] = np.asarray(us)

        # Record metrics segment
        for k, v in metrics_seg.items():
            metrics_hist[k][seg_start:seg_end] = np.asarray(v)

        # Print some timing information
        sim_time = float(mjx_data_sim.time)
        elapsed = time.time() - step_start_time
        rtr = step_dt / elapsed
        wall_time = time.time() - sim_wall_start
        print(
            f"Realtime rate: {rtr:.2f}, plan time: {plan_time:.4f}s"
            f", Sim time: {sim_time:.3f}s"
            f", Wall time: {wall_time:.3f}s",
            end="\r",
        )

    sim_wall_time = time.time() - sim_wall_start

    # total sim and wall time
    print("")
    print(f"Simulation finished.")
    print(f"Total sim time: {duration:.3f}s")
    print(f"Total wall time: {sim_wall_time:.3f}s")

    # make the trajectory and metrics dictionaries for return
    trajectory = {
        "sim_dt": round(float(sim_dt), 6),
        "ctrl_dt": round(float(step_dt), 6),
        "qpos": qpos_hist,
        "qvel": qvel_hist,
        "ctrl": ctrl_hist,
    }
    metrics = {
        "total_wall_time": sim_wall_time,
        "termination_time": duration,
        **metrics_hist,
    }

    # results dictionary
    results = {
        "trajectory": trajectory,
        "metrics": metrics,
    }

    return results