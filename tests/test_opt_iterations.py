import time

import jax
import mujoco
from mujoco import mjx

from hydrax.algs import MPPI
from hydrax.tasks.double_cart_pole import DoubleCartPole


def test_opt_iterations() -> None:
    """Test the optimization iterations feature."""
    task = DoubleCartPole()
    ctrl = MPPI(
        task,
        num_samples=1024,
        num_randomizations=4,
        plan_horizon=1.0,
        noise_level=0.3,
        temperature=0.1,
        spline_type="zero",
        num_knots=10,
        iterations=1,
    )

    # Define the model used for simulation
    mj_model = task.mj_model
    mj_model.opt.timestep = 0.005
    mj_model.opt.iterations = 50
    mj_data = mujoco.MjData(mj_model)

    mjx_data = mjx.put_data(mj_model, mj_data)
    params = ctrl.init_params()

    # Jit
    jit_optimize = jax.jit(ctrl.optimize)

    # Start timer
    start_time = time.perf_counter()

    jit_optimize(mjx_data, params)
    jit_optimize(mjx_data, params)

    # End timer
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(f"\nTwo manual iterations time: {elapsed_time:.4f} seconds")

    ctrl = MPPI(
        task,
        num_samples=1024,
        num_randomizations=4,
        plan_horizon=1.0,
        noise_level=0.3,
        temperature=0.1,
        spline_type="zero",
        num_knots=10,
        iterations=2,
    )

    mjx_data = mjx.put_data(mj_model, mj_data)
    params = ctrl.init_params()

    # Jit
    jit_optimize = jax.jit(ctrl.optimize)

    # Start timer
    start_time = time.perf_counter()

    jit_optimize(mjx_data, params)

    # End timer
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(f"\nTwo iterations time: {elapsed_time:.4f} seconds")
