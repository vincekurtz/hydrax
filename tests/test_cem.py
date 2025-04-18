import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mujoco import mjx

from hydrax.algs.cem import CEM
from hydrax.tasks.pendulum import Pendulum


def test_open_loop() -> None:
    """Use CEM for open-loop pendulum swingup."""
    # Task and optimizer setup
    task = Pendulum()
    opt = CEM(
        task,
        num_samples=32,
        num_elites=4,
        sigma_start=1.0,
        sigma_min=0.1,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=11,
    )
    jit_opt = jax.jit(opt.optimize)

    # Initialize the system state and policy parameters
    state = mjx.make_data(task.model)
    params = opt.init_params()

    for _ in range(100):
        # Do an optimization step
        params, rollouts = jit_opt(state, params)

    # Roll out the solution, check that it's good enough
    knots = params.mean[None]
    tk = jnp.linspace(0.0, opt.plan_horizon, opt.num_knots)
    tq = jnp.linspace(0.0, opt.plan_horizon - opt.dt, opt.ctrl_steps)
    controls = opt.interp_func(tq, tk, knots)
    states, final_rollout = jax.jit(opt.eval_rollouts)(
        task.model, state, controls, knots
    )
    theta = states.qpos[0, :, 0]
    theta_dot = states.qvel[0, :, 0]

    total_cost = jnp.sum(final_rollout.costs[0])
    assert total_cost <= 9.0
    assert jnp.all(params.cov >= opt.sigma_min)

    if __name__ == "__main__":
        # Plot the solution
        _, ax = plt.subplots(3, 1, sharex=True)
        times = jnp.arange(opt.ctrl_steps) * task.dt

        ax[0].plot(times, theta)
        ax[0].set_ylabel(r"$\theta$")

        ax[1].plot(times, theta_dot)
        ax[1].set_ylabel(r"$\dot{\theta}$")

        ax[2].step(times, final_rollout.controls[0], where="post")
        ax[2].axhline(-1.0, color="black", linestyle="--")
        ax[2].axhline(1.0, color="black", linestyle="--")
        ax[2].set_ylabel("u")
        ax[2].set_xlabel("Time (s)")

        time_samples = jnp.linspace(0, times[-1], 100)
        controls = jax.vmap(opt.get_action, in_axes=(None, 0))(
            params, time_samples
        )
        ax[2].plot(time_samples, controls, color="gray", alpha=0.5)

        plt.show()


def test_explore_fraction() -> None:
    """Unit test for sampling controls with different explore_fraction values.

    This test uses the Pendulum task as a dummy task to verify:
      - The overall controls array shape is correct.
      - The split between main and exploration samples is as expected.
    """
    num_samples = 10
    num_elites = 2
    sigma_start = 1.0
    sigma_min = 0.1

    # Test different fractions: no exploration, partial exploration, full exploration.
    for explore_fraction in [0.0, 0.3, 0.5, 0.75, 1.0]:
        task = Pendulum()
        opt = CEM(
            task=task,
            num_samples=num_samples,
            num_elites=num_elites,
            sigma_start=sigma_start,
            sigma_min=sigma_min,
            explore_fraction=explore_fraction,
            plan_horizon=1.0,
            spline_type="zero",
            num_knots=11,
        )
        params = opt.init_params(seed=42)
        controls, new_params = opt.sample_knots(params)

        # Check the overall shape of the controls array.
        expected_shape = (num_samples, opt.num_knots, task.model.nu)
        assert controls.shape == expected_shape, (
            f"Expected shape {expected_shape} but got {controls.shape} "
            f"for explore_fraction = {explore_fraction}"
        )

        # Calculate expected number of exploration samples.
        num_explore = int(explore_fraction * num_samples)
        num_main = num_samples - num_explore

        # The implementation concatenates main samples first and exploration samples later.
        main_controls = controls[:num_main]
        explore_controls = controls[num_main:]

        # Verify that the main and exploration segments have the correct shapes.
        expected_main_shape = (num_main, opt.num_knots, task.model.nu)
        expected_explore_shape = (
            num_explore,
            opt.num_knots,
            task.model.nu,
        )
        assert main_controls.shape == expected_main_shape, (
            f"Expected main controls shape {expected_main_shape} but got {main_controls.shape} "
            f"for explore_fraction = {explore_fraction}"
        )
        assert explore_controls.shape == expected_explore_shape, (
            f"Expected explore controls shape {expected_explore_shape} but got {explore_controls.shape} "
            f"for explore_fraction = {explore_fraction}"
        )


if __name__ == "__main__":
    test_open_loop()
    test_explore_fraction()
