import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mujoco import mjx

from hydrax.alg_base import Trajectory
from hydrax.algs.dial import DIAL
from hydrax.tasks.pendulum import Pendulum


def test_open_loop() -> None:
    """Use DIAL for open-loop pendulum swingup."""
    # Task and optimizer setup
    task = Pendulum()
    opt = DIAL(
        task,
        num_samples=32,
        noise_level=0.4,
        beta_opt_iter=1.0,
        beta_horizon=1.0,
        temperature=0.001,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=11,
        iterations=10,
    )
    jit_opt = jax.jit(opt.optimize)

    # Initialize the system state and policy parameters
    state = mjx.make_data(task.model)
    params = opt.init_params()

    for _ in range(20):
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
    assert total_cost <= 15.0

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


def test_sample_knots_shape() -> None:
    """Test that sample_knots returns the correct shape and updates params."""
    task = Pendulum()
    opt = DIAL(
        task,
        num_samples=20,
        noise_level=0.4,
        beta_opt_iter=1.5,
        beta_horizon=1.5,
        temperature=0.8,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=8,
    )

    params = opt.init_params(seed=123)
    original_rng = params.rng

    knots, updated_params = opt.sample_knots(params)

    # Check shape
    expected_shape = (opt.num_samples, opt.num_knots, task.model.nu)
    assert knots.shape == expected_shape, (
        f"Expected knots shape {expected_shape}, got {knots.shape}"
    )

    # Check that RNG was updated
    assert not jnp.array_equal(original_rng, updated_params.rng), (
        "RNG should be updated after sampling"
    )

    # Check that other parameters remain unchanged
    assert jnp.array_equal(params.mean, updated_params.mean)
    assert jnp.array_equal(params.tk, updated_params.tk)
    assert params.opt_iteration == updated_params.opt_iteration


def test_opt_iteration() -> None:
    """Test that opt_iteration is properly initialized and updated."""
    task = Pendulum()
    controller = DIAL(
        task,
        num_samples=10,
        noise_level=0.4,
        beta_opt_iter=1.0,
        beta_horizon=1.0,
        temperature=1.0,
        plan_horizon=0.5,
        spline_type="zero",
        num_knots=3,
        iterations=3,
    )

    # Test initial opt_iteration value
    params = controller.init_params()
    assert params.opt_iteration == 0, (
        f"Expected opt_iteration to be 0, got {params.opt_iteration}"
    )

    # Test that opt_iteration is reset after n iterations
    for _ in range(controller.iterations):
        _, params = controller.sample_knots(params)
    assert params.opt_iteration == 0, (
        f"Expected opt_iteration to be 0, got {params.opt_iteration}"
    )

    # Test that opt_iteration is reset after optimization
    state = mjx.make_data(task.model)
    jit_opt = jax.jit(controller.optimize)
    final_params, _ = jit_opt(state, params)
    assert final_params.opt_iteration == 0, (
        f"Expected opt_iteration to be 0, got {final_params.opt_iteration}"
    )


def test_update_params() -> None:
    """Test that update_params correctly updates the mean."""
    task = Pendulum()
    opt = DIAL(
        task,
        num_samples=4,
        noise_level=0.4,
        beta_opt_iter=1.0,
        beta_horizon=1.0,
        temperature=1.0,
        plan_horizon=0.5,
        spline_type="zero",
        num_knots=3,
    )

    params = opt.init_params(seed=456)
    original_mean = params.mean.copy()

    # Create mock rollouts with different costs
    num_samples = opt.num_samples
    num_knots = opt.num_knots
    nu = task.model.nu
    ctrl_steps = opt.ctrl_steps

    # Create some dummy knots and costs
    knots = jax.random.normal(jax.random.key(789), (num_samples, num_knots, nu))
    costs = jnp.array([10.0, 5.0, 15.0, 8.0])  # Different costs for each sample
    controls = jnp.zeros((num_samples, ctrl_steps, nu))
    trace_sites = jnp.zeros((num_samples, ctrl_steps + 1, 3))

    rollouts = Trajectory(
        controls=controls,
        knots=knots,
        costs=jnp.tile(costs[:, None], (1, ctrl_steps + 1)),
        trace_sites=trace_sites,
    )

    updated_params = opt.update_params(params, rollouts)

    # Check that mean was updated
    assert not jnp.array_equal(original_mean, updated_params.mean), (
        "Mean should be updated after calling update_params"
    )

    # Manually compute expected weighted average
    total_costs = jnp.sum(rollouts.costs, axis=1)
    weights = jax.nn.softmax(-total_costs / opt.temperature, axis=0)
    expected_mean = jnp.sum(weights[:, None, None] * knots, axis=0)

    assert jnp.allclose(updated_params.mean, expected_mean, atol=1e-6), (
        "Updated mean should match manually computed weighted average"
    )


if __name__ == "__main__":
    test_open_loop()
    test_sample_knots_shape()
    test_update_params()
    test_opt_iteration()
