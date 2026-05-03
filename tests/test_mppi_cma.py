import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mujoco import mjx

from hydrax.algs.mppi_cma import MppiCma
from hydrax.tasks.particle import Particle
from hydrax.tasks.pendulum import Pendulum


def test_params_update() -> None:
    """Test that the MPPICMA parameter update works."""
    # Task and optimizer setup
    task = Pendulum()
    opt = MppiCma(
        task,
        num_samples=32,
        initial_noise_level=0.1,
        covariance_adaptation_rate=0.1,
        temperature=0.01,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=11,
    )

    # Initialize the system state and policy parameters
    params = opt.init_params()

    assert params.mean.shape == (opt.num_knots, task.model.nu)
    assert params.covariance.shape == (
        opt.num_knots,
        task.model.nu,
        task.model.nu,
    )

    # Sample from the action distribution, check shapes
    knots, params = opt.sample_knots(params)
    assert knots.shape == (opt.num_samples, opt.num_knots, task.model.nu)

    # Collect some rollouts
    state = mjx.make_data(task.model)
    new_params, rollouts = opt.optimize(state, params)

    assert new_params.mean.shape == (opt.num_knots, task.model.nu)
    assert new_params.covariance.shape == (
        opt.num_knots,
        task.model.nu,
        task.model.nu,
    )
    assert not jnp.allclose(new_params.covariance, params.covariance)


def test_open_loop() -> None:
    """Use MPPICMA for open-loop pendulum swingup."""
    # Task and optimizer setup
    task = Pendulum()
    opt = MppiCma(
        task,
        num_samples=32,
        initial_noise_level=0.1,
        minimum_noise_level=0.05,
        covariance_adaptation_rate=0.1,
        temperature=0.01,
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
        params, _ = jit_opt(state, params)

    knots = params.mean[None]
    tk = jnp.linspace(0.0, opt.plan_horizon, opt.num_knots)
    tq = jnp.linspace(0.0, opt.plan_horizon - opt.dt, opt.ctrl_steps)
    controls = opt.interp_func(tq, tk, knots)

    # Roll out the solution, check that it's good enough
    states, final_rollout = jax.jit(opt.eval_rollouts)(
        task.model, state, controls, knots
    )
    total_cost = jnp.sum(final_rollout.costs[0])
    assert total_cost <= 9.0

    if __name__ == "__main__":
        # Plot the solution
        _, ax = plt.subplots(3, 1, sharex=True)
        times = jnp.arange(opt.ctrl_steps) * task.dt

        ax[0].plot(times, states.qpos[0, :, 0])
        ax[0].set_ylabel(r"$\theta$")

        ax[1].plot(times, states.qvel[0, :, 0])
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


def test_multi_input() -> None:
    """Check that MPPI-CMA performs as expected on a problem with nu > 1."""
    task = Particle()
    opt = MppiCma(
        task,
        num_samples=32,
        initial_noise_level=0.2,
        minimum_noise_level=0.1,
        covariance_adaptation_rate=1.0,
        temperature=0.01,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=11,
    )

    # Initialize the system state and policy parameters
    params = opt.init_params()

    assert params.mean.shape == (opt.num_knots, task.model.nu)
    assert params.covariance.shape == (
        opt.num_knots,
        task.model.nu,
        task.model.nu,
    )

    # Sample from the action distribution, check shapes
    knots, params = opt.sample_knots(params)
    assert knots.shape == (opt.num_samples, opt.num_knots, task.model.nu)

    mu_init = params.mean
    cov_init = params.covariance

    state = mjx.make_data(task.model)
    params, _ = jax.jit(opt.optimize)(state, params)

    mu_final = params.mean
    cov_final = params.covariance

    assert mu_init.shape == mu_final.shape
    assert cov_init.shape == cov_final.shape

    # Initial covariance is diagonal, but final is not
    assert jnp.allclose(
        cov_init, jax.vmap(jnp.diag)(jnp.diagonal(cov_init, axis1=-2, axis2=-1))
    )
    assert not jnp.allclose(
        cov_final,
        jax.vmap(jnp.diag)(jnp.diagonal(cov_final, axis1=-2, axis2=-1)),
    )

    # Final covariance has eigenvalues that are not too small
    tol = 1e-6
    eigvals_final = jnp.linalg.eigvalsh(cov_final)
    assert jnp.all(eigvals_final >= opt.minimum_noise_level**2 - tol)


if __name__ == "__main__":
    test_params_update()
    test_open_loop()
    test_multi_input()
