import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mujoco import mjx

from hydrax.algs.predictive_sampling import PredictiveSampling
from hydrax.tasks.pendulum import Pendulum


def test_predictive_sampling() -> None:
    """Test the PredictiveSampling algorithm."""
    task = Pendulum()
    opt = PredictiveSampling(
        task,
        num_samples=32,
        noise_level=0.1,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=11,
    )

    # Initialize the policy parameters
    params = opt.init_params()
    assert params.mean.shape == (opt.num_knots, 1)
    assert isinstance(params.rng, jax._src.prng.PRNGKeyArray)

    # Sample control sequences from the policy
    knots, new_params = opt.sample_knots(params)
    tk = jnp.linspace(0.0, opt.plan_horizon, opt.num_knots)
    tq = jnp.linspace(0.0, opt.plan_horizon - opt.dt, opt.ctrl_steps)
    controls = opt.interp_func(tq, tk, knots)
    assert controls.shape == (opt.num_samples, opt.ctrl_steps, 1)
    assert knots.shape == (opt.num_samples, opt.num_knots, 1)
    assert new_params.rng != params.rng

    # Roll out the control sequences
    state = mjx.make_data(task.model)
    _, rollouts = opt.eval_rollouts(task.model, state, controls, knots)

    assert rollouts.costs.shape == (
        opt.num_samples,
        opt.ctrl_steps + 1,
    )
    assert rollouts.controls.shape == (
        opt.num_samples,
        opt.ctrl_steps,
        1,
    )
    assert rollouts.knots.shape == (
        opt.num_samples,
        opt.num_knots,
        1,
    )
    assert rollouts.trace_sites.shape == (
        opt.num_samples,
        opt.ctrl_steps + 1,
        len(task.trace_site_ids),
        3,
    )

    # Pick the best rollout
    updated_params = opt.update_params(new_params, rollouts)
    assert updated_params.mean.shape == (opt.num_knots, 1)
    assert jnp.all(updated_params.mean != new_params.mean)


def test_open_loop() -> None:
    """Use predictive sampling for open-loop optimization."""
    # Task and optimizer setup
    task = Pendulum()
    opt = PredictiveSampling(
        task,
        num_samples=32,
        noise_level=0.1,
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

    # Pick the best rollout (first axis is for domain randomization, unused)
    total_costs = jnp.sum(rollouts.costs, axis=1)
    best_idx = jnp.argmin(total_costs)
    best_ctrl = rollouts.controls[best_idx]
    best_knots = rollouts.knots[best_idx]
    assert total_costs[best_idx] <= 9.0

    states, _ = jax.jit(opt.eval_rollouts)(
        task.model, state, best_ctrl[None], best_knots[None]
    )

    if __name__ == "__main__":
        # Plot the solution
        _, ax = plt.subplots(3, 1, sharex=True)
        times = jnp.arange(opt.ctrl_steps) * task.dt

        ax[0].plot(times, states.qpos[0, :, 0])
        ax[0].set_ylabel(r"$\theta$")

        ax[1].plot(times, states.qvel[0, :, 1])
        ax[1].set_ylabel(r"$\dot{\theta}$")

        ax[2].step(times, best_ctrl, where="post")
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


if __name__ == "__main__":
    test_predictive_sampling()
    test_open_loop()
