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
        T=1.0,
        dt=0.1,
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
    tq = jnp.linspace(0.0, opt.T - opt.dt, opt.H)  # ctrl query times
    controls = opt.interp_func(tq, opt.tk, knots)
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
        times = jnp.arange(opt.H) * task.dt

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


if __name__ == "__main__":
    test_open_loop()
