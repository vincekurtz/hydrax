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
        task, num_samples=32, num_elites=4, sigma_start=1.0, sigma_min=0.1
    )
    jit_opt = jax.jit(opt.optimize)

    # Initialize the system state and policy parameters
    state = mjx.make_data(task.model)
    params = opt.init_params()

    for _ in range(100):
        # Do an optimization step
        params, _ = jit_opt(state, params)

    # Roll out the solution, check that it's good enough
    final_rollout = jax.jit(opt.eval_rollouts)(
        task.model, state, params.mean[None]
    )
    total_cost = jnp.sum(final_rollout.costs[0])
    assert total_cost <= 9.0
    assert jnp.all(params.cov >= opt.sigma_min)

    if __name__ == "__main__":
        # Plot the solution
        _, ax = plt.subplots(3, 1, sharex=True)
        times = jnp.arange(task.planning_horizon + 1) * task.dt

        ax[0].plot(times, final_rollout.observations[0, :, 0])
        ax[0].set_ylabel(r"$\theta$")

        ax[1].plot(times, final_rollout.observations[0, :, 1])
        ax[1].set_ylabel(r"$\dot{\theta}$")

        ax[2].step(times[0:-1], final_rollout.controls[0], where="post")
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
