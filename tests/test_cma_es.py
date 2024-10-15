import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mujoco import mjx

from hydrax.algs.cma_es import CMAES
from hydrax.tasks.pendulum import Pendulum


def test_cmaes() -> None:
    """Test the CMAES algorithm."""
    task = Pendulum()
    ctrl = CMAES(task, num_samples=32)

    # Initialize the policy parameters
    params = ctrl.init_params()
    assert params.opt_state.C.shape == (19, 19)
    assert params.opt_state.weights.shape == (32,)

    # Sample control sequences from the policy
    controls, params = ctrl.sample_controls(params)
    assert controls.shape == (32, 19, 1)

    # Roll out the control sequences
    state = mjx.make_data(task.model)
    rollouts = ctrl.eval_rollouts(state, controls)
    assert rollouts.costs.shape == (32, 20)

    # Update the policy parameters
    params = ctrl.update_params(params, rollouts)
    assert params.controls.shape == (19, 1)
    assert jnp.all(params.controls != jnp.zeros((19, 1)))
    assert params.opt_state.best_fitness > 0.0


def test_open_loop() -> None:
    """Use CMA-ES for open-loop optimization."""
    # Task and optimizer setup
    task = Pendulum()
    opt = CMAES(task, num_samples=32, elite_ratio=0.1)
    jit_opt = jax.jit(opt.optimize)

    # Initialize the system state and policy parameters
    state = mjx.make_data(task.model)
    params = opt.init_params()

    for _ in range(100):
        # Do an optimization step
        params, _ = jit_opt(state, params)

    # Pick the best rollout
    best_cost = params.opt_state.best_fitness
    best_ctrl = params.controls
    final_rollout = jax.jit(opt.eval_rollouts)(state, best_ctrl[None])
    assert jnp.allclose(best_cost, jnp.sum(final_rollout.costs[0]))

    if __name__ == "__main__":
        # Plot the solution
        _, ax = plt.subplots(3, 1, sharex=True)
        times = jnp.arange(task.planning_horizon) * task.dt

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
    test_cmaes()
    test_open_loop()
