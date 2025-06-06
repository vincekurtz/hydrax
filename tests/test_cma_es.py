from evosax.algorithms.distribution_based.cma_es import CMA_ES
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mujoco import mjx

from hydrax.algs.evosax import Evosax
from hydrax.tasks.pendulum import Pendulum


def test_cmaes() -> None:
    """Test the CMAES algorithm."""
    task = Pendulum()
    ctrl = Evosax(
        task,
        CMA_ES,
        num_samples=32,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=11,
    )

    # Initialize the policy parameters
    params = ctrl.init_params()
    assert params.opt_state.C.shape == (ctrl.num_knots, ctrl.num_knots)
    assert ctrl.es_params.weights.shape == (32,) # weights in evosax 0.2.0 stay in params

    # Sample control sequences from the policy
    knots, params = ctrl.sample_knots(params)
    assert knots.shape == (32, ctrl.num_knots, 1)

    tk = jnp.linspace(0.0, ctrl.plan_horizon, ctrl.num_knots)
    tq = jnp.linspace(0.0, ctrl.plan_horizon - ctrl.dt, ctrl.ctrl_steps)
    controls = ctrl.interp_func(tq, tk, knots)

    # Roll out the control sequences
    state = mjx.make_data(task.model)
    _, rollouts = ctrl.eval_rollouts(task.model, state, controls, knots)
    assert rollouts.costs.shape == (32, ctrl.ctrl_steps + 1)

    # Update the policy parameters
    params = ctrl.update_params(params, rollouts)
    assert params.mean.shape == (ctrl.num_knots, 1)
    assert jnp.all(params.mean != jnp.zeros((ctrl.num_knots, 1)))
    assert params.opt_state.best_fitness > 0.0


def test_open_loop() -> None:
    """Use CMA-ES for open-loop optimization."""
    # Task and optimizer setup
    task = Pendulum()
    opt = Evosax(
        task,
        CMA_ES,
        num_samples=32,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=11,
    )

    # elite_ratio was not an argument of the constructor in exosax 0.2.0, it was hard-coded
    opt.elite_ratio = 0.1

    jit_opt = jax.jit(opt.optimize)

    # Initialize the system state and policy parameters
    state = mjx.make_data(task.model)
    params = opt.init_params()

    for _ in range(100):
        # Do an optimization step
        params, final_rollout = jit_opt(state, params)

    # Test consistency of best rollout identification
    best_cost = params.opt_state.best_fitness
    final_costs = jnp.sum(final_rollout.costs, axis=-1)
    best_idx = jnp.argmin(final_costs)
    assert jnp.allclose(best_cost, final_costs[best_idx])

    # rollout the best control sequence and update it once more
    best_knots = params.mean[None]
    tk = jnp.linspace(0.0, opt.plan_horizon, opt.num_knots)
    tq = jnp.linspace(0.0, opt.plan_horizon - opt.dt, opt.ctrl_steps)
    controls = opt.interp_func(tq, tk, best_knots)
    states, final_rollout = jax.jit(opt.eval_rollouts)(
        task.model, state, controls, best_knots
    )

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


if __name__ == "__main__":
    test_cmaes()
    test_open_loop()
