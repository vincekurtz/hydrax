import os

# Use only CPU, which gives us the opportunity to treat each CPU core as a
# device, and avoid the need for multiple GPUs.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.sharding import PartitionSpec
from mujoco import mjx

from hydrax.algs.predictive_sampling import PredictiveSampling
from hydrax.tasks.pendulum import Pendulum


@jax.jit
def my_toy_function(rng: jax.Array) -> jax.Array:
    """Do some toy compute, kind of like SamplingBasedController.optimize."""
    controls = jax.random.normal(rng, (8192, 60))

    mesh = jax.sharding.Mesh(jax.devices(), ("x"))
    sharding = jax.sharding.NamedSharding(mesh, PartitionSpec("x"))
    controls = jax.lax.with_sharding_constraint(controls, sharding)

    costs = jax.vmap(lambda c: jnp.sum(jnp.square(c)), in_axes=0)(controls)
    return costs


def test_toy_function() -> None:
    """Test multi-device parallelism on a toy function."""
    assert len(jax.devices()) == 8
    rng = jax.random.key(0)
    rng, sample_rng = jax.random.split(rng)

    # Warm up
    costs = my_toy_function(sample_rng)
    costs.block_until_ready()

    # Time the jitted function
    st = time.time()
    costs = my_toy_function(sample_rng)
    costs.block_until_ready()
    print(f"Time taken: {time.time() - st:.5f}s")
    jax.debug.visualize_array_sharding(costs)


def test_open_loop_ps() -> None:
    """Use predictive sampling for open-loop optimization."""
    # Task and optimizer setup
    task = Pendulum()
    opt = PredictiveSampling(task, num_samples=32, noise_level=0.1)
    jit_opt = jax.jit(opt.optimize)

    # Initialize the system state and policy parameters
    state = mjx.make_data(task.model)
    params = opt.init_params()

    for _ in range(100):
        # Do an optimization step
        params, rollouts = jit_opt(state, params)
        jax.block_until_ready(rollouts.costs)

    # Sharding visualization
    print(rollouts.costs.shape)
    jax.debug.visualize_array_sharding(rollouts.costs[0])

    # Pick the best rollout (first axis is for domain randomization, unused)
    total_costs = jnp.sum(rollouts.costs[0], axis=1)
    best_idx = jnp.argmin(total_costs)
    best_obs = rollouts.observations[0, best_idx]
    best_ctrl = rollouts.controls[0, best_idx]
    assert total_costs[best_idx] <= 9.0

    if __name__ == "__main__":
        # Plot the solution
        _, ax = plt.subplots(3, 1, sharex=True)
        times = jnp.arange(task.planning_horizon + 1) * task.dt

        ax[0].plot(times, best_obs[:, 0])
        ax[0].set_ylabel(r"$\theta$")

        ax[1].plot(times, best_obs[:, 1])
        ax[1].set_ylabel(r"$\dot{\theta}$")

        ax[2].step(times[0:-1], best_ctrl, where="post")
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
    # test_toy_function()
    test_open_loop_ps()
