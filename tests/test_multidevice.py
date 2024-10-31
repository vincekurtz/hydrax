import os

# Use only CPU, which gives us the opportunity to treat each CPU core as a
# device, and avoid the need for multiple GPUs.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import time

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec


@jax.jit
def my_test_function(rng: jax.Array) -> jax.Array:
    """Do some toy compute, kind of like SamplingBasedController.optimize."""
    controls = jax.random.normal(rng, (8192, 60))

    mesh = jax.sharding.Mesh(jax.devices(), ("x"))
    sharding = jax.sharding.NamedSharding(mesh, PartitionSpec("x"))
    controls = jax.lax.with_sharding_constraint(controls, sharding)

    costs = jax.vmap(lambda c: jnp.sum(jnp.square(c)), in_axes=0)(controls)
    return costs


if __name__ == "__main__":
    assert len(jax.devices()) == 8
    rng = jax.random.key(0)
    rng, sample_rng = jax.random.split(rng)

    # Warm up
    costs = my_test_function(sample_rng)
    costs.block_until_ready()
    jax.debug.visualize_array_sharding(costs)

    # Time the jitted function
    st = time.time()
    costs = my_test_function(sample_rng)
    costs.block_until_ready()
    print(f"Time taken: {time.time() - st:.5f}s")
