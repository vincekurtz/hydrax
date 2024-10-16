from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT


def domain_randomize(model: mjx.Model, rng: jax.Array) -> Dict[str, jax.Array]:
    """Generate randomized friction parameters."""
    new_frictions = jax.random.uniform(
        rng, model.geom_friction[:, 0].shape, minval=0.1, maxval=2.0
    )
    new_frictions = model.geom_friction.at[:, 0].set(new_frictions)
    return {"geom_friction": new_frictions}


def apply_randomization(
    model: mjx.Model, randomization: Dict[str, jax.Array]
) -> Tuple[mjx.Model, mjx.Model]:
    """Apply a randomization to a model.

    Return both the randomized model and a pytree with the randomized axes.
    """
    new_model = model.tree_replace(randomization)

    in_axes = jax.tree.map(lambda x: None, model)
    in_axes = in_axes.tree_replace(
        {key: 0 for key in randomization.keys()},
    )

    return new_model, in_axes


def test_domain_randomization() -> None:
    """Smoke test for domain randomization."""
    rng = jax.random.key(0)

    # Load a model
    mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/particle/scene.xml")
    model = mjx.put_model(mj_model)

    # Randomize multiple models at once
    rng, subrng = jax.random.split(rng)
    randomization = jax.vmap(domain_randomize, in_axes=(None, 0))(
        model, jax.random.split(subrng, 10)
    )
    batch_model, in_axes = apply_randomization(model, randomization)

    assert batch_model.geom_friction.shape[0] == 10
    assert batch_model.geom_friction.shape[1:] == model.geom_friction.shape

    # Step all the randomized models
    data = mjx.make_data(model)
    next_data = mjx.step(model, data)
    print(next_data.qpos.shape)

    batch_data = jax.vmap(lambda _, x: x, in_axes=(0, None))(
        jnp.arange(10), data
    )

    next_batch_data = jax.vmap(mjx.step, in_axes=(in_axes, 0))(
        batch_model, batch_data
    )
    print(batch_data.qpos.shape)
    print(next_batch_data.qpos.shape)


if __name__ == "__main__":
    test_domain_randomization()
