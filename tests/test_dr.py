from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.algs import PredictiveSampling
from hydrax.tasks.particle import Particle


def domain_randomize(model: mjx.Model, rng: jax.Array) -> Dict[str, jax.Array]:
    """Generate randomized model parameters."""
    rng, friction_rng, actuator_rng = jax.random.split(rng, 3)

    # Randomize friction
    new_frictions = jax.random.uniform(
        friction_rng, model.geom_friction[:, 0].shape, minval=0.1, maxval=2.0
    )
    new_frictions = model.geom_friction.at[:, 0].set(new_frictions)

    new_gains = (
        jax.random.uniform(
            actuator_rng,
            model.actuator_gainprm[:, 0].shape,
            minval=0.1,
            maxval=2.0,
        )
        + model.actuator_gainprm[:, 0]
    )
    new_gains = model.actuator_gainprm.at[:, 0].set(new_gains)

    return {"geom_friction": new_frictions, "actuator_gainprm": new_gains}


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
    num_randomizations = 3
    rng, subrng = jax.random.split(rng)
    randomization = jax.vmap(domain_randomize, in_axes=(None, 0))(
        model, jax.random.split(subrng, num_randomizations)
    )
    batch_model, in_axes = apply_randomization(model, randomization)

    assert batch_model.geom_friction.shape[0] == num_randomizations
    assert batch_model.geom_friction.shape[1:] == model.geom_friction.shape

    # Step all the randomized models
    data = mjx.make_data(model)
    data = data.tree_replace({"ctrl": jnp.ones(model.nu)})
    next_data = mjx.step(model, data)
    print(next_data.qpos.shape)

    batch_data = jax.vmap(lambda _, x: x, in_axes=(0, None))(
        jnp.arange(num_randomizations), data
    )

    next_batch_data = jax.vmap(mjx.step, in_axes=(in_axes, 0))(
        batch_model, batch_data
    )
    print(batch_data.qpos.shape)
    print(next_batch_data.qpos.shape)

    print(next_batch_data.qpos)


def test_opt() -> None:
    """Test optimization with domain randomization for the particle."""
    task = Particle()
    ctrl = PredictiveSampling(
        task, num_samples=10, noise_level=0.1, num_randomizations=3
    )
    params = ctrl.init_params()

    # Create a random initial state
    state = mjx.make_data(task.model)
    state = state.replace(mocap_pos=jnp.array([[0.5, 0.5, 0.0]]))
    assert state.qpos.shape == (2,)

    # Run an optimization step
    params, rollouts = ctrl.optimize(state, params)

    # Check the rollout shapes. Should be
    # (randomizations, samples, timestep, ...)
    assert rollouts.costs.shape == (3, 11, 5)
    assert rollouts.controls.shape == (3, 11, 4, 2)
    assert rollouts.observations.shape == (3, 11, 5, 4)

    # Check the updated parameters
    assert params.mean.shape == (4, 2)

    # Check that the rollout costs are different across different models
    costs = jnp.sum(rollouts.costs, axis=(1, 2))
    assert not jnp.allclose(costs[0], costs[1])


if __name__ == "__main__":
    # test_domain_randomization()
    test_opt()
