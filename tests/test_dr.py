import jax.numpy as jnp
from mujoco import mjx

from hydrax.algs import PredictiveSampling
from hydrax.tasks.particle import Particle


def test_domain_randomization() -> None:
    """Test basic domain randomization utilities."""
    task = Particle()
    ctrl1 = PredictiveSampling(
        task, num_samples=10, noise_level=0.1, num_randomizations=1
    )
    ctrl2 = PredictiveSampling(
        task, num_samples=10, noise_level=0.1, num_randomizations=2
    )

    # The models should have different numbers of randomizations
    original_shape = task.model.actuator_gainprm.shape
    assert ctrl1.model.actuator_gainprm.shape == original_shape
    assert ctrl2.model.actuator_gainprm.shape == (2, *original_shape)

    # The randomized parameters should be different from the original model
    assert not jnp.allclose(
        ctrl2.model.actuator_gainprm[0], task.model.actuator_gainprm
    )
    assert jnp.allclose(
        ctrl1.model.actuator_gainprm, task.model.actuator_gainprm
    )


def test_opt() -> None:
    """Test optimization with domain randomization for the particle."""
    task = Particle()
    ctrl = PredictiveSampling(
        task, num_samples=10, noise_level=0.1, num_randomizations=3
    )
    params = ctrl.init_params()

    # Create a random initial state
    state = mjx.make_data(task.model)
    assert state.qpos.shape == (2,)

    # Run an optimization step
    params, rollouts = ctrl.optimize(state, params)

    # Check the rollout shapes. Should be
    # (randomizations, samples, timestep, ...)
    assert rollouts.costs.shape == (3, 11, 6)
    assert rollouts.controls.shape == (3, 11, 5, 2)
    assert rollouts.observations.shape == (3, 11, 6, 4)

    # Check the updated parameters
    assert params.mean.shape == (5, 2)

    # Check that the rollout costs are different across different models
    costs = jnp.sum(rollouts.costs, axis=(1, 2))
    assert not jnp.allclose(costs[0], costs[1])


if __name__ == "__main__":
    test_domain_randomization()
    test_opt()
