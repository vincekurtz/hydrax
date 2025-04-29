import jax.numpy as jnp
import jax

from hydrax.alg_base import Trajectory
from hydrax.tasks.particle import Particle
from hydrax.algs.cem import CEM


def test_traj() -> None:
    """Make sure we can define a Trajectory object."""
    batch, horizon, num_knots = 5, 10, 3
    U = jnp.zeros((batch, horizon, 3))
    K = jnp.zeros((batch, num_knots, 3))
    J = jnp.zeros((batch, horizon + 1))
    P = jnp.zeros((batch, horizon + 1, 0, 3))

    traj = Trajectory(controls=U, knots=K, costs=J, trace_sites=P)
    assert len(traj) == horizon


def test_init_params() -> None:
    """Make sure we can initialize the policy parameters."""
    task = Particle()
    controller = CEM(
        task, num_samples=10, num_elites=5, sigma_start=1.0, sigma_min=0.1
    )
    params = controller.init_params()
    expected_shape = (
        controller.num_knots,
        task.model.nu,
    )
    assert params.mean.shape == expected_shape
    assert params.cov.shape == expected_shape
    assert params.rng.shape == ()
    assert params.tk.shape == (controller.num_knots,)

    # Test with initial control knots
    key = jax.random.key(0)  # seed
    initial_knots = jax.random.uniform(
        key, shape=(controller.num_knots, task.model.nu)
    )
    params = controller.init_params(initial_knots=initial_knots)
    assert params.mean.shape == expected_shape
    assert params.cov.shape == expected_shape
    assert params.rng.shape == ()
    assert params.tk.shape == (controller.num_knots,)
    assert jnp.all(params.mean == initial_knots)

def test_get_action() -> None:
    """Make sure we can get the action from the policy parameters of the correct shape."""
    task = Particle()
    controller = CEM(
        task, num_samples=10, num_elites=5, sigma_start=1.0, sigma_min=0.1
    )
    params = controller.init_params()
    action = controller.get_action(params, 0)
    expected_shape = task.model.nu
    assert action.shape[0] == expected_shape

if __name__ == "__main__":
    test_traj()
    test_init_params()
    test_get_action()
