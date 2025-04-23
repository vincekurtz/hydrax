import jax.numpy as jnp

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
    assert params.mean.shape == (task.model.nu * controller.num_knots,)
    assert params.cov.shape == (task.model.nu * controller.num_knots,)
    assert params.rng.shape == ()
    assert params.tk.shape == (controller.num_knots,)

    # Test with initial control
    initial_control = jnp.rand((task.model.nu * controller.num_knots,))
    params = controller.init_params(
        initial_control=jnp.ones((task.model.nu * controller.num_knots,))
    )
    assert params.mean.shape == (task.model.nu * controller.num_knots,)
    assert params.cov.shape == (task.model.nu * controller.num_knots,)
    assert params.rng.shape == ()
    assert params.tk.shape == (controller.num_knots,)
    assert jnp.all(params.mean == initial_control)


if __name__ == "__main__":
    test_traj()
