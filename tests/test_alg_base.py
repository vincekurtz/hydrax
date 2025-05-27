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


def test_opt_iteration() -> None:
    """Test that opt_iteration is properly initialized and updated during optimization."""
    task = Particle()
    controller = CEM(
        task,
        num_samples=10,
        num_elites=5,
        sigma_start=1.0,
        sigma_min=0.1,
        iterations=3,
    )

    # Test initial opt_iteration value
    params = controller.init_params()
    assert params.opt_iteration == 0, (
        f"Expected opt_iteration to be 0, got {params.opt_iteration}"
    )

    # Test that opt_iteration is reset during optimize
    from mujoco import mjx

    state = mjx.make_data(task.model)

    # Manually set opt_iteration to a non-zero value to test reset
    params = params.replace(opt_iteration=5)

    # Run optimization and check that opt_iteration gets reset and incremented
    jit_opt = jax.jit(controller.optimize)
    final_params, _ = jit_opt(state, params)

    # After optimization, opt_iteration should equal the number of iterations
    assert final_params.opt_iteration == controller.iterations, (
        f"Expected opt_iteration to be {controller.iterations} after optimization, "
        f"got {final_params.opt_iteration}"
    )


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
    test_opt_iteration()
    test_get_action()
