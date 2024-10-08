import jax.numpy as jnp

from hydra.utils import Trajectory


def test_traj() -> None:
    """Make sure we can define a Trajectory object."""
    batch, horizon = 5, 10
    U = jnp.zeros((batch, horizon - 1, 3))
    Y = jnp.zeros((batch, horizon, 4))
    J = jnp.zeros((batch, horizon))

    traj = Trajectory(controls=U, costs=J, observations=Y)
    assert len(traj) == horizon


if __name__ == "__main__":
    test_traj()
