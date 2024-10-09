import jax.numpy as jnp

from hydra.base import Trajectory


def test_traj() -> None:
    """Make sure we can define a Trajectory object."""
    batch, horizon = 5, 10
    U = jnp.zeros((batch, horizon - 1, 3))
    Y = jnp.zeros((batch, horizon, 4))
    J = jnp.zeros((batch, horizon))
    P = jnp.zeros((batch, horizon, 0, 3))

    traj = Trajectory(controls=U, costs=J, observations=Y, trace_sites=P)
    assert len(traj) == horizon


if __name__ == "__main__":
    test_traj()
