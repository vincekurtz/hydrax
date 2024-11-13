import jax.numpy as jnp

from hydrax.alg_base import Trajectory


def test_traj() -> None:
    """Make sure we can define a Trajectory object."""
    batch, horizon = 5, 10
    U = jnp.zeros((batch, horizon, 3))
    J = jnp.zeros((batch, horizon + 1))
    P = jnp.zeros((batch, horizon + 1, 0, 3))

    traj = Trajectory(controls=U, costs=J, trace_sites=P)
    assert len(traj) == horizon


if __name__ == "__main__":
    test_traj()
